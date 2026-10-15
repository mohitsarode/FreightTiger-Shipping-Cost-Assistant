import csv
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional
from src.config import PipelineConfig, DEFAULT_CONFIG
from src.core_metrics import load_and_preprocess_shipments, compute_weekly_metrics
from src.vector_store import ContextNotesStore
from src.token_tracker import TokenTracker
from src.explainer import AnomalyExplainer
from src.anomaly_detector import RouteCostAnomalyDetector

class ShippingCostPipeline:
    """
    End-to-End Pipeline for FreightTiger Cost Monitoring Assistant.
    """
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or DEFAULT_CONFIG
        self.token_tracker = TokenTracker(
            input_rate_per_million=self.config.input_cost_per_million,
            output_rate_per_million=self.config.output_cost_per_million
        )
        self.notes_store = ContextNotesStore(self.config.context_notes_path)
        self.explainer = AnomalyExplainer(token_tracker=self.token_tracker)
        self.detector = RouteCostAnomalyDetector(
            notes_store=self.notes_store,
            explainer=self.explainer,
            config=self.config
        )
        
    def run(self) -> Dict[str, Any]:
        """
        Executes the full pipeline:
        1. Preprocesses raw shipment data
        2. Aggregates weekly cost/tonne-km metrics and baselines
        3. Identifies spikes, performs RAG retrieval, runs anti-hallucination guardrails
        4. Writes compliant output CSVs and token/cost reports
        """
        print(f"Loading shipment records from: {self.config.shipment_records_path}")
        raw_df = load_and_preprocess_shipments(self.config.shipment_records_path)
        print(f"Loaded {len(raw_df)} shipment records across {raw_df['route'].nunique()} routes.")
        
        print("Computing weekly cost per tonne-km and comparison baselines...")
        weekly_df = compute_weekly_metrics(raw_df, rolling_window=self.config.rolling_weeks_window)
        print(f"Generated {len(weekly_df)} route-week records.")
        
        print("Running RAG retrieval, guardrails, and anomaly decision engine...")
        anomalies_df, all_weeks_df = self.detector.process_weekly_records(weekly_df)
        print(f"Identified and evaluated {len(anomalies_df)} candidate route anomalies.")
        
        # Save output CSV matching sample_output_format.csv exactly
        # Note: quoting=csv.QUOTE_MINIMAL ensures commas in reason prose are properly quoted
        print(f"Writing primary output CSV to: {self.config.output_path}")
        anomalies_df.to_csv(self.config.output_path, index=False, quoting=csv.QUOTE_MINIMAL)
        
        print(f"Writing all-weeks CSV to: {self.config.output_all_weeks_path}")
        all_weeks_df.to_csv(self.config.output_all_weeks_path, index=False, quoting=csv.QUOTE_MINIMAL)
        
        # Save token & cost log
        print(f"Writing token and cost log to: {self.config.token_cost_log_path}")
        self.token_tracker.save_log(self.config.token_cost_log_path)
        
        summary = {
            "total_shipments_processed": len(raw_df),
            "total_route_weeks": len(weekly_df),
            "candidate_anomalies_evaluated": len(anomalies_df),
            "flagged_unexplained_yes": int((anomalies_df['flagged'] == 'Yes').sum()),
            "flagged_justified_no": int((anomalies_df['flagged'] == 'No (justified)').sum()),
            "token_and_cost": self.token_tracker.get_summary(),
            "output_files": {
                "primary_anomalies_csv": str(self.config.output_path),
                "all_weeks_csv": str(self.config.output_all_weeks_path),
                "token_log": str(self.config.token_cost_log_path)
            }
        }
        return summary
