import pandas as pd
from typing import Tuple, List, Dict, Any, Optional
from src.vector_store import ContextNotesStore
from src.guardrails import AntiHallucinationGuardrail, GuardrailResult
from src.explainer import AnomalyExplainer
from src.config import PipelineConfig, DEFAULT_CONFIG

class RouteCostAnomalyDetector:
    """
    Core Intelligence Engine:
    Identifies rising/out-of-the-ordinary route costs, retrieves context notes,
    applies anti-hallucination guardrails, and produces structured decisions.
    """
    
    def __init__(
        self,
        notes_store: ContextNotesStore,
        explainer: AnomalyExplainer,
        config: Optional[PipelineConfig] = None
    ):
        self.notes_store = notes_store
        self.explainer = explainer
        self.config = config or DEFAULT_CONFIG
        
    def is_candidate_anomaly(self, row: pd.Series) -> bool:
        """
        Flag rule: 'cost is rising and looks out of the ordinary compared to its own history or to similar routes.'
        1. Cost is rising: pct_vs_own > 0 (or rising vs recent history)
        2. Out of ordinary: pct_vs_own >= own_threshold OR pct_vs_peer >= peer_threshold
        """
        pct_own = row['pct_vs_own']
        pct_peer = row['pct_vs_peer']
        
        if pd.isna(pct_own) and pd.isna(pct_peer):
            return False
            
        # Is cost rising?
        cost_rising = (pd.notna(pct_own) and pct_own > self.config.min_own_rise_pct)
        
        # Is it out of ordinary vs own history?
        unusual_own = (pd.notna(pct_own) and pct_own >= self.config.own_history_threshold_pct)
        
        # Is it out of ordinary vs peer routes?
        unusual_peer = (pd.notna(pct_peer) and pct_peer >= self.config.peer_threshold_pct)
        
        # If cost is rising AND (unusual vs own history OR unusual vs similar routes)
        return bool(cost_rising and (unusual_own or unusual_peer))

    def evaluate_route_week(self, row: pd.Series) -> Dict[str, Any]:
        """
        Evaluates a single route-week record.
        """
        route = row['route']
        week_of = row['week_of']
        cost_display = f"{row['cost_per_tonne_km']:.2f}"
        vs_own_str = row['vs_own_history']
        vs_peer_str = row['vs_similar_routes']
        
        is_candidate = self.is_candidate_anomaly(row)
        
        if not is_candidate:
            return {
                "route": route,
                "week_of": week_of,
                "cost_per_tonne_km": cost_display,
                "vs_own_history": vs_own_str,
                "vs_similar_routes": vs_peer_str,
                "flagged": "No",
                "matched_note_id": "",
                "reason": self.explainer.generate_normal_explanation(route, week_of),
                "is_candidate": False
            }
            
        # For candidate anomalies, retrieve notes and verify with guardrails
        retrieved_notes = self.notes_store.retrieve(route=route, week_of=week_of, max_results=3)
        guardrail_result = AntiHallucinationGuardrail.evaluate(
            route=route,
            week_of=week_of,
            retrieved_notes=retrieved_notes
        )
        
        # Decision:
        # If justified: flagged = "No (justified)", matched_note_id = note_id
        # If unexplained: flagged = "Yes", matched_note_id = ""
        if guardrail_result.is_justified:
            flagged = "No (justified)"
            matched_note_id = guardrail_result.matched_note_id or ""
        else:
            flagged = "Yes"
            matched_note_id = ""
            
        reason = self.explainer.generate_explanation(
            route=route,
            week_of=week_of,
            cost_per_tkm=row['cost_per_tonne_km'],
            vs_own_str=vs_own_str,
            vs_peer_str=vs_peer_str,
            guardrail_result=guardrail_result
        )
        
        return {
            "route": route,
            "week_of": week_of,
            "cost_per_tonne_km": cost_display,
            "vs_own_history": vs_own_str,
            "vs_similar_routes": vs_peer_str,
            "flagged": flagged,
            "matched_note_id": matched_note_id,
            "reason": reason,
            "is_candidate": True
        }

    def process_weekly_records(self, weekly_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Processes all records and produces:
        1. candidate_df: records where cost rose and was investigated (matching sample_output_format.csv)
        2. all_df: all records across the full dataset
        """
        all_results = []
        for _, row in weekly_df.iterrows():
            res = self.evaluate_route_week(row)
            all_results.append(res)
            
        all_df = pd.DataFrame(all_results)
        
        # Contract columns in sample_output_format.csv
        contract_cols = [
            "route",
            "week_of",
            "cost_per_tonne_km",
            "vs_own_history",
            "vs_similar_routes",
            "flagged",
            "matched_note_id",
            "reason"
        ]
        
        candidate_df = all_df[all_df['is_candidate'] == True][contract_cols].reset_index(drop=True)
        all_df = all_df[contract_cols].reset_index(drop=True)
        
        return candidate_df, all_df
