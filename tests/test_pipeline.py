import pytest
import pandas as pd
from pathlib import Path
from src.config import DEFAULT_CONFIG
from src.pipeline import ShippingCostPipeline

def test_full_pipeline_run():
    pipeline = ShippingCostPipeline(DEFAULT_CONFIG)
    summary = pipeline.run()
    
    assert summary['total_shipments_processed'] == 2940
    assert summary['total_route_weeks'] == 728
    assert summary['candidate_anomalies_evaluated'] > 0
    
    # Check output file exists
    assert DEFAULT_CONFIG.output_path.exists()
    out_df = pd.read_csv(DEFAULT_CONFIG.output_path)
    
    # Check schema
    expected_cols = [
        "route", "week_of", "cost_per_tonne_km", "vs_own_history",
        "vs_similar_routes", "flagged", "matched_note_id", "reason"
    ]
    assert list(out_df.columns) == expected_cols
    
    # Check that flagged column only has valid values
    valid_flags = {"Yes", "No (justified)"}
    assert set(out_df['flagged']).issubset(valid_flags)
    
    # Check that justified rows have non-empty matched_note_id
    justified = out_df[out_df['flagged'] == 'No (justified)']
    assert (justified['matched_note_id'].notna() & (justified['matched_note_id'] != "")).all()
    
    # Check that unexplained rows have blank matched_note_id
    unexplained = out_df[out_df['flagged'] == 'Yes']
    assert (unexplained['matched_note_id'].isna() | (unexplained['matched_note_id'] == "")).all()
