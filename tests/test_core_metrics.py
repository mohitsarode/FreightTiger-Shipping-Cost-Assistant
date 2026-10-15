import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from src.config import DEFAULT_CONFIG
from src.core_metrics import load_and_preprocess_shipments, compute_weekly_metrics

@pytest.fixture(scope="module")
def processed_data():
    raw_df = load_and_preprocess_shipments(DEFAULT_CONFIG.shipment_records_path)
    weekly_df = compute_weekly_metrics(raw_df, rolling_window=8)
    return raw_df, weekly_df

def test_shipment_date_to_monday_grouping(processed_data):
    raw_df, weekly_df = processed_data
    # Verify that all week_of dates are Mondays
    monday_dates = pd.to_datetime(weekly_df['week_of'])
    assert (monday_dates.dt.weekday == 0).all(), "All week_of dates must be Mondays (weekday == 0)"

def test_cost_per_tonne_km_formula(processed_data):
    raw_df, weekly_df = processed_data
    # Pick a route-week and verify manual calculation
    r_target = "Delhi-Jaipur"
    w_target = "2024-11-11"
    
    sub = raw_df[(raw_df['route'] == r_target) & (raw_df['week_of'] == w_target)]
    expected_cost = sub['freight_cost_inr'].sum() / (sub['quantity_tonnes'] * sub['distance_km']).sum()
    
    row = weekly_df[(weekly_df['route'] == r_target) & (weekly_df['week_of'] == w_target)].iloc[0]
    assert np.isclose(row['cost_per_tonne_km'], expected_cost, atol=1e-5)
    assert row['cost_per_tonne_km_display'] == "4.17"

def test_rolling_average_no_lookahead(processed_data):
    _, weekly_df = processed_data
    # Verify strictly prior weeks are used
    for route, grp in weekly_df.groupby('route'):
        grp = grp.sort_values('week_of').reset_index(drop=True)
        # Week 0 should have NaN / 0 prior weeks
        assert pd.isna(grp.loc[0, 'own_history_avg'])
        assert grp.loc[0, 'prior_weeks_count'] == 0
        
        # Week 1 should average only Week 0
        assert np.isclose(grp.loc[1, 'own_history_avg'], grp.loc[0, 'cost_per_tonne_km'])
        assert grp.loc[1, 'prior_weeks_count'] == 1
        
        # Max prior weeks should never exceed 8
        assert (grp['prior_weeks_count'] <= 8).all()

def test_peer_average_excludes_self(processed_data):
    _, weekly_df = processed_data
    for (route_type, wk), grp in weekly_df.groupby(['route_type', 'week_of']):
        for idx, row in grp.iterrows():
            peers = grp[grp['route'] != row['route']]
            if len(peers) > 0:
                expected_peer = peers['cost_per_tonne_km'].mean()
                assert np.isclose(row['peer_avg'], expected_peer, atol=1e-5)
                assert row['peer_routes_count'] == len(peers)

def test_ground_truth_sample_numeric_matches(processed_data):
    _, weekly_df = processed_data
    
    # 1. Delhi-Jaipur 2024-11-11
    dj = weekly_df[(weekly_df['route'] == 'Delhi-Jaipur') & (weekly_df['week_of'] == '2024-11-11')].iloc[0]
    assert dj['cost_per_tonne_km_display'] == "4.17"
    assert dj['vs_own_history'] == "+35.5% vs this route's past average"
    assert dj['vs_similar_routes'] == "+21.0% vs similar-length routes this week"
    
    # 2. Ahmedabad-Mumbai 2025-01-20
    am = weekly_df[(weekly_df['route'] == 'Ahmedabad-Mumbai') & (weekly_df['week_of'] == '2025-01-20')].iloc[0]
    assert am['cost_per_tonne_km_display'] == "3.29"
    assert am['vs_own_history'] == "+29.5% vs this route's past average"
    assert am['vs_similar_routes'] == "+22.5% vs similar-length routes this week"
    
    # 3. Mumbai-Pune 2025-09-15
    mp = weekly_df[(weekly_df['route'] == 'Mumbai-Pune') & (weekly_df['week_of'] == '2025-09-15')].iloc[0]
    assert mp['cost_per_tonne_km_display'] == "3.98"
    assert mp['vs_own_history'] == "+9.2% vs this route's past average"
    assert mp['vs_similar_routes'] == "+23.6% vs similar-length routes this week"
