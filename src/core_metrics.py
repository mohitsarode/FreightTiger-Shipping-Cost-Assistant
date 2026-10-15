import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple

def load_and_preprocess_shipments(file_path: Path) -> pd.DataFrame:
    """
    Loads shipment records and standardizes dates and corridors.
    """
    df = pd.read_csv(file_path)
    
    # Ensure shipment_date is datetime
    df['shipment_date'] = pd.to_datetime(df['shipment_date'])
    
    # Calculate Monday of each week (week runs Monday to Sunday)
    # df['shipment_date'].dt.weekday: Monday is 0, Sunday is 6
    df['week_of'] = df['shipment_date'].apply(
        lambda d: d - pd.Timedelta(days=d.weekday())
    ).dt.strftime('%Y-%m-%d')
    
    # Route format: Origin-Destination
    df['route'] = df['origin'].str.strip() + '-' + df['destination'].str.strip()
    df['route_type'] = df['route_type'].str.strip()
    
    # Tonne-km per shipment = quantity_tonnes * distance_km
    df['tonne_km'] = df['quantity_tonnes'] * df['distance_km']
    
    return df

def compute_weekly_metrics(df: pd.DataFrame, rolling_window: int = 8) -> pd.DataFrame:
    """
    Aggregates shipment records by route and week_of, then calculates:
    1. cost_per_tonne_km = total freight cost / total (quantity * distance)
    2. vs_own_history: trailing 8-week rolling average cost per tonne-km (strictly prior weeks)
    3. vs_similar_routes: average cost per tonne-km in that same week across all other routes sharing route_type
    """
    # Group by route, route_type, week_of
    weekly = df.groupby(['route', 'route_type', 'week_of'], as_index=False).agg(
        total_cost=('freight_cost_inr', 'sum'),
        total_tonne_km=('tonne_km', 'sum'),
        shipment_count=('shipment_id', 'count')
    )
    
    # Weighted cost per tonne-km
    weekly['cost_per_tonne_km'] = weekly['total_cost'] / weekly['total_tonne_km']
    
    # Sort chronologically by route and week_of
    weekly = weekly.sort_values(['route', 'week_of']).reset_index(drop=True)
    
    # 1. Trailing rolling average vs. own history (no look-ahead, strictly prior weeks)
    weekly['own_history_avg'] = np.nan
    weekly['prior_weeks_count'] = 0
    
    for route, grp in weekly.groupby('route'):
        indices = grp.index
        costs = grp['cost_per_tonne_km'].values
        for i in range(len(indices)):
            start_idx = max(0, i - rolling_window)
            prior_costs = costs[start_idx:i]
            if len(prior_costs) > 0:
                weekly.loc[indices[i], 'own_history_avg'] = float(np.mean(prior_costs))
                weekly.loc[indices[i], 'prior_weeks_count'] = len(prior_costs)
            else:
                weekly.loc[indices[i], 'own_history_avg'] = np.nan
                weekly.loc[indices[i], 'prior_weeks_count'] = 0

    # 2. Peer average vs. similar routes in that same week (excluding the route itself)
    weekly['peer_avg'] = np.nan
    weekly['peer_routes_count'] = 0
    
    for (route_type, wk), grp in weekly.groupby(['route_type', 'week_of']):
        for idx, row in grp.iterrows():
            other_routes = grp[grp['route'] != row['route']]
            if len(other_routes) > 0:
                weekly.loc[idx, 'peer_avg'] = float(other_routes['cost_per_tonne_km'].mean())
                weekly.loc[idx, 'peer_routes_count'] = len(other_routes)
            else:
                weekly.loc[idx, 'peer_avg'] = np.nan
                weekly.loc[idx, 'peer_routes_count'] = 0

    # Numeric percentage differences
    weekly['pct_vs_own'] = np.where(
        weekly['own_history_avg'].notna() & (weekly['own_history_avg'] > 0),
        (weekly['cost_per_tonne_km'] - weekly['own_history_avg']) / weekly['own_history_avg'] * 100.0,
        np.nan
    )
    
    weekly['pct_vs_peer'] = np.where(
        weekly['peer_avg'].notna() & (weekly['peer_avg'] > 0),
        (weekly['cost_per_tonne_km'] - weekly['peer_avg']) / weekly['peer_avg'] * 100.0,
        np.nan
    )
    
    # Formatted comparison strings matching sample_output_format.csv exactly
    # e.g.: "+35.5% vs this route's past average"
    # e.g.: "+21.0% vs similar-length routes this week"
    def format_own(row):
        val = row['pct_vs_own']
        if pd.isna(val):
            return "N/A (first week - no prior history)"
        return f"{val:+.1f}% vs this route's past average"

    def format_peer(row):
        val = row['pct_vs_peer']
        if pd.isna(val):
            return "N/A (no peer routes available)"
        return f"{val:+.1f}% vs similar-length routes this week"

    weekly['vs_own_history'] = weekly.apply(format_own, axis=1)
    weekly['vs_similar_routes'] = weekly.apply(format_peer, axis=1)
    
    # Formatted cost rounded to 2 decimal places
    weekly['cost_per_tonne_km_display'] = weekly['cost_per_tonne_km'].apply(lambda c: f"{c:.2f}")

    return weekly
