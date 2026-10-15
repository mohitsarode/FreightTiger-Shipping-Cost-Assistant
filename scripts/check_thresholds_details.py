import pandas as pd
import numpy as np

df = pd.read_csv('AI Intern Case Study/shipment_records.csv')
df['shipment_date'] = pd.to_datetime(df['shipment_date'])
df['week_of'] = df['shipment_date'].apply(lambda d: d - pd.Timedelta(days=d.weekday())).dt.strftime('%Y-%m-%d')
df['route'] = df['origin'] + '-' + df['destination']
df['tonne_km'] = df['quantity_tonnes'] * df['distance_km']

weekly = df.groupby(['route', 'route_type', 'week_of']).agg(
    total_cost=('freight_cost_inr', 'sum'),
    total_tonne_km=('tonne_km', 'sum')
).reset_index()
weekly['cost_per_tonne_km'] = weekly['total_cost'] / weekly['total_tonne_km']
weekly = weekly.sort_values(['route', 'week_of']).reset_index(drop=True)

# Trailing 8 weeks
weekly['own_history_avg'] = np.nan
for route, grp in weekly.groupby('route'):
    indices = grp.index
    costs = grp['cost_per_tonne_km'].values
    for i in range(len(indices)):
        start_idx = max(0, i - 8)
        prior_costs = costs[start_idx:i]
        if len(prior_costs) > 0:
            weekly.loc[indices[i], 'own_history_avg'] = prior_costs.mean()

# Peer average
weekly['peer_avg'] = np.nan
for (route_type, wk), grp in weekly.groupby(['route_type', 'week_of']):
    for idx, row in grp.iterrows():
        other_routes = grp[grp['route'] != row['route']]
        if len(other_routes) > 0:
            weekly.loc[idx, 'peer_avg'] = other_routes['cost_per_tonne_km'].mean()

weekly['pct_vs_own'] = (weekly['cost_per_tonne_km'] - weekly['own_history_avg']) / weekly['own_history_avg'] * 100
weekly['pct_vs_peer'] = (weekly['cost_per_tonne_km'] - weekly['peer_avg']) / weekly['peer_avg'] * 100

print("=== CANDIDATES WITH pct_vs_own >= 20% OR pct_vs_peer >= 20% ===")
cands20 = weekly[(weekly['pct_vs_own'] >= 20.0) | (weekly['pct_vs_peer'] >= 20.0)].sort_values('week_of')
print(f"Total: {len(cands20)}")
for _, r in cands20.iterrows():
    print(f"{r['week_of']} | {r['route']:<20} | cost={r['cost_per_tonne_km']:.2f} | vs_own={r['pct_vs_own']:+6.1f}% | vs_peer={r['pct_vs_peer']:+6.1f}%")

print("\n=== CANDIDATES WITH pct_vs_own >= 10% OR pct_vs_peer >= 20% ===")
cands10 = weekly[(weekly['pct_vs_own'] >= 10.0) | (weekly['pct_vs_peer'] >= 20.0)].sort_values('week_of')
print(f"Total: {len(cands10)}")
for _, r in cands10.iterrows():
    print(f"{r['week_of']} | {r['route']:<20} | cost={r['cost_per_tonne_km']:.2f} | vs_own={r['pct_vs_own']:+6.1f}% | vs_peer={r['pct_vs_peer']:+6.1f}%")
