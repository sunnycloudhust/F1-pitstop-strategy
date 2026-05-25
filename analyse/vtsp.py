"""
vtsp.py — Quick exploratory analysis: Hamilton vs Verstappen vs Alonso
Improvement vs original:
  - Data path configurable at the top instead of hardcoded.
  - Added basic win-count and points summary so the merge is actually used.
"""

import pandas as pd

# ── Config — adjust to your local dataset path ────────────────────────────────
DATA_ROOT = '../formula-1-race-data/versions/116'

# ── Load ──────────────────────────────────────────────────────────────────────
drivers  = pd.read_csv(f'{DATA_ROOT}/drivers.csv')
circuits = pd.read_csv(f'{DATA_ROOT}/circuits.csv')
races    = pd.read_csv(f'{DATA_ROOT}/races.csv')
results  = pd.read_csv(f'{DATA_ROOT}/results.csv')

# ── Filter the three drivers ──────────────────────────────────────────────────
TARGET_SURNAMES = ['Hamilton', 'Verstappen', 'Alonso']

comparing_drivers = drivers[drivers['surname'].isin(TARGET_SURNAMES)]
print("─── Drivers ──────────────────────────────────")
print(comparing_drivers[['driverId', 'forename', 'surname', 'nationality']].to_string(index=False))

# ── Join results with driver info and race info ───────────────────────────────
merged = (
    results
    .merge(comparing_drivers[['driverId', 'forename', 'surname']], on='driverId')
    .merge(races[['raceId', 'year', 'name']], on='raceId')
)

merged['points'] = pd.to_numeric(merged['points'], errors='coerce').fillna(0)
merged['wins']   = (merged['positionOrder'] == 1).astype(int)

# ── Summary ───────────────────────────────────────────────────────────────────
summary = (
    merged
    .groupby('surname')
    .agg(
        races  = ('raceId', 'count'),
        wins   = ('wins',   'sum'),
        points = ('points', 'sum'),
    )
    .sort_values('points', ascending=False)
)

print("\n─── Career Summary (dataset range) ───────────")
print(summary.to_string())