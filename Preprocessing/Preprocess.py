"""
Preprocess.py — Feature engineering + dataset assembly pipeline
Improvements vs original:
  - Configurable paths at the top (no hardcoded Windows paths in functions).
  - `preprocess_telemetry` sorts by LapNumber before computing diff/cumsum
    so delta_laptime and CumulativeTimeStint are computed in lap order.
  - `combine_race_data` validates Year is actually numeric before casting.
  - Added basic dtype report after combining.
  - `tire_degradation_optimized` returns a cleaner result (kept logic unchanged).
"""

import os
import pandas as pd

# ── Config ────────────────────────────────────────────────────────────────────
RAW_DATA_ROOT       = './f1_data_csv_export'
PROCESSED_DATA_ROOT = './Preprocessed'
OUTPUT_CSV          = 'all_data.csv'

VALID_COMPOUNDS = ['Soft', 'Medium', 'Hard']
DROP_COMPOUNDS  = ['Wet', 'Intermediate', 'UNKNOWN']

REQUIRED_COLUMNS = [
    'DriverNumber', 'Driver', 'Team', 'Compound',
    'LapTime_Seconds', 'Position', 'LapNumber', 'Stint', 'TyreLife',
    'TrackStatus', 'delta_laptime', 'CumulativeTimeStint',
    'race_progress_fraction', 'HasPitStop',
]

TIME_COLUMNS = [
    'LapTime', 'Sector1Time', 'Sector2Time', 'Sector3Time',
    'Sector1SessionTime', 'Sector2SessionTime', 'Sector3SessionTime',
]

# ── Helpers ───────────────────────────────────────────────────────────────────

def drop_na_wet(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with missing Compound or wet/intermediate tyres."""
    df = df.dropna(subset=['Compound'])
    df = df[~df['Compound'].isin(DROP_COMPOUNDS)]
    return df


def preprocess_telemetry(df: pd.DataFrame) -> pd.DataFrame:
    """Feature engineering for a single race lap DataFrame."""
    df = drop_na_wet(df)

    # Convert timedelta strings → seconds
    for col in TIME_COLUMNS:
        if col in df.columns:
            df[f'{col}_Seconds'] = pd.to_timedelta(df[col], errors='coerce').dt.total_seconds()

    # Target label
    df['HasPitStop'] = df['PitInTime'].notna().astype(int)

    # Sort within each driver so diff/cumsum are in lap order
    df = df.sort_values(['Driver', 'LapNumber'])

    df['delta_laptime']       = df.groupby('Driver')['LapTime_Seconds'].diff().fillna(0)
    df['CumulativeTimeStint'] = df.groupby(['Driver', 'Stint'])['LapTime_Seconds'].cumsum()

    max_laps = df['LapNumber'].max()
    df['race_progress_fraction'] = df['LapNumber'] / max_laps if max_laps else 0.0

    # Fill missing track status with mode (fallback 1.0)
    mode_vals = df['TrackStatus'].mode()
    df['TrackStatus'] = df['TrackStatus'].fillna(mode_vals.iloc[0] if not mode_vals.empty else 1.0)

    return df


def process_all_files(input_root: str, output_root: str) -> None:
    """Preprocess every *_Laps.csv and write *_Laps_processed.csv."""
    for folder in os.listdir(input_root):
        folder_path = os.path.join(input_root, folder)
        if not os.path.isdir(folder_path):
            continue

        dest_folder = os.path.join(output_root, folder)
        os.makedirs(dest_folder, exist_ok=True)

        for file in os.listdir(folder_path):
            if not file.endswith('_Laps.csv'):
                continue

            print(f"  Processing: {folder}/{file}")
            file_path = os.path.join(folder_path, file)
            df = pd.read_csv(file_path, low_memory=False)

            processed = preprocess_telemetry(df)

            out_name = file.replace('_Laps.csv', '_Laps_processed.csv')
            processed.to_csv(os.path.join(dest_folder, out_name), index=False)


def combine_race_data(input_root: str, output_filename: str = OUTPUT_CSV) -> None:
    """Combine all processed CSVs into one master file with Year and RaceID columns."""
    all_dfs = []

    for folder in os.walk(input_root):
        root, _, files = folder
        year_str = os.path.basename(root)

        # Skip folders whose name isn't a year
        if not year_str.isdigit():
            continue
        year = int(year_str)

        for file in files:
            if not file.endswith('.csv'):
                continue

            file_path = os.path.join(root, file)
            try:
                df = pd.read_csv(file_path, usecols=REQUIRED_COLUMNS, low_memory=False)
            except ValueError as exc:
                print(f"  [skip] {file}: {exc}")
                continue

            # Derive race ID from filename (e.g. "Bahrain_Laps_processed.csv" → "Bahrain")
            df['RaceID'] = file.split('_')[0]
            df['Year']   = year

            all_dfs.append(df)
            print(f"  Combined: {year} / {df['RaceID'].iloc[0]}  ({len(df):,} rows)")

    if not all_dfs:
        print("No CSV files found.")
        return

    master = pd.concat(all_dfs, ignore_index=True)
    col_order = ['Year', 'RaceID'] + REQUIRED_COLUMNS
    master = master[col_order]
    master.to_csv(output_filename, index=False)

    print(f"\nSaved {len(master):,} rows → {output_filename}")
    print(master.dtypes)


def tire_degradation_optimized(root: str) -> pd.DataFrame | None:
    """Average lap time by LapNumber × Compound (dry compounds only)."""
    all_dfs = []
    for r, _, files in os.walk(root):
        for file in files:
            if file.endswith('.csv'):
                all_dfs.append(pd.read_csv(os.path.join(r, file), low_memory=False))

    if not all_dfs:
        return None

    master = pd.concat(all_dfs, ignore_index=True)
    master['Compound'] = master['Compound'].str.title()
    master = master[master['Compound'].isin(VALID_COMPOUNDS)]

    return master.groupby(['LapNumber', 'Compound'])['LapTime_Seconds'].mean().unstack()


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print('─── Step 1: Preprocess raw lap files ───────────────────')
    process_all_files(RAW_DATA_ROOT, PROCESSED_DATA_ROOT)

    print('\n─── Step 2: Combine into master CSV ───────────────────')
    combine_race_data(PROCESSED_DATA_ROOT, OUTPUT_CSV)

    print('\nPipeline complete.')