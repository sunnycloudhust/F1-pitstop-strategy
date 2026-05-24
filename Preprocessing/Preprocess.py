import os
import pandas as pd

def tire_degradation_optimized(root):
    """Calculates average tire degradation per lap and compound."""
    all_dfs = []
    
    for r, d, f in os.walk(root):
        for file in f:
            if file.endswith(".csv"):
                file_path = os.path.join(r, file)
                all_dfs.append(pd.read_csv(file_path))

    if not all_dfs:
        return None

    master_df = pd.concat(all_dfs, ignore_index=True)
    master_df['Compound'] = master_df['Compound'].str.title() 

    valid_compounds = ["Soft", "Medium", "Hard"]
    master_df = master_df[master_df['Compound'].isin(valid_compounds)]
    
    avg_degradation = master_df.groupby(['LapNumber', 'Compound'])['LapTime_Seconds'].mean().unstack()
    return avg_degradation 

def combine_race_data(input_root, output_filename="all_data.csv"):
    """Combines processed race CSVs, loading only the necessary columns for the model."""
    all_dfs = []
    
    required_columns = [
        'DriverNumber', 'Driver', 'Team', 'Compound',
        'LapTime_Seconds', 'Position', 'LapNumber', 'Stint', 'TyreLife',
        'TrackStatus', 'delta_laptime', 'CumulativeTimeStint', 
        'race_progress_fraction', 'HasPitStop'
    ]
    
    for r, d, f in os.walk(input_root):
        for file in f:
            if file.endswith('.csv'):
                file_path = os.path.join(r, file)
                
                try:
                    df = pd.read_csv(file_path, usecols=required_columns)
                except ValueError as e:
                    print(f"Skipping {file}: {e}")
                    continue
                
                race_id = os.path.basename(file).split('_')[0] 
                df['RaceID'] = race_id
                
                year = os.path.basename(r)
                df['Year'] = int(year)
                
                all_dfs.append(df)
                print(f"Combined: {year} {race_id} ({len(df)} rows)")
            
    if all_dfs:
        master_df = pd.concat(all_dfs, ignore_index=True)
        
        final_column_order = ['Year', 'RaceID'] + required_columns
        master_df = master_df[final_column_order]
        
        master_df.to_csv(output_filename, index=False)
        print(f"Successfully saved filtered data to {output_filename}")
    else:
        print("No CSV files found to combine.")

def drop_na_wet(df):
    """Drops missing compounds and filters out Wet/Intermediate tires."""
    df = df.dropna(subset=['Compound'])
    df = df[~df['Compound'].isin(['Wet', 'Intermediate', 'UNKNOWN'])]
    return df

def preprocess_telemetry(df):
    """Applies all feature engineering and data cleaning to a single race DataFrame."""
    df = drop_na_wet(df)

    time_columns = [
        'LapTime', 'Sector1Time', 'Sector2Time', 'Sector3Time',
        'Sector1SessionTime', 'Sector2SessionTime', 'Sector3SessionTime'
    ]
    for col in time_columns:
        if col in df.columns:
            df[f'{col}_Seconds'] = pd.to_timedelta(df[col]).dt.total_seconds()

    df['HasPitStop'] = df['PitInTime'].notna().astype(int)

    df['delta_laptime'] = df.groupby('Driver')['LapTime_Seconds'].diff().fillna(0)
    df['CumulativeTimeStint'] = df.groupby(['Driver', 'Stint'])['LapTime_Seconds'].cumsum()
    
    max_laps = df['LapNumber'].max()
    df['race_progress_fraction'] = df['LapNumber'] / max_laps

    mode_series = df['TrackStatus'].mode()
    track_status_mode = mode_series[0] if not mode_series.empty else 1.0
    df['TrackStatus'] = df['TrackStatus'].fillna(track_status_mode)

    return df

def process_all_files(input_root, output_root):
    """Iterates through all raw folders, processes the CSVs, and saves them."""
    for folder in os.listdir(input_root):
        folder_path = os.path.join(input_root, folder)
        
        if os.path.isdir(folder_path):
            dest_folder = os.path.join(output_root, folder)
            os.makedirs(dest_folder, exist_ok=True)
            
            for file in os.listdir(folder_path):
                if file.endswith("_Laps.csv"):
                    print(f"Processing: {file}")
                    
                    file_path = os.path.join(folder_path, file)
                    df = pd.read_csv(file_path)
                    processed_df = preprocess_telemetry(df)
                    
                    new_filename = f"{os.path.splitext(file)[0]}_processed.csv"
                    output_path = os.path.join(dest_folder, new_filename)
                    processed_df.to_csv(output_path, index=False)

if __name__ == "__main__":
    
    RAW_DATA_ROOT = r"C:\Nguyen Tri\Code\Statisanalyss\f1_data_csv_export\\"
    PROCESSED_DATA_ROOT = r"C:\Nguyen Tri\Code\Statisanalyss\Preprocessed\\"
    
    print("--- Starting Telemetry Preprocessing ---")
    process_all_files(RAW_DATA_ROOT, PROCESSED_DATA_ROOT)
    
    print("\n--- Combining Processed Files ---")
    combine_race_data(PROCESSED_DATA_ROOT, output_filename="all_data.csv")
    
    print("\nPipeline Complete!")