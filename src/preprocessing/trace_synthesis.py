import pandas as pd
import numpy as np
import statsmodels.api as sm
from src.utils.logger import get_logger
from src.ingestion.dataset_client import F1DatasetIngestor

class TraceSynthesizer:
    def __init__(self, ingestor: F1DatasetIngestor):
        self.ingestor = ingestor
        self.logger = get_logger(self.__class__.__name__)

    def _time_to_seconds(self, time_str) -> float:
        try:
            if pd.isna(time_str): return np.nan
            parts = str(time_str).split(':')
            return float(parts[0]) * 60 + float(parts[1]) if len(parts) == 2 else float(time_str)
        except: return np.nan

    def build_features(self, target_year: int, target_round: int) -> pd.DataFrame:
        self.logger.info(f"Tổng hợp tính năng: Năm {target_year}, Chặng {target_round}")
        
        # 1. Hợp nhất dữ liệu (Relational JOIN In-memory)
        races = self.ingestor.get_table('Race')
        laps = self.ingestor.get_table('LapTime').copy()
        pits = self.ingestor.get_table('PitStop').copy()
        results = self.ingestor.get_table('Result').copy()

        race_id = races[(races['year'] == target_year) & (races['round'] == target_round)]['raceId'].iloc[0]
        
        df_laps = laps[laps['raceId'] == race_id].copy()
        df_laps['time_seconds'] = df_laps['time'].apply(self._time_to_seconds)
        
        # Ghép PitStop và Result
        df_merged = pd.merge(df_laps, pits[['raceId', 'driverId', 'lap', 'duration']], on=['raceId', 'driverId', 'lap'], how='left')
        df_merged.rename(columns={'duration': 'pit_duration'}, inplace=True)
        df_merged = pd.merge(df_merged, results[['raceId', 'driverId', 'grid', 'position']], on=['raceId', 'driverId'], how='left')

        # 2. Xử lý thuật toán LOWESS Smoothing cho từng tay đua để tính Pace Degradation
        df_merged['smoothed_pace'] = np.nan
        for driver in df_merged['driverId'].unique():
            idx = df_merged['driverId'] == driver
            driver_laps = df_merged[idx].sort_values('lap')
            
            # Lọc bỏ các vòng out-lap/in-lap bất thường để không làm méo đồ thị
            valid_laps = driver_laps[driver_laps['time_seconds'] < driver_laps['time_seconds'].quantile(0.95)]
            if len(valid_laps) > 5:
                smoothed = sm.nonparametric.lowess(valid_laps['time_seconds'], valid_laps['lap'], frac=0.1)
                df_merged.loc[valid_laps.index, 'smoothed_pace'] = smoothed[:, 1]
                
        # 3. Tính toán Cumulative Delta (Delta tích lũy)
        median_pace = df_merged.groupby('lap')['time_seconds'].median()
        df_merged['lap_delta'] = df_merged['time_seconds'] - df_merged['lap'].map(median_pace)
        
        self.logger.info("Hoàn tất xử lý LOWESS và tính toán Delta.")
        return df_merged