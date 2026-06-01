import pandas as pd
from src.utils.logger import get_logger
from src.ingestion.dataset_client import F1DatasetIngestor

class UnifiedRaceTimelineProcessor:
    """Hợp nhất các bảng rời rạc thành Timeline duy nhất."""
    
    def __init__(self, ingestor: F1DatasetIngestor):
        self.ingestor = ingestor
        self.logger = get_logger(self.__class__.__name__)

    def _convert_time_to_seconds(self, time_str) -> float:
        """Chuyển chuỗi thời gian (VD: '1:34.567') sang giây."""
        try:
            if pd.isna(time_str):
                return None
            parts = str(time_str).split(':')
            if len(parts) == 2:
                return float(parts[0]) * 60 + float(parts[1])
            return float(time_str)
        except Exception:
            return None

    def build_unified_timeline(self, target_year: int, target_round: int) -> pd.DataFrame:
        self.logger.info(f"Đang xử lý Timeline: Năm {target_year}, Chặng {target_round}...")
        
        races_df = self.ingestor.get_table('Race')
        lap_times_df = self.ingestor.get_table('LapTime')
        pit_stops_df = self.ingestor.get_table('PitStop')
        results_df = self.ingestor.get_table('Result')

        # Xác định raceId
        race_info = races_df[(races_df['year'] == target_year) & (races_df['round'] == target_round)]
        if race_info.empty:
            self.logger.error("Không tìm thấy dữ liệu chặng đua.")
            return pd.DataFrame()
            
        current_race_id = race_info.iloc[0]['raceId']
        
        # Lọc dữ liệu theo raceId
        laps = lap_times_df[lap_times_df['raceId'] == current_race_id].copy()
        pits = pit_stops_df[pit_stops_df['raceId'] == current_race_id].copy()
        results = results_df[results_df['raceId'] == current_race_id].copy()

        if laps.empty:
            self.logger.warning("Không có dữ liệu LapTime cho chặng này.")
            return pd.DataFrame()

        # Tiền xử lý thời gian
        laps['time_seconds'] = laps['time'].apply(self._convert_time_to_seconds)

        # LEFT JOIN 1: LapTime + PitStop (để biết vòng nào có pit)
        timeline_df = pd.merge(
            laps, 
            pits[['raceId', 'driverId', 'lap', 'duration']], 
            on=['raceId', 'driverId', 'lap'], 
            how='left'
        )
        timeline_df.rename(columns={'duration': 'pit_duration'}, inplace=True)
        
        # LEFT JOIN 2: Gắn thêm thông tin Result (vị trí, điểm số)
        timeline_df = pd.merge(
            timeline_df,
            results[['raceId', 'driverId', 'grid', 'position', 'statusId']],
            on=['raceId', 'driverId'],
            how='left'
        )

        self.logger.info(f"Xử lý thành công Timeline: {len(timeline_df)} dòng.")
        return timeline_df