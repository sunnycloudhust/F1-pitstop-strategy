import pandas as pd
from lifelines import CoxPHFitter
from src.utils.logger import get_logger

class TacticalSurvivalModel:
    """Mô hình Cox tính toán xác suất vào Pit tức thời (Hazard Rate)."""
    
    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)
        self.cph = CoxPHFitter(penalizer=0.1) # Dùng L2 penalty để tránh quá khớp

    def fit_model(self, df: pd.DataFrame):
        self.logger.info("Bắt đầu huấn luyện mô hình Cox Proportional Hazards...")
        
        # Định nghĩa biến sự kiện (Event): Có vào pit ở vòng đua này không? (1 = Có, 0 = Không)
        df['is_pit_stop'] = df['pit_duration'].notna().astype(int)
        
        # Lựa chọn Feature cho mô hình
        features = ['lap', 'position', 'time_seconds', 'smoothed_pace', 'is_pit_stop']
        model_data = df[features].dropna()
        
        try:
            # lap đóng vai trò là duration (t), is_pit_stop là event (E)
            self.cph.fit(model_data, duration_col='lap', event_col='is_pit_stop', show_progress=False)
            self.logger.info("Huấn luyện thành công. Tóm tắt hệ số (Coefficients):")
            self.cph.print_summary()
        except Exception as e:
            self.logger.error(f"Lỗi hội tụ mô hình: {e}")

    def predict_pit_hazard(self, current_state: pd.DataFrame) -> pd.Series:
        """Dự đoán rủi ro/xác suất vào pit tại vòng hiện tại."""
        return self.cph.predict_partial_hazard(current_state)