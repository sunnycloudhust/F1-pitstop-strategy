import numpy as np
import pandas as pd
from src.utils.logger import get_logger

class CounterfactualSimulator:
    def __init__(self, num_simulations: int = 1000):
        self.num_simulations = num_simulations
        self.logger = get_logger(self.__class__.__name__)

    def run_simulation(self, baseline_df: pd.DataFrame) -> pd.DataFrame:
        self.logger.info(f"Khởi chạy Monte Carlo {self.num_simulations} lần lặp...")
        
        # Mảng lưu trữ kết quả vị trí ngẫu nhiên
        results = []
        drivers = baseline_df['driverId'].unique()
        
        for i in range(self.num_simulations):
            sim_result = {}
            for driver in drivers:
                # Mô phỏng quá trình ngẫu nhiên: Thêm nhiễu Gaussian vào pace dự kiến
                # Bản chất ở đây có thể được mở rộng bằng các phương trình SDE (Stochastic Differential Equations)
                pace_variance = np.random.normal(0, 0.5) 
                sim_result[driver] = baseline_df[baseline_df['driverId'] == driver]['time_seconds'].mean() + pace_variance
            
            # Sắp xếp lại vị trí dựa trên thời gian mô phỏng
            sorted_drivers = sorted(sim_result.items(), key=lambda x: x[1])
            for pos, (driver, _) in enumerate(sorted_drivers, 1):
                results.append({'simulation_id': i, 'driverId': driver, 'simulated_position': pos})
                
        return pd.DataFrame(results)