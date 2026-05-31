# q3_correlation.py
import pandas as pd
from scipy import stats

def analyze_correlation():
    print("\n--- PHÂN TÍCH TƯƠNG QUAN: PIT STOP TIME vs CHAMPIONSHIP POINTS ---")
    path = './kaggle_data/'
    try:
        pit_stops = pd.read_csv(f'{path}pit_stops.csv')
        results = pd.read_csv(f'{path}results.csv')
        races = pd.read_csv(f'{path}races.csv')
        cons_standings = pd.read_csv(f'{path}constructor_standings.csv')
        
        # Lấy ConstructorID cho từng xe
        pit_results = pit_stops.merge(results[['raceId', 'driverId', 'constructorId']], on=['raceId', 'driverId'])
        pit_races = pit_results.merge(races[['raceId', 'year']], on='raceId')
        
        # Lấy kỷ nguyên Turbo Hybrid (2014 trở đi)
        recent = pit_races[pit_races['year'] >= 2014]
        # Tính Median Pit Time
        pit_med = recent.groupby(['year', 'constructorId'])['milliseconds'].median().reset_index()
        
        # Lấy điểm số vô địch cuối năm
        standings_races = cons_standings.merge(races[['raceId', 'year']], on='raceId')
        final_standings = standings_races.sort_values('raceId').groupby(['year', 'constructorId']).tail(1)
        
        # Hợp nhất dữ liệu
        df = pit_med.merge(final_standings[['year', 'constructorId', 'points']], on=['year', 'constructorId'])
        df = df.dropna()
        
        pearson_r, p_pearson = stats.pearsonr(df['milliseconds'], df['points'])
        
        print(f"Hệ số Pearson (r): {pearson_r:.3f}")
        print(f"P-value: {p_pearson:.5f}")
        
        if p_pearson < 0.05:
            print("=> KẾT LUẬN: Giả thuyết đúng. Có ý nghĩa thống kê.")
            if pearson_r < 0:
                print("Tương quan nghịch: Đội có thời gian pit stop càng ngắn (ít mili-giây), điểm số càng cao.")
        else:
             print("=> KẾT LUẬN: Không đủ ý nghĩa thống kê.")
             
    except FileNotFoundError:
        print("Lỗi: Không tìm thấy data Kaggle. Hãy chắc chắn bạn đã để file trong folder 'kaggle_data/'")

if __name__ == "__main__":
    analyze_correlation()