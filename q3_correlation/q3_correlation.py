# q3_correlation.py
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_correlation():
    print("\n--- Analysing: PIT STOP TIME vs CHAMPIONSHIP POINTS ---")
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
        recent = pit_races[pit_races['year'] >= 2014].copy()
        
        # SỬA LỖI TẠI ĐÂY: Ép kiểu cột milliseconds sang dạng số, các chuỗi lỗi (như '\N') sẽ bị biến thành NaN
        recent['milliseconds'] = pd.to_numeric(recent['milliseconds'], errors='coerce')
        
        # Tính Median Pit Time (Pandas sẽ tự động bỏ qua NaN khi tính median)
        pit_med = recent.groupby(['year', 'constructorId'])['milliseconds'].median().reset_index()
        
        # Lấy điểm số vô địch cuối năm
        standings_races = cons_standings.merge(races[['raceId', 'year']], on='raceId')
        final_standings = standings_races.sort_values('raceId').groupby(['year', 'constructorId']).tail(1)
        
        # Hợp nhất dữ liệu
        df = pit_med.merge(final_standings[['year', 'constructorId', 'points']], on=['year', 'constructorId'])
        df = df.dropna()
        
        pearson_r, p_pearson = stats.pearsonr(df['milliseconds'], df['points'])
        spearman_r, p_spearman = stats.spearmanr(df['milliseconds'], df['points'])
        
        print(f"Hệ số Pearson (Tuyến tính): {pearson_r:.3f} | P-value: {p_pearson:.5f}")
        print(f"Hệ số Spearman (Thứ hạng): {spearman_r:.3f} | P-value: {p_spearman:.5f}")
        ######################## plotting #################3
        plt.figure(figsize=(10, 6))
        sns.regplot(x='milliseconds', y='points', data=df, 
                    scatter_kws={'alpha':0.6, 'color':'blue'}, 
                    line_kws={'color':'red', 'lw':2})
        
        plt.title('Tương quan giữa Thời gian Pit Stop (Median) và Điểm số Vô địch (2014-nay)', fontsize=14)
        plt.xlabel('Thời gian Pit Stop (Mili-giây)', fontsize=12)
        # Giới hạn trục X để bỏ đi các outliers quá dị (vd pit mất 1 phút) giúp biểu đồ dễ nhìn hơn
        plt.xlim(20000, 35000) 
        plt.ylabel('Tổng điểm Vô địch (Points)', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Lưu file ảnh
        plt.savefig('Q3_Correlation_Plot.png', dpi=300, bbox_inches='tight')
        print("Đã lưu biểu đồ thành file 'Q3_Correlation_Plot.png'")
        plt.show() # Bỏ comment dòng này nếu muốn biểu đồ bật lên ngay khi chạy
        
        if p_spearman < 0.05:
            print("=> KẾT LUẬN H4.3: ĐÚNG. Có ý nghĩa thống kê.")
            if spearman_r < 0:
                print("Tương quan nghịch: Đội có thời gian pit stop (median) càng ngắn, điểm số càng cao.")
        else:
             print("=> KẾT LUẬN: Không đủ bằng chứng thống kê.")
             
    except FileNotFoundError:
        print("Lỗi: Không tìm thấy data Kaggle. Hãy chắc chắn bạn đã để file trong folder 'kaggle_data/'")

if __name__ == "__main__":
    analyze_correlation()