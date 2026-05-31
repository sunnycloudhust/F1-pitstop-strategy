# q4_agility_ttest.py
import os
import pandas as pd
from scipy import stats

def analyze_agility_using_repo_data(base_path='./Preprocessed'):
    print("\n--- KIỂM ĐỊNH T-TEST: HIỆU SUẤT ĐỘI 'AGILE' VỚI VSC/SC ---")
    data_points = []

    # Quét thư mục data của Repo cũ
    for year in os.listdir(base_path):
        year_path = os.path.join(base_path, year)
        if not os.path.isdir(year_path): continue
            
        for file in os.listdir(year_path):
            if not file.endswith('_processed.csv'): continue
                
            try:
                df = pd.read_csv(os.path.join(year_path, file))
                # Map tên cột (tùy thuộc vào repo cũ đặt tên là gì)
                driver_col = 'Driver' if 'Driver' in df.columns else 'DriverNumber'
                
                # Phân tích từng xe
                for driver in df[driver_col].unique():
                    driver_data = df[df[driver_col] == driver].sort_values('LapNumber')
                    if len(driver_data) < 10: continue # Bỏ qua DNF sớm
                    
                    start_pos = driver_data.iloc[0]['Position']
                    finish_pos = driver_data.iloc[-1]['Position']
                    pos_gained = start_pos - finish_pos
                    
                    # Tìm xem chặng này có VSC/SC không (TrackStatus 4 hoặc 6)
                    sc_laps = df[df['TrackStatus'].astype(str).str.contains('4|6')]['LapNumber'].unique()
                    is_agile = False
                    
                    if len(sc_laps) > 0 and 'PitOutTime' in df.columns:
                        pit_laps = driver_data[driver_data['PitOutTime'].notna()]['LapNumber'].values
                        for p_lap in pit_laps:
                            if any(abs(p_lap - sc_lap) <= 1 for sc_lap in sc_laps):
                                is_agile = True
                                break
                    
                    data_points.append({'Driver': driver, 'Gained': pos_gained, 'Agile': is_agile})
                    
            except Exception:
                continue # Bỏ qua các file lỗi hoặc thiếu cột

    # Phân tích thống kê
    results_df = pd.DataFrame(data_points).dropna()
    print(f"Tổng số mẫu thu thập từ Repo: {len(results_df)}")
    
    agile_group = results_df[results_df['Agile'] == True]['Gained']
    normal_group = results_df[results_df['Agile'] == False]['Gained']
    
    print(f"Nhóm Agile (Pit under SC): {len(agile_group)} xe | Tăng trung bình: {agile_group.mean():.2f} hạng")
    print(f"Nhóm Bình thường:         {len(normal_group)} xe | Tăng trung bình: {normal_group.mean():.2f} hạng")
    
    t_stat, p_val = stats.ttest_ind(agile_group, normal_group, equal_var=False)
    print(f"\n=> T-Statistic: {t_stat:.3f} | P-Value: {p_val:.5f}")
    
    if p_val < 0.05 and agile_group.mean() > normal_group.mean():
        print("=> KẾT LUẬN: H4 ĐÚNG! Sự nhạy bén mang lại ưu thế tăng hạng (Ý nghĩa thống kê 95%)")
    else:
        print("=> KẾT LUẬN: Không đủ bằng chứng thống kê.")

if __name__ == "__main__":
    analyze_agility_using_repo_data()