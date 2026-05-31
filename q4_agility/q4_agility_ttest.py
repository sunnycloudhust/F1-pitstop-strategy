# q4_agility_ttest.py
import os
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_agility_using_repo_data(base_path='./Preprocessed'):
    print("\n--- KIỂM ĐỊNH T-TEST: HIỆU SUẤT ĐỘI 'AGILE' VỚI VSC/SC ---")
    data_points = []

    for year in os.listdir(base_path):
        year_path = os.path.join(base_path, year)
        if not os.path.isdir(year_path): continue
            
        for file in os.listdir(year_path):
            if not file.endswith('_processed.csv'): continue
                
            try:
                df = pd.read_csv(os.path.join(year_path, file))
                driver_col = 'Driver' if 'Driver' in df.columns else 'DriverNumber'
                for driver in df[driver_col].unique():
                    driver_data = df[df[driver_col] == driver].sort_values('LapNumber')
                    if len(driver_data) < 10: continue 
                    
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
    ############### plotting #####################
    plt.figure(figsize=(8, 6))
    
    # Đổi tên nhãn cho đẹp
    results_df['Nhóm Chiến Thuật'] = results_df['Agile'].map({True: 'Agile (Pit dưới SC/VSC)', False: 'Bình thường (Non-Agile)'})
    
    sns.boxplot(x='Nhóm Chiến Thuật', y='Gained', data=results_df, palette='Set2')
    
    plt.title('So sánh Cải thiện Thứ hạng: Nhóm Agile vs Non-Agile', fontsize=14)
    plt.ylabel('Số hạng tăng được (Positions Gained)', fontsize=12)
    plt.xlabel('') # Bỏ trống vì đã có nhãn trục x
    plt.axhline(0, color='red', linestyle='--', alpha=0.5) # Đường baseline (Không tăng không giảm)
    plt.show()
    plt.savefig('Q4_Boxplot_Agility.png', dpi=300, bbox_inches='tight')
    print("Đã lưu biểu đồ thành file 'Q4_Boxplot_Agility.png'")
    #############################################
    if p_val < 0.05 and agile_group.mean() > normal_group.mean():
        print("=> KẾT LUẬN: H4 ĐÚNG! Sự nhạy bén mang lại ưu thế tăng hạng (Ý nghĩa thống kê 95%)")
    else:
        print("=> KẾT LUẬN: Không đủ bằng chứng thống kê.")

if __name__ == "__main__":
    analyze_agility_using_repo_data()