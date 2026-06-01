import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
import warnings
warnings.filterwarnings('ignore')

def analyze_q4_agility_kaggle():
    print("Q4: AGILITY UNDER SC/VSC vs OVERPERFORMANCE")
    path = './kaggle_data/'
    
    try:
        # ==========================================
        # STEP 1: LOAD DATA & DATA CLEANING
        # ==========================================
        laps = pd.read_csv(f'{path}lap_times.csv')
        pits = pd.read_csv(f'{path}pit_stops.csv')
        races = pd.read_csv(f'{path}races.csv')
        results = pd.read_csv(f'{path}results.csv')
        
        # --- DATA CLEANING ---
        laps['milliseconds'] = pd.to_numeric(laps['milliseconds'], errors='coerce')
        laps['lap'] = pd.to_numeric(laps['lap'], errors='coerce')
        pits['lap'] = pd.to_numeric(pits['lap'], errors='coerce')
        results['grid'] = pd.to_numeric(results['grid'], errors='coerce')
        results['positionOrder'] = pd.to_numeric(results['positionOrder'], errors='coerce')
        
        # Bỏ đi các dòng NaN vừa được chuyển đổi để tránh lỗi toán học
        laps = laps.dropna(subset=['milliseconds', 'lap'])
        
        # Filter for modern era (2018 onwards)
        recent_races = races[races['year'] >= 2018]['raceId']
        laps = laps[laps['raceId'].isin(recent_races)].copy()
        pits = pits[pits['raceId'].isin(recent_races)].copy()
        results = results[results['raceId'].isin(recent_races)].copy()
        
        print("[INFO] Identifying SC/VSC laps and calculating driver agility...")
        
        # ==========================================
        # STEP 2: INFER SC/VSC LAPS (ANOMALY DETECTION)
        # ==========================================
        pits['is_pit'] = True
        laps = laps.merge(pits[['raceId', 'driverId', 'lap', 'is_pit']], on=['raceId', 'driverId', 'lap'], how='left')
        laps['is_pit'] = laps['is_pit'].fillna(False)
        
        non_pit_laps = laps[laps['is_pit'] == False]
        ref_times = non_pit_laps.groupby(['raceId', 'lap'])['milliseconds'].median().reset_index()
        race_pace = non_pit_laps.groupby('raceId')['milliseconds'].median().reset_index()
        
        ref_times = ref_times.merge(race_pace, on='raceId', suffixes=('', '_race'))
        
        # Rule: If the pack is >20% slower than normal race pace, it's a SC/VSC lap
        ref_times['is_sc_vsc'] = ref_times['milliseconds'] > (1.20 * ref_times['milliseconds_race'])
        sc_laps_df = ref_times[ref_times['is_sc_vsc']][['raceId', 'lap']]
        
        sc_dict = sc_laps_df.groupby('raceId')['lap'].apply(list).to_dict()
        
        # ==========================================
        # STEP 3: FLAG AGILE PIT STOPS
        # ==========================================
        def check_agility(row):
            r_id = row['raceId']
            p_lap = row['lap']
            if r_id in sc_dict:
                if any(abs(p_lap - sc_lap) <= 1 for sc_lap in sc_dict[r_id]):
                    return True
            return False
            
        pits['is_agile'] = pits.apply(check_agility, axis=1)
        driver_agility = pits.groupby(['raceId', 'driverId'])['is_agile'].any().reset_index()
        
        # ==========================================
        # STEP 4: CALCULATE OVERPERFORMANCE
        # ==========================================
        finishers = results[results['statusId'].isin([1, 11, 12, 13, 14, 15, 16, 17, 18, 19])].copy()
        finishers = finishers.dropna(subset=['grid', 'positionOrder'])

        finishers = finishers[finishers['grid'] > 0]
        
        finishers['positions_gained'] = finishers['grid'] - finishers['positionOrder']
        
        final_df = finishers.merge(driver_agility, on=['raceId', 'driverId'], how='left')
        final_df['is_agile'] = final_df['is_agile'].fillna(False)
        
        # ==========================================
        # STEP 5: T-TEST & OUTPUT
        # ==========================================
        agile_group = final_df[final_df['is_agile'] == True]['positions_gained']
        normal_group = final_df[final_df['is_agile'] == False]['positions_gained']
        
        print("\n[RESULT] Statistical Analysis (Welch's T-Test):")
        print(f"Total valid finisher samples: {len(final_df)}")
        print(f"- Agile Group (Pit under SC) : {len(agile_group)} samples | Mean Gained: {agile_group.mean():.2f} positions")
        print(f"- Normal Group (Green Pit)   : {len(normal_group)} samples | Mean Gained: {normal_group.mean():.2f} positions")
        
        t_stat, p_val = stats.ttest_ind(agile_group, normal_group, equal_var=False)
        
        print(f"\n=> T-Statistic: {t_stat:.3f}")
        print(f"=> P-Value    : {p_val:.5f}")
        
        if p_val < 0.05 and agile_group.mean() > normal_group.mean():
            print("\n[CONCLUSION] Hypothesis H4 is SUPPORTED (Statistically Significant).")
            print("Teams that agilely react to race neutralizations consistently finish higher than their starting positions.")
        else:
            print("\n[CONCLUSION] Hypothesis H4 is NOT SUPPORTED (Not Statistically Significant).")
            
        # ==========================================
        # STEP 6: GENERATE BOXPLOT
        # ==========================================
        os.makedirs('q4_agility', exist_ok=True)
        
        plt.figure(figsize=(9, 6), facecolor='white')
        
        final_df['Strategy Type'] = final_df['is_agile'].map({
            True: 'Agile (SC/VSC Pit)', 
            False: 'Normal (Green Flag Pit)'
        })
        
        sns.boxplot(x='Strategy Type', y='positions_gained', data=final_df, 
                    palette=['#3498db', '#e74c3c'], width=0.5, fliersize=3)
        
        plt.axhline(0, color='#2c3e50', linestyle='--', linewidth=1.5, alpha=0.7)
        
        plt.title('Performance Impact: Agile vs Normal Strategies\n(2018 - Present)', 
                  fontsize=14, fontweight='bold', pad=15)
        plt.ylabel('Positions Gained (Start - Finish)', fontsize=12, fontweight='bold')
        plt.xlabel('')
        plt.grid(axis='y', linestyle=':', alpha=0.6)
        
        file_name = 'plots/Q4_Agility_Boxplot.png'
        plt.savefig(file_name, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\n[INFO] Boxplot saved successfully to: '{file_name}'")

    except FileNotFoundError as e:
        print(f"[ERROR] Missing Kaggle dataset file. {e}")
    except Exception as e:
        print(f"[ERROR] An unexpected error occurred: {e}")

if __name__ == "__main__":
    analyze_q4_agility_kaggle()