import pandas as pd
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')

def analyze(target_year, race_keyword):
    print(f"\n{'='*60}")
    print(f"ANALYSIS: OVERTAKES IN {race_keyword.upper()} GRAND PRIX {target_year}")
    print(f"{'='*60}")
    
    data_path = './kaggle_data/'
    
    try:
        laps_df = pd.read_csv(f'{data_path}lap_times.csv')  
        pits_df = pd.read_csv(f'{data_path}pit_stops.csv')  
        races_df = pd.read_csv(f'{data_path}races.csv')     
        race_info = races_df[(races_df['year'] == target_year) & 
                             (races_df['name'].str.contains(race_keyword, case=False, na=False))]
        
        ### foolproof check ###
        if race_info.empty:
            print(f"[ERROR] No data found for keyword '{race_keyword}' in {target_year}.")
            return
        race_id = race_info.iloc[0]['raceId']
        full_race_name = race_info.iloc[0]['name']
        print(f"[INFO] Successfully loaded race: {full_race_name} (Race ID: {race_id})")
        
        race_laps = laps_df[laps_df['raceId'] == race_id]
        
        position_matrix = race_laps.pivot(index='lap', columns='driverId', values='position')
    
        race_pits = pits_df[pits_df['raceId'] == race_id]
        
        if position_matrix.empty:
            print("[ERROR] Lap-by-lap data is missing for this race.")
            return

        strategic_overtakes = 0
        on_track_overtakes = 0
        
        max_lap = position_matrix.index.max()
        
        for lap in range(2, max_lap + 1):
            if lap not in position_matrix.index or (lap - 1) not in position_matrix.index:
                continue
                
            current_lap_positions = position_matrix.loc[lap]
            previous_lap_positions = position_matrix.loc[lap - 1]
            
            # Duyệt qua từng tay đua có mặt trên đường đua
            for driver in position_matrix.columns:
                curr_pos = current_lap_positions[driver]
                prev_pos = previous_lap_positions[driver]
                
                # Bỏ qua nếu tay đua đã bỏ cuộc (NaN)
                if pd.isna(curr_pos) or pd.isna(prev_pos):
                    continue
                if curr_pos <= 10 and curr_pos < prev_pos:
                    passed_drivers = previous_lap_positions[
                        (previous_lap_positions < prev_pos) & 
                        (current_lap_positions > curr_pos)
                    ].index
                    for passed_driver in passed_drivers:
                        recent_pits = race_pits[
                            (race_pits['driverId'].isin([driver, passed_driver])) & 
                            (race_pits['lap'] >= lap - 2) & 
                            (race_pits['lap'] <= lap + 2)
                        ]
                        
                        if not recent_pits.empty:
                            strategic_overtakes += 1
                        else:
                            on_track_overtakes += 1

        total_overtakes = strategic_overtakes + on_track_overtakes
        print("\n[RESULT] Overtake Classification (Top 10):")
        print(f"Total Overtakes Recorded : {total_overtakes}")
        
        if total_overtakes > 0:
            strat_pct = (strategic_overtakes / total_overtakes) * 100
            track_pct = (on_track_overtakes / total_overtakes) * 100
            print(f"- Strategic Overtakes    : {strategic_overtakes} ({strat_pct:.1f}%)")
            print(f"- On-Track Overtakes     : {on_track_overtakes} ({track_pct:.1f}%)")
            

            plt.figure(figsize=(8, 6), facecolor='white')
            
            labels = ['Strategic Overtakes\n(Pit-related)', 'On-Track Overtakes\n(Racing)']
            sizes = [strategic_overtakes, on_track_overtakes]
        
            colors = ['#E00600', '#00A19B'] 

            explode = (0.03, 0.03) if (strategic_overtakes > 0 and on_track_overtakes > 0) else (0, 0)
            wedges, texts, autotexts = plt.pie(
                sizes, 
                explode=explode, 
                labels=labels, 
                colors=colors, 
                autopct='%1.1f%%',
                startangle=90,
                pctdistance=0.75,
                wedgeprops=dict(width=0.4, edgecolor='white', linewidth=3),
                textprops={'fontsize': 12, 'weight': 'bold', 'color': '#333333'}
            )
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontsize(14)
                autotext.set_weight('bold')
            plt.title(f'TOP 10 OVERTAKE ANALYSIS\n{full_race_name.upper()} {target_year}', 
                      fontsize=15, fontweight='900', color='#111111', pad=20)
            
            os.makedirs('plots', exist_ok=True)
            filename = f"plots/Q1_Chart_{full_race_name.replace(' ', '')}_{target_year}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close() 
            
            print(f"\n[INFO] Chart saved successfully in folder: '{filename}'")
            
        else:
            print("[INFO] No overtakes found in Top 10 for this race.")

    except FileNotFoundError as e:
        print(f"[ERROR] Missing Kaggle dataset file. Make sure data is in './kaggle_data/'.\nDetails: {e}")
    except Exception as e:
        print(f"[ERROR] An unexpected error occurred: {e}")

if __name__ == "__main__":    
    analyze(2023, 'Monaco')
    analyze(2023, 'Monaco')
    analyze(2022, 'Italian')