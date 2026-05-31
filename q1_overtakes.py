# q1_overtakes.py
import fastf1
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

fastf1.Cache.enable_cache('cache')

def analyze_overtakes(year, race_name):
    print(f"\nPhân tích chặng: {year} {race_name}")
    session = fastf1.get_session(year, race_name, 'R')
    session.load(telemetry=False, weather=False)
    laps = session.laps
    
    strategic_overtakes, on_track_overtakes = 0, 0
    lap_pos = laps.pivot(index='LapNumber', columns='DriverNumber', values='Position')
    
    for lap in range(2, int(lap_pos.index.max())):
        for driver in laps['DriverNumber'].unique():
            try:
                curr_pos = lap_pos.loc[lap, driver]
                prev_pos = lap_pos.loc[lap-1, driver]
                
                # Chỉ xét Top 10 và có sự tăng hạng (giảm số Position)
                if curr_pos <= 10 and curr_pos < prev_pos:
                    driver_laps = laps[laps['DriverNumber'] == driver]
                    # Xét window +- 2 laps
                    recent_pit = driver_laps[(driver_laps['LapNumber'] >= lap-2) & (driver_laps['LapNumber'] <= lap+1)]
                    
                    if not recent_pit['PitInTime'].isna().all():
                        strategic_overtakes += 1
                    else:
                        on_track_overtakes += 1
            except KeyError:
                continue

    total = strategic_overtakes + on_track_overtakes
    if total > 0:
        print(f"- Tổng số pha vượt (Top 10): {total}")
        print(f"- Strategic Overtakes (Pit): {strategic_overtakes} ({(strategic_overtakes/total)*100:.1f}%)")
        print(f"- On-Track Overtakes:      {on_track_overtakes} ({(on_track_overtakes/total)*100:.1f}%)")
    return strategic_overtakes, on_track_overtakes

if __name__ == "__main__":
    # Test trên 2 chặng đặc thù
    analyze_overtakes(2023, 'Bahrain')
    analyze_overtakes(2023, 'Monaco') 