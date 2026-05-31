import fastf1
import pandas as pd
import os
fastf1.Cache.enable_cache('cache')

def calculate_vsc_time_save(year, race_name):
    print(f"\nCalculating SC/VSC delta: {year} {race_name}")
    try:
        session = fastf1.get_session(year, race_name, 'R')
        session.load(telemetry=False, weather=False)
        laps = session.laps
        
        # TrackStatus '1': Clear (Green), '6': VSC
        green_laps = laps[laps['TrackStatus'] == '1']
        vsc_laps = laps[laps['TrackStatus'].str.contains('6', na=False)]
        
        green_pace = green_laps['LapTime'].dt.total_seconds().median()
        vsc_pace = vsc_laps['LapTime'].dt.total_seconds().median()
        
        if pd.isna(vsc_pace):
            print("=> No SC/VSC")
            return

        time_saved = vsc_pace - green_pace
        
        print(f"- Race Pace (Green): {green_pace:.2f} s")
        print(f"- Race Pace (VSC):   {vsc_pace:.2f} s")
        print(f"=> Saving {time_saved:.2f} seconds when pitting under SC/VSC")
    except Exception as e:
        print(f"Lỗi: {e}")

if __name__ == "__main__":
    calculate_vsc_time_save(2021, 'Monza')