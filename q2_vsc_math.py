# q2_vsc_math.py
import fastf1
import pandas as pd

fastf1.Cache.enable_cache('cache')

def calculate_vsc_time_save(year, race_name):
    print(f"\nTính toán VSC Delta - Chặng: {year} {race_name}")
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
            print("=> Chặng này không có VSC hoặc VSC quá ngắn.")
            return

        time_saved = vsc_pace - green_pace
        
        print(f"- Race Pace (Green): {green_pace:.2f} s")
        print(f"- Race Pace (VSC):   {vsc_pace:.2f} s")
        print(f"=> Khi vào pit dưới VSC, tay đua tiết kiệm toán học: {time_saved:.2f} giây so với đối thủ!")
    except Exception as e:
        print(f"Lỗi: {e}")

if __name__ == "__main__":
    # Monza 2022 có pha VSC rất rõ ràng
    calculate_vsc_time_save(2022, 'Monza')