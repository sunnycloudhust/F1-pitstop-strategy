import os
import time
import fastf1
import pandas as pd

# ── Config ────────────────────────────────────────────────────────────────────
CACHE_DIR      = './f1_raw_cache'
OUTPUT_DIR     = './f1_data_csv_export'
SEASONS        = range(2018, 2026)        # inclusive
TELEMETRY_FROM = 2018                     # only load telemetry from this year+
POLITE_DELAY   = 3                        # seconds between successful races
ERROR_DELAY    = 10                       # seconds after a race-level error
API_FAIL_DELAY = 60                       # seconds when the schedule itself fails

# ── Setup ─────────────────────────────────────────────────────────────────────
os.makedirs(CACHE_DIR,  exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
fastf1.Cache.enable_cache(CACHE_DIR)


def export_season(year: int) -> None:
    year_dir = os.path.join(OUTPUT_DIR, str(year))
    os.makedirs(year_dir, exist_ok=True)

    try:
        schedule = fastf1.get_event_schedule(year)
    except Exception as exc:
        print(f"[{year}] Could not load schedule: {exc}. Waiting {API_FAIL_DELAY}s …")
        time.sleep(API_FAIL_DELAY)
        return

    for _, event in schedule.iterrows():
        if event['EventFormat'] == 'testing':
            continue

        gp       = event['EventName']
        laps_csv = os.path.join(year_dir, f"{gp}_Laps.csv")

        if os.path.exists(laps_csv):
            print(f"[{year}] Skipping {gp} — already downloaded.")
            continue

        print(f"[{year}] Downloading: {gp} …")
        load_telemetry = year >= TELEMETRY_FROM

        try:
            session = fastf1.get_session(year, event['RoundNumber'], 'R')
            session.load(telemetry=load_telemetry)

            # Results
            if session.results is not None and not session.results.empty:
                session.results.to_csv(os.path.join(year_dir, f"{gp}_Results.csv"))

            # Laps
            session.laps.to_csv(laps_csv)

            # Telemetry (per-driver, individual failures skipped)
            if load_telemetry:
                tel_frames = []
                for driver in session.drivers:
                    try:
                        tel = session.laps.pick_driver(driver).get_telemetry()
                        tel['Driver'] = driver
                        tel_frames.append(tel)
                    except Exception as driver_exc:
                        print(f"  [warn] Telemetry for driver {driver} failed: {driver_exc}")

                if tel_frames:
                    pd.concat(tel_frames, ignore_index=True).to_csv(
                        os.path.join(year_dir, f"{gp}_Telemetry.csv")
                    )

            print(f"[{year}] ✓ {gp}")
            time.sleep(POLITE_DELAY)

        except Exception as exc:
            print(f"[{year}] Error on {gp}: {exc}. Waiting {ERROR_DELAY}s …")
            time.sleep(ERROR_DELAY)


if __name__ == "__main__":
    for season in SEASONS:
        print(f"\n{'─'*50}")
        print(f"  Season {season}")
        print(f"{'─'*50}")
        export_season(season)
    print("\nAll seasons downloaded.")