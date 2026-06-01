import os
from src.ingestion.dataset_client import F1DatasetIngestor
from src.preprocessing.trace_synthesis import TraceSynthesizer
from src.models.survival_model import TacticalSurvivalModel
from src.models.monte_carlo_sim import CounterfactualSimulator

def main():
    os.makedirs('./data/raw', exist_ok=True)
    os.makedirs('./data/processed', exist_ok=True)
    
    # 1. Nạp dữ liệu
    ingestor = F1DatasetIngestor('./data/f1_data')
    ingestor.load_tables(['races', 'lap_times', 'pit_stops', 'results'])    
    # 2. Tiền xử lý & Tính toán Delta
    synthesizer = TraceSynthesizer(ingestor)
    features_df = synthesizer.build_features(target_year=2023, target_round=1)
    
    if features_df.empty:
        return
        
    features_df.to_csv('./data/processed/engineered_features.csv', index=False)
    
    # 3. Chạy Mô hình Sinh tồn (Cox Proportional Hazards)
    survival_model = TacticalSurvivalModel()
    survival_model.fit_model(features_df)
    
    # 4. Giả lập Counterfactuals bằng Monte Carlo
    simulator = CounterfactualSimulator(num_simulations=500)
    sim_results = simulator.run_simulation(features_df)
    sim_results.to_csv('./data/processed/monte_carlo_results.csv', index=False)
    
    print("\n[+] Toàn bộ Pipeline đã hoàn tất. Kết quả được lưu trong thư mục './data/processed/'.")

if __name__ == "__main__":
    main()