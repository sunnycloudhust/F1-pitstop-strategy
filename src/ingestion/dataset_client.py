import pandas as pd
import os
from typing import Dict, Optional, List
from src.utils.logger import get_logger

class F1DatasetIngestor:    
    def __init__(self, data_dir: str = './data/raw'):
        self.logger = get_logger(self.__class__.__name__)
        self.data_dir = data_dir
        self.tables: Dict[str, pd.DataFrame] = {}

    def load_tables(self, table_names: List[str]) -> bool:
        success = True
        for name in table_names:
            file_path = os.path.join(self.data_dir, f"{name}.csv")
            if not os.path.exists(file_path):
                self.logger.error(f"Không tìm thấy file: {file_path}")
                success = False
                continue
                
            try:
                df = pd.read_csv(file_path, na_values='\\N')
                self.tables[name] = df
                self.logger.info(f"Đã tải bảng '{name}' ({len(df)} dòng).")
            except Exception as e:
                self.logger.error(f"Lỗi khi đọc bảng '{name}': {e}")
                success = False
        return success
        
    def get_table(self, table_name: str) -> Optional[pd.DataFrame]:
        return self.tables.get(table_name)