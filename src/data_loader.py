# -*- coding: utf-8 -*-
"""
Created on Tue Feb 24 15:30:04 2026

@author: flyro
"""

import pandas as pd
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR

def load_raw_data(filename="train.csv"):
    file_path = RAW_DATA_DIR / filename
    return pd.read_csv(file_path)

def save_processed_data(df, filename="processed_data.csv"):
    file_path = PROCESSED_DATA_DIR / filename
    df.to_parquet(file_path, index=False)
    print("datos guardados en: ", file_path)
    
def load_processed_data(filename="processed_data.parquet"):
    file_path = PROCESSED_DATA_DIR / filename
    return pd.read_parquet(file_path)