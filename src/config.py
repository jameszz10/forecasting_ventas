# -*- coding: utf-8 -*-
"""
Created on Tue Feb 24 15:18:10 2026

@author: flyro
"""

from pathlib import Path 

# directorios base
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR/"data"
RAW_DATA_DIR = DATA_DIR/ "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Configuración de datos
TARGET = "Sales"
DATE_COL = "Date"

FEATURES = [
    "Year", "mes", "day_of_week", "is_weekend",
    "lag_1", "lag_7", "lag_14", "lag_28",
    "rolling_mean_7", "rolling_std_7",
    "month_sin", "month_cos", "dow_sin", "dow_cos"
]

# Parámetros del modelo
LAGS = [1, 7, 14, 28]
ROLLING_WINDOWS = [7, 14, 28]
TEST_SPLIT_DATE = "2015-06-01"