# -*- coding: utf-8 -*-
"""
Created on Tue Feb 24 21:43:52 2026

@author: flyro
"""

# feature_engineering.py - Creación de características
import pandas as pd
import numpy as np
from src.config import LAGS, DATE_COL, TARGET

def create_date_features(df):
    """Crea características basadas en fechas"""
    df = df.copy()
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    
    df["Year"] = df[DATE_COL].dt.year
    df["mes"] = df[DATE_COL].dt.month
    df["mes_nombre"] = df[DATE_COL].dt.month_name()
    df['day'] = df[DATE_COL].dt.day
    df['day_of_week'] = df[DATE_COL].dt.dayofweek
    df['week_of_year'] = df[DATE_COL].dt.isocalendar().week.astype(int)
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    
    return df

def create_cyclical_features(df):
    """Crea caracteristicas ciclicas (seno/coseno)"""
    df = df.copy()
    
    # ciclo mensual
    df["month_sin"] = np.sin(2 * np.pi * df["mes"]/12)
    df["month_cos"] = np.cos(2 * np.pi * df["mes"]/12) 
    
    df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    
    return df

def create_lag_features(df, lags=LAGS):
    """Crea características de rezago"""
    df = df.copy()
    df = df.sort_values(by=DATE_COL)
    
    for lag in lags:
        df[f"lag_{lag}"] = df[TARGET].shift(lag)
    
    return df
    
def create_rolling_features(df, windows=[7,14,28]):
    """Crea características de medias móviles"""
    df = df.copy()
    df = df.sort_values(by=DATE_COL)
    
    for window in windows:
        df[f"rolling_mean_{window}"] = df[TARGET].shift(1).rolling(window).mean()
        df[f"rolling_std_{window}"] = df[TARGET].shift(1).rolling(window).std()
    
    return df

def clean_state_holiday(df):
    """Limpia la columna StateHoliday convirtiendo a formato uniforme"""
    df = df.copy()
    
    if 'StateHoliday' in df.columns:
        # Convertir todo a string primero
        df['StateHoliday'] = df['StateHoliday'].astype(str)
        
        # Reemplazar '0' por '0' (ya está bien)
        # Los valores 'a', 'b', 'c' se mantienen como strings
        
        # Opción 1: Mantener como categorías (recomendado para XGBoost)
        df['StateHoliday'] = df['StateHoliday'].astype('category')
        
        # Opción 2: Convertir a numérico (one-hot encoding después)
        # holiday_map = {'0': 0, 'a': 1, 'b': 2, 'c': 3}
        # df['StateHoliday'] = df['StateHoliday'].map(holiday_map)
        
        print(f"Valores únicos en StateHoliday: {df['StateHoliday'].unique()}")
    
    return df  


def prepare_features(df):
    """Pipeline completo de feature engineering"""
    
    df = clean_state_holiday(df)
    df = create_date_features(df)
    df = create_cyclical_features(df)
    df = create_lag_features(df)
    df = create_rolling_features(df)
    
    return df

