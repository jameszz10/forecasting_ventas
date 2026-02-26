# -*- coding: utf-8 -*-
"""
Created on Wed Feb 25 15:05:59 2026

@author: flyro
"""

# model.py - Modelado y evaluación
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score
)
from src.config import FEATURES, TARGET, TEST_SPLIT_DATE, DATE_COL


def split_data(df, split_date=TEST_SPLIT_DATE):
    """Divide datos en train/test respetando el tiempo"""
    train = df[df[DATE_COL] < split_date].copy()
    test = df[df[DATE_COL] >= split_date].copy()
    
    X_train = train[FEATURES]
    y_train = train[TARGET]
    X_test = test[FEATURES]
    y_test = test[TARGET]
    
    return X_train, X_test, y_train, y_test, train, test

def create_model():
    """Crea el pipeline del modelo"""
    xgb_model = XGBRegressor(
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1
    )
    
    return Pipeline(steps=[("modelo", xgb_model)])

def hyperparameter_tuning(X_train, y_train):
    """Realiza búsqueda de hiperparámetros"""
    
    # 👇 SOLUCIÓN: Forzar TODOS los datos a float64
    print("\n🔧 Forzando tipos de datos a float64...")
    
    # Convertir todas las columnas a numérico, forzando errores a NaN
    for col in X_train.columns:
        X_train[col] = pd.to_numeric(X_train[col], errors='coerce')
    
    # Verificar si hay NaN después de la conversión
    nan_count = X_train.isna().sum().sum()
    if nan_count > 0:
        print(f"⚠️ Se encontraron {nan_count} valores NaN después de conversión")
        # Llenar NaN con 0 o con la media
        X_train = X_train.fillna(0)
    
    # Verificar tipos finales
    print("Tipos de datos finales:")
    print(X_train.dtypes.value_counts())
    
    # Asegurar que sea float64
    X_train = X_train.astype('float64')
    
    # Reducir parámetros para evitar problemas de memoria
    pipe = create_model()
    
    tscv = TimeSeriesSplit(n_splits=3)  # Reducir splits
    
    param_distributions = {
        "modelo__max_depth": [3, 5],  # Valores reducidos
        "modelo__learning_rate": [0.05, 0.1],
        "modelo__n_estimators": [100, 150],  # Reducido
        "modelo__subsample": [0.8],
        "modelo__colsample_bytree": [0.8]
    }
    
    random_search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_distributions,
        n_iter=5,  # Reducido de 20 a 5
        cv=tscv,
        scoring="neg_root_mean_squared_error",
        random_state=42,
        n_jobs=1,  # Importante: usar 1 proceso
        verbose=2,
    )
    
    print("\n🚀 Iniciando entrenamiento...")
    random_search.fit(X_train, y_train)
    return random_search

def evaluate_model(model, X_train, X_test, y_train, y_test):
    """Evalúa el modelo y calcula métricas"""
    # Predicciones
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Métricas de test
    rmse_test = mean_squared_error(y_test, y_pred_test, squared=False)
    mae_test = mean_absolute_error(y_test, y_pred_test)
    r2_test = r2_score(y_test, y_pred_test)
    
    # Métricas de train
    rmse_train = mean_squared_error(y_train, y_pred_train, squared=False)
    mae_train = mean_absolute_error(y_train, y_pred_train)
    r2_train = r2_score(y_train, y_pred_train)
    
    # Métricas naive para comparación
    y_pred_naive = np.roll(y_test.values, 1)
    y_pred_naive[0] = y_train.values[-1]
    rmse_naive = mean_squared_error(y_test, y_pred_naive, squared=False)
    improvement = (rmse_naive - rmse_test) / rmse_naive * 100
    
    # Resultados
    results = {
        'test': {'rmse': rmse_test, 'mae': mae_test, 'r2': r2_test},
        'train': {'rmse': rmse_train, 'mae': mae_train, 'r2': r2_train},
        'naive': {'rmse': rmse_naive},
        'improvement_vs_naive': improvement,
        'overfitting': {
            'rmse_diff': rmse_test - rmse_train,
            'mae_diff': mae_test - mae_train,
            'r2_diff': r2_test - r2_train
        }
    }
    
    return results

def print_results(results):
    """Imprime los resultados de evaluación"""
    print("=" * 50)
    print("RESULTADOS DEL MODELO")
    print("=" * 50)
    
    print("\n MÉTRICAS DE TEST:")
    print(f"  RMSE: {results['test']['rmse']:,.2f}")
    print(f"  MAE: {results['test']['mae']:,.2f}")
    print(f"  R²: {results['test']['r2']:.3f}")
    
    print("\n MÉTRICAS DE TRAIN:")
    print(f"  RMSE: {results['train']['rmse']:,.2f}")
    print(f"  MAE: {results['train']['mae']:,.2f}")
    print(f"  R²: {results['train']['r2']:.3f}")
    
    print("\n COMPARATIVA:")
    print(f"  RMSE Naive: {results['naive']['rmse']:,.2f}")
    print(f"  Mejora vs Naive: {results['improvement_vs_naive']:.2f}%")
    
    print("\n SOBREAJUSTE:")
    print(f"  Diferencia RMSE: {results['overfitting']['rmse_diff']:,.2f}")
    print(f"  Diferencia MAE: {results['overfitting']['mae_diff']:,.2f}")
    print(f"  Diferencia R²: {results['overfitting']['r2_diff']:.3f}")