# -*- coding: utf-8 -*-
"""
Created on Wed Feb 25 16:03:31 2026

@author: flyro
"""

# main.py - Script principal para ejecutar todo el pipeline
import pandas as pd
from src.data_loader import load_raw_data, save_processed_data
from src.feature_engineering import prepare_features
from src.model import split_data, hyperparameter_tuning, evaluate_model, print_results
from src.visualization import (
    plot_sales_by_month, plot_sales_by_year,
    plot_sales_histogram_by_year, plot_time_series
)
from src.config import FEATURES, TARGET

def main():
    print(" Iniciando pipeline de forecasting...")
    
    # 1. Cargar datos
    print("\n Cargando datos...")
    df = load_raw_data("train.csv")
    print(f"   Datos cargados: {df.shape[0]} filas, {df.shape[1]} columnas")
    
    # 2. Feature engineering
    print("\n Creando características...")
    df = prepare_features(df)
    print(f"   Features creadas: {df.shape[1]} columnas totales")
    
    # 3. Visualizaciones EDA
    print("\n Generando visualizaciones...")
    plot_sales_by_month(df)
    plot_sales_by_year(df)
    plot_sales_histogram_by_year(df)
    plot_time_series(df)
    
    # 4. Eliminar nulos y preparar datos
    df_model = df.dropna().copy()
    
    # 5. Dividir datos
    print("\n✂️ Dividiendo datos en train/test...")
    X_train, X_test, y_train, y_test, train_df, test_df = split_data(df_model)
    print(f"   Train: {X_train.shape[0]} muestras")
    print(f"   Test: {X_test.shape[0]} muestras")
    
    # 6. Guardar datos procesados
    save_processed_data(df_model, "data_model.csv")
    print("   Datos procesados guardados")
    
    # 7. Entrenar modelo con tuning
    print("\n🤖 Entrenando modelo con búsqueda de hiperparámetros...")
    search_results = hyperparameter_tuning(X_train, y_train)
    
    print("\n✨ Mejores parámetros encontrados:")
    print(search_results.best_params_)
    
    # 8. Evaluar modelo
    print("\n📈 Evaluando modelo...")
    best_model = search_results.best_estimator_
    results = evaluate_model(best_model, X_train, X_test, y_train, y_test)
    
    # 9. Mostrar resultados
    print_results(results)
    
    print("\n Pipeline completado exitosamente!")

if __name__ == "__main__":
    main()
