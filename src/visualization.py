# -*- coding: utf-8 -*-
"""
Created on Wed Feb 25 15:25:17 2026

@author: flyro
"""

# visualization.py - Funciones de visualización
import matplotlib.pyplot as plt
import pandas as pd
from src.config import FIGURES_DIR

def plot_sales_by_month(df):
    """Diagrama de cajas de ventas por mes"""
    plt.figure(figsize=(12, 6))
    df.boxplot(column='Sales', by='mes', grid=False)
    plt.title('Distribución de ventas por mes')
    plt.suptitle('')
    plt.xlabel('Mes')
    plt.ylabel('Ventas')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'ventas_por_mes.png')
    plt.show()


def plot_sales_by_year(df):
    """Diagrama de cajas de ventas por año"""
    plt.figure(figsize=(12, 6))
    df.boxplot(column='Sales', by='Year', grid=False)
    plt.title('Distribución de ventas por año')
    plt.suptitle('')
    plt.xlabel('Año')
    plt.ylabel('Ventas')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'ventas_por_ano.png')
    plt.show()
    
    
def plot_sales_histogram_by_year(df):
    """Histograma de ventas por año"""
    plt.figure(figsize=(12, 6))
    for year in sorted(df['Year'].unique()):
        plt.hist(
            df[df['Year'] == year]['Sales'],
            bins=50,
            alpha=0.5,
            label=str(year)
        )
    plt.title('Histograma de ventas por año')
    plt.xlabel('Ventas')
    plt.ylabel('Frecuencia')
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'histograma_ventas.png')
    plt.show()
    
def plot_time_series(df):
    """Serie temporal completa"""
    ventas_mensuales = (df.groupby(['Year', 'mes'], as_index=False)
                        .agg({'Sales': 'sum'}))
    ventas_mensuales['YearMonth'] = pd.to_datetime(
        ventas_mensuales['Year'].astype(str) + '-' +
        ventas_mensuales['mes'].astype(str) + '-01'
    )
    plt.figure(figsize=(12, 5))
    plt.plot(ventas_mensuales['YearMonth'], ventas_mensuales['Sales'])
    plt.title('Ventas mensuales (todas las tiendas)')
    plt.xlabel('Fecha')
    plt.ylabel('Ventas')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'serie_temporal.png')
    plt.show()

    
    