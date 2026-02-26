# Proyecto de Forecasting de Ventas con XGBoost

![Python](https://img.shields.io/badge/Python-3.9-blue)
![XGBoost](https://img.shields.io/badge/XGBoost-1.7-orange)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.2-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

##  Descripción

Este proyecto implementa un modelo de **forecasting de ventas** utilizando **XGBoost** para predecir ventas futuras de una cadena de tiendas. El modelo utiliza características temporales, lags y medias móviles para capturar patrones estacionales y tendencias.

##  Resultados del Modelo

| Métrica | Valor | Comparación |
|---------|-------|-------------|
| **RMSE** | 2,685.70 | 27.58% mejor que modelo naive |
| **MAE** | 1,820.65 | - |
| **R²** | 0.501 | Explica 50% de la varianza |

###  Rendimiento vs Modelo Naive
- **Modelo Naive**: RMSE 3,708.66
- **Mejora**: **+27.58%**

##  Estructura del Proyecto
📦 forecasting-ventas
├── 📁 data/ # Datos del proyecto
│ ├── 📁 raw/ # Datos crudos (train.csv)
│ └── 📁 processed/ # Datos procesados
├── 📁 src/ # Código fuente
│ ├── 📄 config.py # Configuraciones y rutas
│ ├── 📄 data_loader.py # Carga y guardado de datos
│ ├── 📄 feature_engineering.py # Creación de características
│ ├── 📄 model.py # Entrenamiento y evaluación
│ └── 📄 visualization.py # Generación de gráficas
├── 📁 models/ # Modelos entrenados
├── 📁 reports/ # Resultados y gráficas
│ └── 📁 figures/ # Visualizaciones generadas
├── 📁 notebooks/ # Análisis exploratorio
├── 📄 main.py # Pipeline principal
├── 📄 requirements.txt # Dependencias
└── 📄 README.md # Este archivo

## 🔧 Características (Features)

El modelo utiliza las siguientes características:

### Temporales
- 📅 Año, mes, día de la semana
- 🏁 Fin de semana (binario)
- 🔄 Features cíclicas (seno/coseno para mes y día)
### Rezagos (Lags)
- ⏱️ lag_1, lag_7, lag_14, lag_28 (ventas de días anteriores)

### Estadísticas móviles
- 📊 Media móvil de 7 días
- 📈 Desviación estándar móvil de 7 días
## 🚀 Cómo Ejecutar

### 1. Clonar el repositorio
```bash
git clone https://github.com/tu-usuario/forecasting-ventas.git
cd forecasting-ventas

##  Conclusiones del Proyecto

###  Logros Alcanzados
-  Modelo funcional con **RMSE de 2,685.70** en datos de test
-  **Mejora del 27.58%** respecto al modelo naive (predicción del día anterior)
-  Modelo con **mínimo sobreajuste** (diferencia de solo 87 RMSE entre train y test)
-  Pipeline completo y automatizado desde carga de datos hasta visualización

###  Aprendizajes Clave
1. **Importancia de features temporales**: Las variables de mes, día de semana y fines de semana fueron cruciales para capturar estacionalidad
2. **Features cíclicas**: Usar seno/coseno para mes y día mejoró la captura de patrones periódicos
3. **Lags estratégicos**: Los rezagos de 1, 7, 14 y 28 días capturaron patrones diarios, semanales y mensuales
4. **Medias móviles**: Ayudaron a suavizar el ruido y capturar tendencias

###  Limitaciones del Modelo
- El R² de 0.50 indica que el modelo explica el 50% de la varianza, hay margen de mejora
- No se incluyeron variables externas como clima o días festivos regionales
- El modelo podría beneficiarse de más datos históricos

###  Aplicaciones Prácticas
Este modelo podría utilizarse para:
- **Gestión de inventarios**: Predecir demanda futura y optimizar stock
- **Planificación de personal**: Anticipar semanas con mayor volumen de ventas
- **Estrategias promocionales**: Evaluar impacto de promociones en ventas futuras