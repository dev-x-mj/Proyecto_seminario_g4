Proyecto de Tesis: Dashboard de Pronóstico de Ventas v2.0

Este proyecto es un sistema de analítica de software completo para el pronóstico de demanda, desarrollado como parte de un trabajo de titulación. La aplicación implementa una arquitectura de software desacoplada (Backend/Frontend) para predecir las ventas de una cadena de retail utilizando modelos estadísticos (SARIMA) y de Machine Learning (XGBoost).

Características Principales

Arquitectura Desacoplada: Un backend (FastAPI) para la lógica y el modelado, y un frontend (Streamlit) para la visualización.

Comparación de Modelos: Permite al usuario elegir y comparar dinámicamente el rendimiento de un modelo estadístico (SARIMA) contra un modelo de Machine Learning (XGBoost).

Pronóstico Dinámico: Filtra las ventas por Categoría y Región para generar pronósticos personalizados.

Validación de Precisión: Utiliza backtesting para calcular y mostrar métricas de error (MAPE y RMSE) para cada modelo, justificando la selección del modelo más preciso.

Análisis Exploratorio (EDA): Incluye un notebook (.ipynb) que analiza la serie de tiempo para identificar tendencia y estacionalidad, justificando la elección de los modelos.

Arquitectura del Proyecto

El proyecto está organizado en una estructura modular de 3 capas para una clara separación de responsabilidades.




Tecnologías Utilizadas

Python 3.11+

Backend (API): FastAPI, Uvicorn

Frontend (Dashboard): Streamlit

Modelado y Data Science:

Pandas (para manipulación de datos)

Statsmodels (para el modelo SARIMA)

XGBoost (para el modelo de Gradient Boosting)

Scikit-learn (para métricas y preprocesamiento de ML)

Jupyter Notebook (para el Análisis Exploratorio)

Soporte de Archivos: openpyxl, xlrd

Cómo Ejecutar el Proyecto

Para ejecutar esta aplicación, necesitarás tener dos terminales abiertas simultáneamente: una para el Backend (API) y otra para el Frontend (Dashboard).

1. Configuración Inicial (Solo la primera vez)

Clonar/Descargar: Asegúrate de que todos los archivos del proyecto estén en la carpeta Proyecto_seminario.

Abrir Terminal: Abre una terminal en la carpeta raíz del proyecto.

Crear Entorno Virtual:

python -m venv venv


Activar Entorno Virtual (Windows):

.\venv\Scripts\Activate


Instalar Dependencias:

pip install -r requirements.txt


2. Ejecución (Cada vez que inicies)

Terminal 1: Iniciar el Backend (API)

Abre una terminal en la raíz del proyecto.

Activa el entorno: .\venv\Scripts\Activate

Inicia el servidor Uvicorn:

uvicorn api_service:app --reload


Espera a ver el mensaje: Uvicorn running on http://127.0.0.1:8000

Deja esta terminal abierta.

Terminal 2: Iniciar el Frontend (Dashboard)

Abre una NUEVA terminal en la raíz del proyecto.

Activa el entorno: .\venv\Scripts\Activate

Inicia el servidor de Streamlit:

streamlit run app.py