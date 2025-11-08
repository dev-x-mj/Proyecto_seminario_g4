
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX
import os # Importar 'os' para construir rutas absolutas

# --- Constante de Ruta Absoluta (Solución Robusta) ---
# 1. Obtener la ruta absoluta del script actual (forecasting_model.py)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# 2. Subir un nivel a 'src', y otro a la raíz del proyecto
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, '..'))
# 3. Construir la ruta absoluta al archivo de datos
FILE_NAME = os.path.join(PROJECT_ROOT, 'data', 'US Superstore data.xls')
# ----------------------------------------------------

def load_data():
    """
    Carga el dataset de Excel, lo limpia y retorna el DataFrame.
    Retorna (DataFrame, status_message).
    """
    try:
        # Usar la ruta absoluta
        df = pd.read_excel(FILE_NAME)
        
        # Renombrar columnas para facilitar el acceso
        df.columns = df.columns.str.replace(' ', '_')
        df.columns = df.columns.str.replace('-', '_')
        
        # Convertir la columna de fecha a datetime
        df['Order_Date'] = pd.to_datetime(df['Order_Date'])
        
        return df, "Success"
    except FileNotFoundError:
        # El mensaje de error ahora mostrará la ruta absoluta, facilitando la depuración
        return None, f"Error: Archivo no encontrado en la ruta esperada: {FILE_NAME}"
    except Exception as e:
        return None, f"Error al cargar o preprocesar los datos: {e}"

def aggregate_sales(df, category="All Categories", region="All Regions"):
    """
    Filtra el DataFrame y agrega las ventas a nivel Mensual (M).
    Retorna (TimeSeries, data_available_boolean).
    """
    # Esta comprobación evita el error si df es None
    if df is None:
        return pd.Series(dtype='float64'), False
        
    df_filtered = df.copy()

    # Aplicar filtros
    if category != "All Categories":
        df_filtered = df_filtered[df_filtered['Category'] == category]
    if region != "All Regions":
        df_filtered = df_filtered[df_filtered['Region'] == region]

    if df_filtered.empty:
        return pd.Series(dtype='float64'), False
    
    # Establecer la fecha como índice
    df_filtered = df_filtered.set_index('Order_Date')
    
    # Agregar las ventas a frecuencia Mensual ('M')
    ts_monthly = df_filtered['Sales'].resample('ME').sum()
    
    # Eliminar valores cero al inicio, si los hay
    ts_monthly = ts_monthly[ts_monthly.index.min():]
    
    return ts_monthly, True

def get_sarima_forecast(ts_history, steps=12):
    """
    Entrena el modelo SARIMA y genera el pronóstico de 'steps' meses futuros.
    Retorna (DataFrame del pronóstico, status_message).
    """
    try:
        # Asegurarse de que la serie de tiempo no sea trivialmente pequeña
        if len(ts_history) < 24: # Necesita al menos 2 años de datos para estacionalidad (12)
            return None, "Datos insuficientes para el entrenamiento SARIMA. Se requiere más de 24 meses de datos."

        # Parámetros SARIMA
        order = (0, 1, 1)
        seasonal_order = (0, 1, 1, 12)

        # Entrenar el modelo
        model = SARIMAX(
            ts_history,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        
        results = model.fit(disp=False) 

        # Generar el pronóstico
        forecast = results.get_forecast(steps=steps)
        forecast_df = forecast.summary_frame(alpha=0.05) # 95% CI

        # Limpiar el DataFrame de salida
        forecast_df.rename(columns={
            'mean': 'Sales Forecast',
            'mean_se': 'Std Error',
            'mean_ci_lower': 'Lower Bound',
            'mean_ci_upper': 'Upper Bound'
        }, inplace=True)

        # Manejar nombres de columnas alternativos (depende de la versión de statsmodels)
        if 'lower 95%' in forecast_df.columns:
            forecast_df.rename(columns={
                'lower 95%': 'Lower Bound',
                'upper 95%': 'Upper Bound'
            }, inplace=True)
            
        final_cols = ['Sales Forecast', 'Lower Bound', 'Upper Bound']
        forecast_df = forecast_df[final_cols]
        
        # Convertir a flotantes y redondear
        forecast_df = forecast_df.astype(float).round(2)

        # Asegurar que el pronóstico no sea negativo
        forecast_df['Sales Forecast'] = forecast_df['Sales Forecast'].clip(lower=0)

        return forecast_df, "Success"

    except Exception as e:
        return None, f"Error en el entrenamiento o pronóstico del modelo SARIMA: {e}"
    
    # --- Contenido existente de forecasting_model.py ---
# ... (load_data, aggregate_sales, get_sarima_forecast) ...

# --- AÑADIR ESTA NUEVA FUNCIÓN AL FINAL ---

def run_backtest(ts_history, test_months=12):
    """
    Realiza un backtest del modelo SARIMA, entrenando con datos históricos
    y probando contra los últimos 'test_months' meses.
    
    Retorna un diccionario con las métricas de error (MAPE, RMSE).
    """
    
    # 1. Validación de datos suficientes
    # Necesitamos al menos 2 años para entrenar (24) + los meses de prueba
    if len(ts_history) < (24 + test_months):
        return {
            "status": "Error",
            "message": f"Datos insuficientes para backtest. Se necesitan > {24 + test_months} meses, se tienen {len(ts_history)}."
        }

    # 2. Dividir en Entrenamiento y Prueba
    train_data = ts_history[:-test_months]
    test_data = ts_history[-test_months:]

    # 3. Entrenar el modelo (mismos parámetros que get_sarima_forecast)
    try:
        model = SARIMAX(
            train_data,
            order=(0, 1, 1),
            seasonal_order=(0, 1, 1, 12),
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        results = model.fit(disp=False)

        # 4. Generar pronóstico para el período de prueba
        forecast = results.get_forecast(steps=test_months)
        predictions = forecast.predicted_mean

        # 5. Calcular Métricas de Error
        
        # RMSE (Root Mean Squared Error) - Mide la magnitud del error en USD
        rmse = np.sqrt(np.mean((test_data.values - predictions.values)**2))

        # MAPE (Mean Absolute Percentage Error) - Mide el error en porcentaje
        # Manejo de división por cero si las ventas reales fueron 0
        mask = test_data.values != 0
        mape = np.mean(
            np.abs((test_data.values[mask] - predictions.values[mask]) / test_data.values[mask])
        ) * 100

        return {
            "status": "Success",
            "test_period_months": test_months,
            "mape": mape,
            "rmse": rmse
        }

    except Exception as e:
        return {
            "status": "Error",
            "message": f"Error durante el backtesting: {e}"
        }