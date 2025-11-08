import pandas as pd
import numpy as np
import os

# --- Constante de Ruta Absoluta ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, '..'))
FILE_NAME = os.path.join(PROJECT_ROOT, 'data', 'US Superstore data.xls')

def load_data():
    """Carga y preprocesa el dataset."""
    try:
        df = pd.read_excel(FILE_NAME)
        df.columns = df.columns.str.replace(' ', '_')
        df.columns = df.columns.str.replace('-', '_')
        df['Order_Date'] = pd.to_datetime(df['Order_Date'])
        return df, "Success"
    except FileNotFoundError:
        return None, f"Error: Archivo no encontrado en: {FILE_NAME}"
    except Exception as e:
        return None, f"Error al cargar datos: {e}"

def aggregate_sales(df, category="All Categories", region="All Regions"):
    """Filtra y agrega las ventas a frecuencia mensual (ME)."""
    if df is None:
        return pd.Series(dtype='float64'), False
        
    df_filtered = df.copy()

    if category != "All Categories":
        df_filtered = df_filtered[df_filtered['Category'] == category]
    if region != "All Regions":
        df_filtered = df_filtered[df_filtered['Region'] == region]

    if df_filtered.empty:
        return pd.Series(dtype='float64'), False
    
    df_filtered = df_filtered.set_index('Order_Date')
    ts_monthly = df_filtered['Sales'].resample('ME').sum()
    ts_monthly = ts_monthly[ts_monthly.index.min():]
    
    return ts_monthly, True

def create_features_for_ml(ts_data):
    """
    Crea características de ML (lags, mes, año) a partir de una serie de tiempo.
    """
    df = pd.DataFrame(ts_data.copy())
    df.columns = ['Sales']
    
    # Características de tiempo
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['quarter'] = df.index.quarter
    
    # Características de Lag (clave para ML en series de tiempo)
    # Usamos lag_12 para capturar la estacionalidad anual
    df['lag_12'] = df['Sales'].shift(12)
    
    # Rellenar NaNs (primeros 12 meses)
    df = df.bfill() # Llenar con el valor siguiente
    
    # Definir X (features) e y (target)
    X = df.drop('Sales', axis=1)
    y = df['Sales']
    
    return X, y