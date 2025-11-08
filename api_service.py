from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from typing import Dict
import json

# --- 1. Importaciones desde los NUEVOS módulos de 'src' ---
from src.data_processing import load_data, aggregate_sales
from src.sarima_model import get_sarima_forecast, run_backtest_sarima
from src.xgboost_model import get_xgboost_forecast, run_backtest_xgboost

# --- 2. Inicialización de la Aplicación y Carga de Datos ---
app = FastAPI(
    title="Retail Forecasting API",
    description="API para obtener pronósticos de ventas (SARIMA y XGBoost) filtrado por categoría y región.",
    version="2.0.0" # ¡Versión 2.0 ahora!
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DF_RAW, STATUS = load_data()

if DF_RAW is None:
    print(f"FATAL ERROR: Datos no cargados. Razón: {STATUS}")
else:
    print("Datos cargados y preprocesados exitosamente.")
    CATEGORIES = ['All Categories'] + sorted(DF_RAW['Category'].unique().tolist())
    REGIONS = ['All Regions'] + sorted(DF_RAW['Region'].unique().tolist())


# --- 3. Endpoints del API ---

@app.get("/")
def read_root():
    return {"message": "Welcome to the Retail Forecasting API. Access /docs for documentation."}

@app.get("/config/filters")
def get_filters():
    """Devuelve las listas de filtros para poblar los selectores del frontend."""
    if DF_RAW is None:
        raise HTTPException(status_code=500, detail=f"Datos no cargados. Razón: {STATUS}")
    return {"categories": CATEGORIES, "regions": REGIONS}


@app.get("/sales/forecast", response_model=Dict)
def sales_forecast_endpoint(
    model_type: str = Query("sarima", description="El modelo a usar: 'sarima' o 'xgboost'"),
    category: str = Query("All Categories", description="Categoría del producto."),
    region: str = Query("All Regions", description="Región geográfica."),
    steps: int = Query(12, description="Número de meses a pronosticar.")
):
    """
    Endpoint dinámico que genera un pronóstico futuro usando el modelo seleccionado.
    """
    if DF_RAW is None:
         raise HTTPException(status_code=500, detail=f"Error de carga de datos inicial: {STATUS}")

    ts_history, data_available = aggregate_sales(df=DF_RAW, category=category, region=region)
    if not data_available:
        return {"status": "error", "message": f"No data found for {category}/{region}."}
    print(f"📅 Rango de fechas para {category}/{region}:",
      ts_history.index.min(), "→", ts_history.index.max())
    

    # --- ENRUTADOR DE MODELO ---
    if model_type == "sarima":
        forecast_df, status = get_sarima_forecast(ts_history, steps)
    elif model_type == "xgboost":
        forecast_df, status = get_xgboost_forecast(ts_history, steps)
    else:
        raise HTTPException(status_code=400, detail="model_type debe ser 'sarima' o 'xgboost'")
        
    if status != "Success" or forecast_df is None:
         raise HTTPException(status_code=400, detail=f"Error en el modelo {model_type}: {status}")

    # --- ¡AQUÍ ESTÁ EL BUG! ---
    # El valor .value de Timestamp es en NANOSEGUNDOS.
    # Para obtener MILISEGUNDOS, debemos dividir por 1,000,000 (10**6).
    # Si dividimos por 10**9, enviamos SEGUNDOS, causando el error de 1971.
    history_json = {
        "index": [i.strftime("%Y-%m-%d") for i in ts_history.index],
        "data": ts_history.values.tolist()
    }

    # --- FIN DE LA CORRECCIÓN ---

    forecast_df.index = forecast_df.index.strftime('%Y-%m-%d')
    forecast_json = forecast_df.reset_index().rename(columns={'index': 'Date'}).to_dict(orient='records')

    return {
        "status": "success",
        "model_used": model_type,
        "history": history_json,
        "forecast": forecast_json
    }

@app.get("/sales/evaluation", response_model=Dict)
def sales_evaluation_endpoint(
    model_type: str = Query("sarima", description="El modelo a evaluar: 'sarima' o 'xgboost'"),
    category: str = Query("All Categories", description="Categoría del producto."),
    region: str = Query("All Regions", description="Región geográfica.")
):
    """
    Realiza un backtest del modelo seleccionado y devuelve las métricas de error.
    """
    if DF_RAW is None:
        raise HTTPException(status_code=500, detail=f"Error de carga de datos inicial: {STATUS}")

    ts_history, data_available = aggregate_sales(df=DF_RAW, category=category, region=region)
    if not data_available:
        return {"status": "error", "message": f"No data found for {category}/{region}."}

    # --- ENRUTADOR DE MODELO (NUEVO) ---
    if model_type == "sarima":
        metrics = run_backtest_sarima(ts_history, test_months=12)
    elif model_type == "xgboost":
        metrics = run_backtest_xgboost(ts_history, test_months=12)
    else:
        raise HTTPException(status_code=400, detail="model_type debe ser 'sarima' o 'xgboost'")
    # --- FIN DEL ENRUTADOR ---

    if metrics["status"] != "Success":
        raise HTTPException(status_code=400, detail=metrics["message"])

    metrics["model_used"] = model_type
    return metrics