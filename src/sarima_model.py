import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pandas as pd

# Parámetros estándar para SARIMA (pueden ser ajustados)
ORDER = (0, 1, 1)
SEASONAL_ORDER = (0, 1, 1, 12)

def get_sarima_forecast(ts_history, steps=12):
    """
    Entrena el modelo SARIMA y genera el pronóstico de 'steps' meses futuros.
    """
    try:
        if len(ts_history) < 24:
            return None, "Datos insuficientes para SARIMA (se requieren > 24 meses)."

        model = SARIMAX(
            ts_history,
            order=ORDER,
            seasonal_order=SEASONAL_ORDER,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        results = model.fit(disp=False) 

        forecast = results.get_forecast(steps=steps)
        forecast_df = forecast.summary_frame(alpha=0.05)

        forecast_df.rename(columns={
            'mean': 'Sales Forecast',
            'mean_ci_lower': 'Lower Bound',
            'mean_ci_upper': 'Upper Bound'
        }, inplace=True)
        if 'lower 95%' in forecast_df.columns:
            forecast_df.rename(columns={
                'lower 95%': 'Lower Bound',
                'upper 95%': 'Upper Bound'
            }, inplace=True)
            
        final_cols = ['Sales Forecast', 'Lower Bound', 'Upper Bound']
        forecast_df = forecast_df[final_cols].astype(float).round(2)
        forecast_df['Sales Forecast'] = forecast_df['Sales Forecast'].clip(lower=0)

        return forecast_df, "Success"

    except Exception as e:
        return None, f"Error en el entrenamiento SARIMA: {e}"


def run_backtest_sarima(ts_history, test_months=12):
    """
    Realiza un backtest del modelo SARIMA.
    """
    if len(ts_history) < (24 + test_months):
        return {
            "status": "Error",
            "message": f"Datos insuficientes para backtest SARIMA. Se necesitan > {24 + test_months} meses."
        }

    train_data = ts_history[:-test_months]
    test_data = ts_history[-test_months:]

    try:
        model = SARIMAX(
            train_data,
            order=ORDER,
            seasonal_order=SEASONAL_ORDER,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        results = model.fit(disp=False)
        forecast = results.get_forecast(steps=test_months)
        predictions = forecast.predicted_mean

        # Calcular Métricas
        rmse = np.sqrt(np.mean((test_data.values - predictions.values)**2))
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
        return {"status": "Error", "message": f"Error en backtesting SARIMA: {e}"}