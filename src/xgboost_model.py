import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from src.data_processing import create_features_for_ml # Importa desde nuestro nuevo módulo

def get_xgboost_forecast(ts_history, steps=12):
    """
    Entrena el modelo XGBoost y genera el pronóstico de 'steps' meses futuros.
    """
    try:
        if len(ts_history) < 24:
            return None, "Datos insuficientes para XGBoost (se requieren > 24 meses)."
            
        # 1. Crear features para todo el historial
        X, y = create_features_for_ml(ts_history)

        # 2. Entrenar el modelo
        model = XGBRegressor(objective='reg:squarederror', n_estimators=100)
        model.fit(X, y)

        # 3. Generar pronóstico futuro
        # Para hacer esto, necesitamos crear los features de los meses futuros
        last_date = ts_history.index[-1]
        future_dates = pd.date_range(start=last_date, periods=steps + 1, freq='ME')[1:]
        
        # Necesitamos el 'lag_12' para los meses futuros.
        # Combinamos historial y futuro para calcular el lag correctamente
        future_df = pd.DataFrame(index=future_dates)
        future_df['Sales'] = np.nan # Placeholder
        
        # Combinamos todo
        full_ts = pd.concat([ts_history, future_df['Sales']])
        
        # Creamos features para el set completo
        full_features_X, _ = create_features_for_ml(full_ts)
        
        # Seleccionamos solo las filas futuras para predecir
        X_future = full_features_X.iloc[-steps:]

        # Predecir
        predictions = model.predict(X_future)
        predictions = predictions.clip(min=0) # No predecir ventas negativas # No predecir ventas negativas
        
        # Crear DataFrame de pronóstico (simplificado, sin CI)
        forecast_df = pd.DataFrame({
            'Sales Forecast': predictions,
            'Lower Bound': np.nan, # XGBoost no da CI por defecto
            'Upper Bound': np.nan
        }, index=future_dates)

        return forecast_df, "Success"

    except Exception as e:
        return None, f"Error en el entrenamiento XGBoost: {e}"


def run_backtest_xgboost(ts_history, test_months=12):
    """
    Realiza un backtest del modelo XGBoost.
    """
    if len(ts_history) < (24 + test_months):
        return {
            "status": "Error",
            "message": f"Datos insuficientes para backtest XGBoost. Se necesitan > {24 + test_months} meses."
        }

    # 1. Crear features para todos los datos
    X, y = create_features_for_ml(ts_history)

    # 2. Dividir en Entrenamiento y Prueba
    X_train, X_test = X.iloc[:-test_months], X.iloc[-test_months:]
    y_train, y_test = y.iloc[:-test_months], y.iloc[-test_months:]

    try:
        # 3. Entrenar el modelo
        model = XGBRegressor(objective='reg:squarederror', n_estimators=100)
        model.fit(X_train, y_train)

        # 4. Predecir en el set de prueba
        predictions = model.predict(X_test)

        # 5. Calcular Métricas
        rmse = np.sqrt(np.mean((y_test.values - predictions)**2))
        mask = y_test.values != 0
        mape = np.mean(
            np.abs((y_test.values[mask] - predictions[mask]) / y_test.values[mask])
        ) * 100

        return {
            "status": "Success",
            "test_period_months": test_months,
            "mape": mape,
            "rmse": rmse
        }

    except Exception as e:
        return {"status": "Error", "message": f"Error en backtesting XGBoost: {e}"}