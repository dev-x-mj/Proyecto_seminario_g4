import streamlit as st
import requests # Para hacer peticiones HTTP al API
import pandas as pd

# --- 1. Configuración de la Página y API ---
st.set_page_config(
    page_title="Demand Planning Dashboard v2.0",
    page_icon="🚀",
    layout="wide"
)

# URL del Backend (FastAPI)
API_URL = "http://127.0.0.1:8000"

# --- 2. Funciones de Comunicación con el API ---

@st.cache_data(ttl=600) 
def get_filters_from_api():
    """Obtiene las listas de categorías y regiones desde el API."""
    try:
        response = requests.get(f"{API_URL}/config/filters")
        response.raise_for_status() 
        data = response.json()
        return data.get("categories", ["All Categories"]), data.get("regions", ["All Regions"])
    except requests.exceptions.ConnectionError:
        st.error(f"Error de Conexión: No se pudo conectar al API (FastAPI) en {API_URL}. ¿Está el servidor Uvicorn corriendo?")
        return ["All Categories"], ["All Regions"]
    except Exception as e:
        st.error(f"Error al cargar filtros: {e}")
        return ["All Categories"], ["All Regions"]

def get_forecast_from_api(model_type, category, region, steps):
    """Obtiene el pronóstico desde el API para un modelo específico."""
    params = {
        "model_type": model_type,
        "category": category,
        "region": region,
        "steps": steps
    }
    try:
        response = requests.get(f"{API_URL}/sales/forecast", params=params)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as e:
        st.error(f"Error del API (Forecast {model_type}): {e.response.json().get('detail', 'Error desconocido')}")
        return None
    except requests.exceptions.ConnectionError:
        st.error(f"Error de Conexión: No se pudo conectar al API (FastAPI) en {API_URL}.")
        return None
    except Exception as e:
        st.error(f"Error inesperado al obtener pronóstico: {e}")
        return None

@st.cache_data(ttl=600)
def get_evaluation_from_api(model_type, category, region):
    """Obtiene las métricas de evaluación (backtest) desde el API."""
    params = {
        "model_type": model_type,
        "category": category,
        "region": region
    }
    try:
        response = requests.get(f"{API_URL}/sales/evaluation", params=params)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as e:
        st.error(f"Error del API (Evaluation {model_type}): {e.response.json().get('detail', 'Error desconocido')}")
        return None
    except requests.exceptions.ConnectionError:
        return None 
    except Exception as e:
        st.error(f"Error inesperado al obtener evaluación: {e}")
        return None

# --- 3. Funciones de Visualización (ELIMINADAS) ---
# ¡Ya no usamos la función plot_forecast() que causaba el bug!

# --- 4. Construcción de la Interfaz (UI) ---

st.title("🚀 Dashboard de Planificación de Demanda v2.0")
st.markdown("Sistema de pronóstico de ventas comparando **SARIMA vs. XGBoost**.")
st.sidebar.header("Filtros del Pronóstico")
CATEGORIES, REGIONS = get_filters_from_api()
selected_model_type = st.sidebar.radio(
    "Seleccionar Modelo:",
    options=['sarima', 'xgboost'],
    format_func=lambda x: "SARIMA (Estadístico)" if x == 'sarima' else "XGBoost (Machine Learning)",
    index=0
)
selected_category = st.sidebar.selectbox('Seleccionar Categoría:', options=CATEGORIES, index=0)
selected_region = st.sidebar.selectbox('Seleccionar Región:', options=REGIONS, index=0)
forecast_steps = st.sidebar.slider('Horizonte de Pronóstico (Meses):', min_value=6, max_value=36, value=12, step=1)
if st.sidebar.button('Generar Pronóstico'):
    model_name_display = "SARIMA" if selected_model_type == 'sarima' else "XGBoost"
    with st.spinner(f"Ejecutando modelo {model_name_display} para {selected_category} en {selected_region}..."):
        eval_response = get_evaluation_from_api(selected_model_type, selected_category, selected_region)
        api_response = get_forecast_from_api(selected_model_type, selected_category, selected_region, forecast_steps)
        st.subheader(f"Precisión del Modelo: {model_name_display}")
        st.markdown(f"*(Backtest sobre los últimos 12 meses de datos históricos)*")
        if eval_response and eval_response.get('status') == 'Success':
            col1, col2 = st.columns(2)
            col1.metric(
                label="Error Porcentual (MAPE)",
                value=f"{eval_response['mape']:.2f} %",
                help="Error promedio en porcentaje (Mean Absolute Percentage Error). Más bajo es mejor."
            )
            col2.metric(
                label="Error Absoluto (RMSE)",
                value=f"$ {eval_response['rmse']:,.2f}",
                help="Error promedio en dólares (Root Mean Squared Error). Más bajo es mejor."
            )
        else:
            st.warning("No se pudieron calcular las métricas de precisión (posiblemente por falta de datos históricos).")
        if api_response and api_response.get('status') == 'success':
            history_data = api_response.get('history')
            forecast_data = api_response.get('forecast')
            
            st.subheader("Gráfico de Pronóstico de Ventas Futuras")
            
            # --- ¡NUEVO CÓDIGO DE GRÁFICO (USA st.line_chart)! ---
            # 1. Preparar histórico (recibe strings "2014-01-31")
            history_df = pd.DataFrame(history_data)
            history_df['Date'] = pd.to_datetime(history_df['index'])
            history_df = history_df.set_index('Date').rename(columns={'data': 'Ventas Históricas'})
            
            # 2. Preparar pronóstico
            forecast_df = pd.DataFrame(forecast_data)
            forecast_df['Date'] = pd.to_datetime(forecast_df['Date'])
            forecast_df = forecast_df.set_index('Date').rename(columns={'Sales Forecast': f"Pronóstico ({model_name_display})"})
            
            # 3. Combinar en un solo DataFrame
            combined_df = pd.concat([history_df['Ventas Históricas'], forecast_df[f"Pronóstico ({model_name_display})"]], axis=1)

            # 4. ¡Graficar con st.line_chart()!
            st.line_chart(combined_df)
            # --- FIN DEL NUEVO CÓDIGO DE GRÁFICO ---

            st.subheader("Datos del Pronóstico (Primeros 12 Meses)")
            df_forecast_table = pd.DataFrame(forecast_data).head(12)
            st.dataframe(df_forecast_table.style.format({
                'Sales Forecast': '${:,.2f}',
                'Lower Bound': '${:,.2f}',
                'Upper Bound': '${:,.2f}'
                
            },na_rep="-"))
            with st.expander(f"Ver respuesta del API (JSON Pronóstico {model_name_display})"):
                st.json(api_response)
        elif api_response and api_response.get('status') == 'error':
            st.warning(f"No se encontraron datos para los filtros seleccionados: {selected_category} en {selected_region}.")
        else:
            st.error("No se pudo generar el pronóstico. Revisa los mensajes de error.")
else:
    st.info("Selecciona un modelo y filtros, y haz clic en 'Generar Pronóstico'.")