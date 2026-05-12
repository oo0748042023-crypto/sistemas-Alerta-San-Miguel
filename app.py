import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from datetime import timedelta

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(page_title="SAT - San Miguel", layout="wide", page_icon="🔥")

# Título y Estilo
st.title("🔥 Sistema de Alerta Temprana (SAT) - San Miguel")
st.markdown("---")

# --- CARGA DE DATOS ---
@st.cache_data
def load_data():
    # Saltamos las primeras filas de metadatos del archivo de Open-Meteo
    df = pd.read_csv('open-meteo-13.46N88.19W118m.csv', skiprows=3)
    df['time'] = pd.to_datetime(df['time'])
    df.columns = ['Fecha', 'Temp_Max']
    return df

try:
    df = load_data()
except:
    st.error("Asegúrate de que el archivo 'open-meteo-13.46N88.19W118m.csv' esté en la misma carpeta.")
    st.stop()

# --- SIDEBAR (NAVEGACIÓN INTERACTIVA) ---
st.sidebar.header("⚙️ Configuración")
menu = st.sidebar.selectbox("Ir a:", ["Dashboard General", "Predicciones ML", "Análisis de Olas de Calor"])
umbral = st.sidebar.slider("Umbral de Alerta (°C)", 30.0, 42.0, 37.0)

# --- PROCESAMIENTO ML ---
def train_model(data):
    df_ml = data.copy()
    df_ml['mes'] = df_ml['Fecha'].dt.month
    df_ml['dia_año'] = df_ml['Fecha'].dt.dayofyear
    # Lags: memoria de los últimos 3 días
    for i in range(1, 4):
        df_ml[f'lag_{i}'] = df_ml['Temp_Max'].shift(i)
    df_ml = df_ml.dropna()
    
    X = df_ml[['mes', 'dia_año', 'lag_1', 'lag_2', 'lag_3']]
    y = df_ml['Temp_Max']
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model, df_ml

model, df_ml = train_model(df)

# --- LÓGICA DE NAVEGACIÓN ---

if menu == "Dashboard General":
    st.subheader("📊 Análisis Histórico de Temperaturas")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Promedio Anual", f"{df['Temp_Max'].mean():.2f} °C")
    col2.metric("Máxima Registrada", f"{df['Temp_Max'].max():.1f} °C")
    col3.metric("Días en Alerta", len(df[df['Temp_Max'] >= umbral]))

    fig = px.line(df, x='Fecha', y='Temp_Max', title="Evolución de Temperatura en San Miguel")
    fig.add_hline(y=umbral, line_dash="dash", line_color="red", annotation_text="UMBRAL DE RIESGO")
    st.plotly_chart(fig, use_container_width=True)

elif menu == "Predicciones ML":
    st.subheader("🔮 Pronóstico con Random Forest (Próximos 7 días)")
    
    # Generar fechas futuras
    ultima_fecha = df['Fecha'].max()
    fechas_futuras = [ultima_fecha + timedelta(days=i) for i in range(1, 8)]
    
    # Predecir paso a paso
    ultimas_temps = list(df['Temp_Max'].tail(3).values)
    preds = []
    
    for f in fechas_futuras:
        input_data = np.array([[f.month, f.dayofyear, ultimas_temps[-1], ultimas_temps[-2], ultimas_temps[-3]]])
        p = model.predict(input_data)[0]
        preds.append(p)
        ultimas_temps.append(p)
    
    res_df = pd.DataFrame({'Fecha': fechas_futuras, 'Predicción (°C)': preds})
    
    c1, c2 = st.columns([1, 2])
    with c1:
        st.write("Tabla de resultados:")
        st.dataframe(res_df.style.background_gradient(cmap='YlOrRd'))
    
    with c2:
        fig_p = px.bar(res_df, x='Fecha', y='Predicción (°C)', color='Predicción (°C)', color_continuous_scale='Reds')
        st.plotly_chart(fig_p)

elif menu == "Análisis de Olas de Calor":
    st.subheader("🔥 Frecuencia de Eventos Extremos")
    df['Mes_Nombre'] = df['Fecha'].dt.month_name()
    
    # Heatmap de meses vs temperatura
    fig_h = px.density_heatmap(df, x="Mes_Nombre", y="Temp_Max", title="Concentración de Calor por Mes",
                               color_continuous_scale='Viridis', nbinsy=20)
    st.plotly_chart(fig_h, use_container_width=True)
    st.info("Este mapa muestra en qué meses la temperatura tiende a agruparse por encima de lo normal.")