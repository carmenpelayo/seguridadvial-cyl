import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy
import plotly.express as px
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Cargar datos
st.cache_data()
def cargar_datos_accidentes():
    return pd.read_csv("accidentalidad-por-carreteras.csv", sep=";")  

st.cache_data()
def cargar_datos_trafico():
    return pd.read_csv("intensidades-medias-de-trafico-y-velocidades-red-regional-de-carreteras.csv", sep=";")  

# Configuración de la página
st.set_page_config(page_title="Dashboard de Carreteras", layout="wide")

# Cargar datasets
df_accidentes = cargar_datos_accidentes()
df_trafico = cargar_datos_trafico()

# Función para la homepage
def mostrar_homepage():
    st.title("🚧 Dashboard de Carreteras 🚧")
    st.subheader("Predicción de Zonas Críticas de Accidentes y Tráfico")
    
    st.markdown("""
    Este dashboard tiene como objetivo ayudar a los administradores de carreteras a identificar zonas críticas en términos de accidentes y tráfico, permitiendo una toma de decisiones más eficiente y preventiva.  
    
    Las predicciones se basan en modelos de machine learning que analizan datos históricos de tráfico y accidentes.
    """)

    # Gráfico de evolución de accidentes mortales y no mortales
    st.markdown("### Evolución de los accidentes al año")
    accidentes_anio = df_accidentes.groupby("AÑO").agg({"MUERTOS": "sum", "HERIDOS": "sum"}).reset_index()
    fig_accidentes = px.line(accidentes_anio, x="AÑO", y=["MUERTOS", "HERIDOS"], markers=True,
                             labels={"value": "Cantidad", "variable": "Tipo de accidente"},
                             title="Evolución de Accidentes Mortales y No Mortales")
    st.plotly_chart(fig_accidentes, use_container_width=True)
    
    # Descripción del equipo
    st.markdown("### Autores del Proyecto")
    st.markdown("""
    - **Autor 1**: Especialista en análisis de datos y modelos predictivos.
    - **Autor 2**: Ingeniero de tráfico con experiencia en análisis de redes viales.
    """)

# Función para la pestaña de predicción de zonas críticas de accidentes
st.cache_resource()
def prediccion_accidentes():
    st.markdown("## Predicción de Zonas Críticas de Accidentes")

    # Variables de entrada
    features_acc = ["LONG.", "IMD", "ASV", "ACV"]  # Ajusta las características
    X_acc = df_accidentes[features_acc]
    y_acc = df_accidentes["MUERTOS"]  # Usar como variable objetivo

    # Entrenamiento del modelo
    X_train_acc, X_test_acc, y_train_acc, y_test_acc = train_test_split(X_acc, y_acc, test_size=0.3, random_state=42)
    modelo_acc = RandomForestClassifier()
    modelo_acc.fit(X_train_acc, y_train_acc)
    y_pred_acc = modelo_acc.predict(X_test_acc)

    # Precisión del modelo
    accuracy_acc = accuracy_score(y_test_acc, y_pred_acc)
    st.metric(label="Precisión del modelo de predicción de accidentes", value=f"{accuracy_acc:.2%}")

# Función para la pestaña de predicción de zonas críticas de tráfico
st.cache_resource()
def prediccion_trafico():
    st.markdown("## Predicción de Zonas Críticas de Tráfico")

    # Variables de entrada
    features_traffic = ["VELOCIDAD MEDIA", "IMD AÑO", "% LIG AÑO", "% PES AÑO"]  # Ajusta según las características disponibles
    X_trafico = df_trafico[features_traffic]
    y_trafico = df_trafico["IMD AÑO"]  # Usar como variable objetivo para predecir intensidad de tráfico

    # Entrenamiento del modelo
    X_train_traffic, X_test_traffic, y_train_traffic, y_test_traffic = train_test_split(X_trafico, y_trafico, test_size=0.3, random_state=42)
    modelo_trafico = RandomForestClassifier()
    modelo_trafico.fit(X_train_traffic, y_train_traffic)
    y_pred_trafico = modelo_trafico.predict(X_test_traffic)

    # Precisión del modelo
    accuracy_traffic = accuracy_score(y_test_traffic, y_pred_trafico)
    st.metric(label="Precisión del modelo de predicción de tráfico", value=f"{accuracy_traffic:.2%}")

# Definir las pestañas
tab1, tab2, tab3 = st.tabs(["Homepage", "Zonas Críticas de Accidentes", "Zonas Críticas de Tráfico"])

with tab1:
    mostrar_homepage()

with tab2:
    prediccion_accidentes()

with tab3:
    prediccion_trafico()
