import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
#from mpl_toolkits.basemap import Basemap
import numpy
#import plotly.express as px
#import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Cargar datos
@st.cache
def cargar_datos_accidentes():
    return pd.read_csv("accidentalidad-por-carreteras.csv", sep=";")  

@st.cache
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
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(accidentes_anio['AÑO'], accidentes_anio['MUERTOS'], marker='o', label='Muertos')
    ax.plot(accidentes_anio['AÑO'], accidentes_anio['HERIDOS'], marker='o', label='Heridos')
    ax.set_xlabel('Año')
    ax.set_ylabel('Número de Accidentes')
    ax.set_title('Evolución de Accidentes Mortales y No Mortales')
    ax.legend(title='Tipo de accidente')
    st.pyplot(fig)

    # Descripción del equipo
    st.markdown("### Autores del Proyecto")
    st.markdown("""
    - **Autor 1**: Especialista en análisis de datos y modelos predictivos.
    - **Autor 2**: Ingeniero de tráfico con experiencia en análisis de redes viales.
    """)

# Función para la pestaña de predicción de zonas críticas de accidentes
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

    '''
    # Mapa interactivo con las zonas críticas de accidentes
    fig_accidentes = px.scatter_mapbox(
        df_accidentes,
        lat="LONG.",  # Reemplaza con latitud si está disponible
        lon="LONG.",  # Reemplaza con longitud
        color="MUERTOS",  # Color según muertes
        size="HERIDOS",
        hover_data=["DESCRIPCIÓN"],
        color_continuous_scale=px.colors.sequential.Reds,
        mapbox_style="carto-positron",
        zoom=5,
        title="Mapa de Accidentes"
    )
    st.plotly_chart(fig_accidentes, use_container_width=True)
    '''

# Función para la pestaña de predicción de zonas críticas de tráfico
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

    '''
    # Mapa interactivo con las zonas críticas de tráfico
    fig_trafico = px.scatter_mapbox(
        df_trafico,
        lat="LAT",  # Reemplaza con latitud si está disponible
        lon="LONG",  # Reemplaza con longitud
        color="IMD AÑO",
        size="VELOCIDAD MEDIA",
        hover_data=["DESCRIPCIÓN DEL TRAMO"],
        color_continuous_scale=px.colors.sequential.Blues,
        mapbox_style="open-street-map",
        zoom=5,
        title="Mapa de Zonas Críticas de Tráfico"
    )
    st.plotly_chart(fig_trafico, use_container_width=True)
    '''

# Definir las pestañas
tab1, tab2, tab3 = st.tabs(["Homepage", "Zonas Críticas de Accidentes", "Zonas Críticas de Tráfico"])

with tab1:
    mostrar_homepage()

with tab2:
    prediccion_accidentes()

with tab3:
    prediccion_trafico()
