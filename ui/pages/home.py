"""
Página de inicio
"""
import streamlit as st
from ui.styles.theme import render_header


def show():
    """Muestra la página de inicio"""
    
    render_header(
        "🚢 Análisis de Supervivencia del Titanic",
        "Machine Learning con Decision Tree y Random Forest"
    )
    
    # Introducción
    st.markdown("""
    ## 📖 Bienvenido al Proyecto
    
    Este proyecto analiza los datos del **RMS Titanic** utilizando técnicas de Machine Learning
    para predecir la supervivencia de los pasajeros. Exploramos dos algoritmos principales:
    **Árbol de Decisión** y **Random Forest**.
    """)
    
    # Columnas para información
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🎯 Objetivos
        
        - Explorar y analizar el dataset del Titanic
        - Preprocesar datos y realizar feature engineering
        - Entrenar modelos de clasificación
        - Comparar el rendimiento de ambos modelos
        - Crear un predictor interactivo
        """)
    
    with col2:
        st.markdown("""
        ### 📊 Dataset
        
        El dataset contiene información de **891 pasajeros**:
        - **Características demográficas**: edad, género, clase
        - **Información familiar**: hermanos, padres/hijos
        - **Datos del viaje**: tarifa, puerto de embarque
        - **Variable objetivo**: supervivencia (0/1)
        """)
    
    st.markdown("---")
    
    # Contexto histórico
    st.markdown("""
    ## 🌊 Contexto Histórico
    
    El **RMS Titanic** fue un transatlántico británico que se hundió en el Océano Atlántico Norte
    en las primeras horas del 15 de abril de 1912, después de chocar con un iceberg durante su
    viaje inaugural desde Southampton a Nueva York.
    
    De las aproximadamente **2,224 personas a bordo**, más de **1,500 murieron**, convirtiéndolo
    en uno de los desastres marítimos más mortales de la historia moderna en tiempos de paz.
    """)
    
    # Información de los modelos
    st.markdown("---")
    st.markdown("## 🤖 Modelos de Machine Learning")
    
    tab1, tab2 = st.tabs(["🌳 Árbol de Decisión", "🌲 Random Forest"])
    
    with tab1:
        st.markdown("""
        ### Árbol de Decisión
        
        Un **árbol de decisión** es un modelo de predicción que utiliza una estructura de árbol
        para tomar decisiones basadas en características de entrada.
        
        **Ventajas:**
        - ✅ Fácil de interpretar y visualizar
        - ✅ No requiere normalización de datos
        - ✅ Puede manejar datos numéricos y categóricos
        - ✅ Captura relaciones no lineales
        
        **Desventajas:**
        - ❌ Propenso al sobreajuste
        - ❌ Sensible a pequeños cambios en los datos
        - ❌ Puede crear árboles demasiado complejos
        
        **Responsable:** Harry Style
        """)
    
    with tab2:
        st.markdown("""
        ### Random Forest
        
        **Random Forest** es un método de ensemble que combina múltiples árboles de decisión
        para mejorar la precisión y reducir el sobreajuste.
        
        **Ventajas:**
        - ✅ Reduce el sobreajuste mediante promedio
        - ✅ Más robusto y estable
        - ✅ Maneja bien datasets grandes
        - ✅ Proporciona importancia de características
        
        **Desventajas:**
        - ❌ Menos interpretable que un árbol simple
        - ❌ Requiere más recursos computacionales
        - ❌ Puede ser lento en predicción
        
        **Responsable:** Tania
        """)
    
    st.markdown("---")
    
    # Navegación rápida
    st.markdown("## 🚀 Comienza la Exploración")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("""
        **📊 Paso 1**
        
        Explora los datos del Titanic y descubre patrones de supervivencia
        """)
    
    with col2:
        st.info("""
        **🛠️ Paso 2**
        
        Prepara y transforma los datos para el modelado
        """)
    
    with col3:
        st.info("""
        **🤖 Paso 3**
        
        Entrena modelos y compara su rendimiento
        """)
    
    st.markdown("---")
    
    # Footer
    st.markdown("""
    <div style='text-align: center; color: #7f8c8d; padding: 2rem;'>
        <p>Desarrollado para el curso: Sistemas Inteligentes</p>
        <p>Utiliza el menú lateral para navegar entre las secciones</p>
    </div>
    """, unsafe_allow_html=True)