"""
Página de exploración de datos
"""
import streamlit as st
import pandas as pd
from src.data.data_loader import load_titanic_data, get_data_info, get_survival_stats
from src.visualization.plots import (
    plot_survival_distribution,
    plot_survival_by_feature,
    plot_age_distribution,
    plot_fare_distribution,
    plot_correlation_heatmap
)
from ui.styles.theme import render_header


def show():
    """Muestra la página de exploración"""
    
    render_header(
        "📊 Exploración de Datos",
        "Análisis exploratorio del dataset del Titanic"
    )
    
    # Cargar datos
    with st.spinner("Cargando datos del Titanic..."):
        df = load_titanic_data()
    
    if df is None:
        st.error("No se pudo cargar el dataset")
        return
    
    st.success("✅ Dataset cargado exitosamente")
    
    # Tabs para organizar contenido
    tab1, tab2, tab3, tab4 = st.tabs([
        "📋 Vista General",
        "📈 Supervivencia",
        "👥 Análisis Demográfico",
        "🔗 Correlaciones"
    ])
    
    # Tab 1: Vista General
    with tab1:
        st.subheader("Información del Dataset")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Pasajeros", len(df))
        with col2:
            st.metric("Características", len(df.columns))
        with col3:
            st.metric("Valores Nulos", df.isnull().sum().sum())
        with col4:
            st.metric("Duplicados", df.duplicated().sum())
        
        st.markdown("---")
        
        # Mostrar primeras filas
        st.subheader("Primeras 10 filas del dataset")
        st.dataframe(df.head(10), use_container_width=True)
        
        # Información de columnas
        st.subheader("Información de Columnas")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Tipos de Datos:**")
            st.dataframe(
                pd.DataFrame({
                    'Columna': df.columns,
                    'Tipo': df.dtypes.values,
                    'No Nulos': df.count().values
                }),
                use_container_width=True
            )
        
        with col2:
            st.markdown("**Valores Nulos por Columna:**")
            null_counts = df.isnull().sum()
            null_pct = (null_counts / len(df) * 100).round(2)
            st.dataframe(
                pd.DataFrame({
                    'Columna': null_counts.index,
                    'Nulos': null_counts.values,
                    'Porcentaje': null_pct.values
                }),
                use_container_width=True
            )
        
        # Estadísticas descriptivas
        st.subheader("Estadísticas Descriptivas")
        st.dataframe(df.describe(), use_container_width=True)
    
    # Tab 2: Supervivencia
    with tab2:
        st.subheader("Análisis de Supervivencia")
        
        # Estadísticas de supervivencia
        survival_stats = get_survival_stats(df)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Sobrevivieron",
                survival_stats['sobrevivieron'],
                f"{survival_stats['tasa_supervivencia']:.1f}%"
            )
        
        with col2:
            st.metric(
                "Murieron",
                survival_stats['murieron'],
                f"{survival_stats['tasa_mortalidad']:.1f}%",
                delta_color="inverse"
            )
        
        with col3:
            st.metric("Total Pasajeros", survival_stats['total_pasajeros'])
        
        st.markdown("---")
        
        # Gráfico de distribución
        st.plotly_chart(
            plot_survival_distribution(df),
            use_container_width=True
        )
        
        st.markdown("---")
        
        # Supervivencia por características
        st.subheader("Supervivencia por Características")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.plotly_chart(
                plot_survival_by_feature(df, 'Pclass', 'Supervivencia por Clase'),
                use_container_width=True
            )
        
        with col2:
            st.plotly_chart(
                plot_survival_by_feature(df, 'Sex', 'Supervivencia por Género'),
                use_container_width=True
            )
        
        st.plotly_chart(
            plot_survival_by_feature(df, 'Embarked', 'Supervivencia por Puerto de Embarque'),
            use_container_width=True
        )
        
        # Insights
        st.info("""
        **💡 Insights clave:**
        - La tasa de supervivencia general fue del ~38%
        - Las mujeres tuvieron una tasa de supervivencia significativamente mayor
        - Los pasajeros de primera clase tuvieron mejores probabilidades de supervivencia
        - El puerto de embarque también mostró diferencias en las tasas de supervivencia
        """)
    
    # Tab 3: Análisis Demográfico
    with tab3:
        st.subheader("Distribución de Edad")
        
        st.plotly_chart(plot_age_distribution(df), use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Estadísticas de Edad:**")
            age_stats = df['Age'].describe()
            st.dataframe(age_stats, use_container_width=True)
        
        with col2:
            st.markdown("**Edad por Supervivencia:**")
            age_by_survival = df.groupby('Survived')['Age'].describe()[['mean', 'std', 'min', 'max']]
            age_by_survival.index = ['Murieron', 'Sobrevivieron']
            st.dataframe(age_by_survival, use_container_width=True)
        
        st.markdown("---")
        
        st.subheader("Distribución de Tarifas")
        
        st.plotly_chart(plot_fare_distribution(df), use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Estadísticas de Tarifa:**")
            fare_stats = df['Fare'].describe()
            st.dataframe(fare_stats, use_container_width=True)
        
        with col2:
            st.markdown("**Tarifa por Supervivencia:**")
            fare_by_survival = df.groupby('Survived')['Fare'].describe()[['mean', 'std', 'min', 'max']]
            fare_by_survival.index = ['Murieron', 'Sobrevivieron']
            st.dataframe(fare_by_survival, use_container_width=True)
        
        st.warning("""
        **⚠️ Observaciones:**
        - Hay valores nulos en la columna Age que necesitan ser tratados
        - La distribución de tarifas es muy sesgada (algunos valores extremadamente altos)
        - Los sobrevivientes pagaron tarifas promedio más altas
        """)
    
    # Tab 4: Correlaciones
    with tab4:
        st.subheader("Matriz de Correlación")
        
        st.plotly_chart(plot_correlation_heatmap(df), use_container_width=True)
        
        st.markdown("""
        **🔍 Análisis de Correlaciones:**
        
        - **Fare y Pclass**: Correlación negativa fuerte (-0.55)
          - A mayor clase (3ra), menor tarifa pagada
        
        - **Survived y Fare**: Correlación positiva moderada (0.26)
          - Pasajeros que pagaron más tuvieron mayor probabilidad de supervivencia
        
        - **Survived y Pclass**: Correlación negativa (-0.34)
          - Pasajeros de primera clase tuvieron mayor supervivencia
        
        - **Age y Pclass**: Correlación positiva leve
          - Pasajeros más jóvenes tendían a estar en clases más altas
        """)
        
        st.success("""
        ✅ **Conclusión de la Exploración:**
        
        Los datos muestran patrones claros que pueden ser útiles para la predicción:
        - Género, clase y tarifa son indicadores importantes de supervivencia
        - Necesitamos manejar valores nulos, especialmente en Age
        - Las variables categóricas (Sex, Embarked) deben ser codificadas
        """)