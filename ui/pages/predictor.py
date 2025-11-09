"""
Página de predictor interactivo
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from ui.styles.theme import render_header


def show():
    """Muestra la página de predictor interactivo"""
    
    render_header(
        "🔮 Predictor Interactivo",
        "Simula un pasajero y predice su supervivencia"
    )
    
    # Verificar que al menos un modelo esté entrenado
    dt_trained = st.session_state.get('dt_trained', False)
    rf_trained = st.session_state.get('rf_trained', False)
    
    if not dt_trained and not rf_trained:
        st.error("""
        ❌ **Ningún modelo está entrenado**
        
        Necesitas entrenar al menos un modelo antes de hacer predicciones:
        - 🌳 Árbol de Decisión (Harry)
        - 🌲 Random Forest (Tania)
        """)
        return
    
    models_available = []
    if dt_trained:
        models_available.append("Decision Tree")
    if rf_trained:
        models_available.append("Random Forest")
    
    st.success(f"✅ Modelos disponibles: {', '.join(models_available)}")
    
    st.markdown("---")
    
    # Instrucciones
    st.markdown("""
    ## 👤 Crea un Perfil de Pasajero
    
    Completa la información del pasajero para predecir su probabilidad de supervivencia
    en el Titanic. Ajusta los parámetros y observa cómo cambia la predicción.
    """)
    
    # Formulario de entrada
    with st.form("passenger_form"):
        st.subheader("Información del Pasajero")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Datos demográficos
            st.markdown("#### 📋 Datos Demográficos")
            
            pclass = st.selectbox(
                "Clase del Ticket",
                options=[1, 2, 3],
                index=2,
                help="1ra clase = Lujo, 3ra clase = Económica"
            )
            
            sex = st.radio(
                "Género",
                options=["male", "female"],
                index=0,
                horizontal=True
            )
            
            age = st.slider(
                "Edad",
                min_value=0,
                max_value=80,
                value=30,
                help="Edad del pasajero en años"
            )
        
        with col2:
            # Información familiar y viaje
            st.markdown("#### 👨‍👩‍👧‍👦 Familia y Viaje")
            
            sibsp = st.number_input(
                "Hermanos/Cónyuges a Bordo",
                min_value=0,
                max_value=8,
                value=0,
                help="Número de hermanos o cónyuge viajando juntos"
            )
            
            parch = st.number_input(
                "Padres/Hijos a Bordo",
                min_value=0,
                max_value=6,
                value=0,
                help="Número de padres o hijos viajando juntos"
            )
            
            fare = st.number_input(
                "Tarifa Pagada (£)",
                min_value=0.0,
                max_value=500.0,
                value=32.0,
                step=0.1,
                help="Precio del ticket en libras esterlinas"
            )
            
            embarked = st.selectbox(
                "Puerto de Embarque",
                options=["S", "C", "Q"],
                index=0,
                help="S=Southampton, C=Cherbourg, Q=Queenstown"
            )
        
        # Botón de predicción
        submitted = st.form_submit_button("🔮 Predecir Supervivencia", type="primary", use_container_width=True)
    
    if submitted:
        # Preparar datos del pasajero
        passenger_data = {
            'Pclass': pclass,
            'Sex': 0 if sex == 'male' else 1,
            'Age': age,
            'SibSp': sibsp,
            'Parch': parch,
            'Fare': fare,
            'Embarked_C': 1 if embarked == 'C' else 0,
            'Embarked_Q': 1 if embarked == 'Q' else 0,
            'Embarked_S': 1 if embarked == 'S' else 0
        }
        
        # Crear DataFrame con el orden correcto de columnas
        feature_names = st.session_state.get('feature_names', list(passenger_data.keys()))
        passenger_df = pd.DataFrame([passenger_data])[feature_names]
        
        st.markdown("---")
        st.markdown("## 📊 Resultados de la Predicción")
        
        # Realizar predicciones con ambos modelos si están disponibles
        predictions = {}
        probabilities = {}
        
        if dt_trained:
            dt_model = st.session_state['dt_model']
            predictions['Decision Tree'] = dt_model.predict(passenger_df)[0]
            probabilities['Decision Tree'] = dt_model.predict_proba(passenger_df)[0]
        
        if rf_trained:
            rf_model = st.session_state['rf_model']
            predictions['Random Forest'] = rf_model.predict(passenger_df)[0]
            probabilities['Random Forest'] = rf_model.predict_proba(passenger_df)[0]
        
        # Mostrar resultados
        if len(predictions) == 2:
            # Ambos modelos disponibles
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 🌳 Decision Tree")
                dt_pred = predictions['Decision Tree']
                dt_prob = probabilities['Decision Tree']
                
                if dt_pred == 1:
                    st.success(f"### ✅ SOBREVIVE\n**Probabilidad: {dt_prob[1]:.1%}**")
                else:
                    st.error(f"### ❌ NO SOBREVIVE\n**Probabilidad: {dt_prob[0]:.1%}**")
                
                # Gauge chart
                fig_dt = create_gauge_chart(dt_prob[1], "Decision Tree")
                st.plotly_chart(fig_dt, use_container_width=True)
            
            with col2:
                st.markdown("### 🌲 Random Forest")
                rf_pred = predictions['Random Forest']
                rf_prob = probabilities['Random Forest']
                
                if rf_pred == 1:
                    st.success(f"### ✅ SOBREVIVE\n**Probabilidad: {rf_prob[1]:.1%}**")
                else:
                    st.error(f"### ❌ NO SOBREVIVE\n**Probabilidad: {rf_prob[0]:.1%}**")
                
                # Gauge chart
                fig_rf = create_gauge_chart(rf_prob[1], "Random Forest")
                st.plotly_chart(fig_rf, use_container_width=True)
            
            # Consenso
            st.markdown("---")
            st.markdown("### 🤝 Consenso de Modelos")
            
            if predictions['Decision Tree'] == predictions['Random Forest']:
                avg_prob = (probabilities['Decision Tree'][1] + probabilities['Random Forest'][1]) / 2
                if predictions['Decision Tree'] == 1:
                    st.success(f"""
                    ✅ **Ambos modelos predicen: SOBREVIVE**
                    
                    Probabilidad promedio de supervivencia: **{avg_prob:.1%}**
                    
                    Los modelos están de acuerdo en que este pasajero tendría buenas
                    probabilidades de sobrevivir al desastre del Titanic.
                    """)
                else:
                    st.error(f"""
                    ❌ **Ambos modelos predicen: NO SOBREVIVE**
                    
                    Probabilidad promedio de muerte: **{1-avg_prob:.1%}**
                    
                    Los modelos están de acuerdo en que este pasajero tendría pocas
                    probabilidades de sobrevivir al desastre del Titanic.
                    """)
            else:
                st.warning("""
                ⚠️ **Los modelos NO están de acuerdo**
                
                Hay discrepancia entre las predicciones. En casos reales, esto sugiere
                que el pasajero está en una zona de incertidumbre y podríamos necesitar
                más información o análisis adicional.
                """)
        
        else:
            # Solo un modelo disponible
            model_name = list(predictions.keys())[0]
            pred = predictions[model_name]
            prob = probabilities[model_name]
            
            st.markdown(f"### {model_name}")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                if pred == 1:
                    st.success(f"## ✅ SOBREVIVE\n**Probabilidad: {prob[1]:.1%}**")
                else:
                    st.error(f"## ❌ NO SOBREVIVE\n**Probabilidad: {prob[0]:.1%}**")
            
            with col2:
                st.metric("Confianza", f"{max(prob):.1%}")
            
            # Gauge chart
            fig = create_gauge_chart(prob[1], model_name)
            st.plotly_chart(fig, use_container_width=True)
        
        # Análisis del perfil
        st.markdown("---")
        st.markdown("## 🔍 Análisis del Perfil")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.info(f"""
            **Clase Social**
            
            {'Primera Clase - Elite' if pclass == 1 else 'Segunda Clase - Medio' if pclass == 2 else 'Tercera Clase - Económica'}
            
            {'✅ Factor positivo' if pclass == 1 else '⚠️ Factor neutral' if pclass == 2 else '❌ Factor negativo'}
            """)
        
        with col2:
            st.info(f"""
            **Género**
            
            {'Femenino' if sex == 'female' else 'Masculino'}
            
            {'✅ Factor muy positivo' if sex == 'female' else '❌ Factor negativo'}
            
            {'Mujeres tuvieron prioridad' if sex == 'female' else 'Hombres tenían menor prioridad'}
            """)
        
        with col3:
            family_size = sibsp + parch
            st.info(f"""
            **Situación Familiar**
            
            Tamaño de familia: {family_size}
            
            {'⚠️ Viaja solo' if family_size == 0 else '✅ Con familia pequeña' if family_size <= 3 else '❌ Familia grande'}
            """)
        
        # Factores clave
        st.markdown("### 💡 Factores Clave en esta Predicción")
        
        factors = []
        
        # Analizar factores
        if sex == 'female':
            factors.append(("✅ Género femenino", "Las mujeres tuvieron ~74% de tasa de supervivencia"))
        else:
            factors.append(("❌ Género masculino", "Los hombres tuvieron ~19% de tasa de supervivencia"))
        
        if pclass == 1:
            factors.append(("✅ Primera clase", "63% de pasajeros de 1ra clase sobrevivieron"))
        elif pclass == 3:
            factors.append(("❌ Tercera clase", "Solo 24% de pasajeros de 3ra clase sobrevivieron"))
        
        if age < 16:
            factors.append(("✅ Niño/Adolescente", "Los niños tuvieron prioridad en la evacuación"))
        elif age > 60:
            factors.append(("⚠️ Edad avanzada", "Personas mayores tuvieron dificultades en la evacuación"))
        
        if fare > 50:
            factors.append(("✅ Tarifa alta", "Correlaciona con mejor ubicación y acceso a botes"))
        
        if family_size == 0:
            factors.append(("⚠️ Sin familia", "Pasajeros solos tuvieron tasas mixtas"))
        elif family_size > 3:
            factors.append(("❌ Familia grande", "Familias grandes tuvieron dificultad coordinándose"))
        
        for emoji_msg, explanation in factors:
            st.markdown(f"**{emoji_msg}**: {explanation}")
        
        # Datos del pasajero
        st.markdown("---")
        st.markdown("### 📋 Resumen del Pasajero")
        
        summary_df = pd.DataFrame({
            'Característica': ['Clase', 'Género', 'Edad', 'Hermanos/Cónyuges', 'Padres/Hijos', 
                              'Tarifa', 'Puerto'],
            'Valor': [pclass, sex, age, sibsp, parch, f"£{fare:.2f}", 
                     'Southampton' if embarked == 'S' else 'Cherbourg' if embarked == 'C' else 'Queenstown']
        })
        
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
    
    # Sección de ejemplos
    st.markdown("---")
    st.markdown("## 🎭 Prueba con Perfiles Históricos")
    
    st.markdown("""
    Experimenta con estos perfiles basados en pasajeros reales del Titanic:
    """)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("👩 Mujer de 1ra Clase", use_container_width=True):
            st.info("""
            **Perfil sugerido:**
            - Clase: 1
            - Género: Female
            - Edad: 35
            - Tarifa: 100
            - Puerto: C (Cherbourg)
            
            Alta probabilidad de supervivencia ✅
            """)
    
    with col2:
        if st.button("👨 Hombre de 3ra Clase", use_container_width=True):
            st.info("""
            **Perfil sugerido:**
            - Clase: 3
            - Género: Male
            - Edad: 25
            - Tarifa: 8
            - Puerto: S (Southampton)
            
            Baja probabilidad de supervivencia ❌
            """)
    
    with col3:
        if st.button("👶 Niño con Familia", use_container_width=True):
            st.info("""
            **Perfil sugerido:**
            - Clase: 2
            - Género: Male
            - Edad: 5
            - Padres: 2
            - Tarifa: 20
            - Puerto: S (Southampton)
            
            Probabilidad moderada-alta ⚠️
            """)
    
    # Nota final
    st.markdown("---")
    st.info("""
    **📝 Nota Importante:**
    
    Este predictor es un modelo de Machine Learning entrenado con datos históricos.
    Las predicciones son probabilísticas y se basan en patrones encontrados en los datos.
    En el evento real del Titanic, muchos factores adicionales (suerte, ubicación exacta,
    momento de la evacuación, etc.) influyeron en la supervivencia individual.
    """)


def create_gauge_chart(probability, model_name):
    """
    Crea un gráfico de gauge para mostrar probabilidad
    
    Args:
        probability: Probabilidad de supervivencia (0-1)
        model_name: Nombre del modelo
        
    Returns:
        plotly.Figure
    """
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=probability * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"Probabilidad de Supervivencia<br><span style='font-size:0.8em'>{model_name}</span>"},
        number={'suffix': "%"},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1},
            'bar': {'color': "#2ecc71" if probability > 0.5 else "#e74c3c"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 33], 'color': '#ffebee'},
                {'range': [33, 66], 'color': '#fff3e0'},
                {'range': [66, 100], 'color': '#e8f5e9'}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': 50
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=80, b=20)
    )
    
    return fig