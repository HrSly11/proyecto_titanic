"""
Página del modelo de Árbol de Decisión
Responsable: Harry Style
"""
import streamlit as st
import pandas as pd
from src.data.data_loader import load_titanic_data
from src.data.preprocessor import TitanicPreprocessor
from src.models.decision_tree import DecisionTreeModel
from src.visualization.tree_viz import (
    plot_decision_tree_matplotlib,
    plot_feature_importance_tree
)
from src.visualization.metrics_viz import plot_confusion_matrix, plot_metrics_comparison
from ui.styles.theme import render_header


def show():
    """Muestra la página de Árbol de Decisión"""
    
    render_header(
        "🌳 Árbol de Decisión",
        "Implementación y análisis del modelo Decision Tree"
    )
    
    st.markdown("""
    ## 📚 Introducción al Árbol de Decisión
    
    Un **Árbol de Decisión** es un algoritmo de aprendizaje supervisado que construye un modelo
    de predicción en forma de estructura de árbol. Divide el dataset en subconjuntos más pequeños
    basándose en el valor de las características de entrada.
    """)
    
    # Cargar y preparar datos
    with st.spinner("⏳ Cargando y preparando datos..."):
        df = load_titanic_data()
        if df is None:
            st.error("Error al cargar datos")
            return
        
        preprocessor = TitanicPreprocessor()
        X_train, X_test, y_train, y_test, df_clean = preprocessor.full_pipeline(df)
        
        # Guardar en session_state
        st.session_state['X_train'] = X_train
        st.session_state['X_test'] = X_test
        st.session_state['y_train'] = y_train
        st.session_state['y_test'] = y_test
        st.session_state['feature_names'] = preprocessor.feature_names
    
    st.success("✅ Datos preparados exitosamente")
    
    # Tabs para organizar contenido
    tabs = st.tabs([
        "⚙️ Configuración",
        "📊 Entrenamiento",
        "🌳 Visualización del Árbol",
        "📈 Métricas",
        "🔍 Análisis de Overfitting"
    ])
    
    # Tab 1: Configuración
    with tabs[0]:
        st.subheader("Configuración del Modelo")
        
        st.markdown("""
        Ajusta los hiperparámetros del árbol de decisión. Estos parámetros controlan
        la complejidad del modelo y pueden ayudar a prevenir el sobreajuste.
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            max_depth = st.slider(
                "Profundidad Máxima del Árbol",
                min_value=2,
                max_value=20,
                value=5,
                help="Controla qué tan profundo puede crecer el árbol. Mayor profundidad = mayor complejidad."
            )
            
            min_samples_split = st.slider(
                "Mínimo de Muestras para Dividir",
                min_value=2,
                max_value=100,
                value=20,
                help="Número mínimo de muestras requeridas para dividir un nodo interno."
            )
        
        with col2:
            min_samples_leaf = st.slider(
                "Mínimo de Muestras en Hoja",
                min_value=1,
                max_value=50,
                value=10,
                help="Número mínimo de muestras requeridas en un nodo hoja."
            )
            
            random_state = st.number_input(
                "Random State (Semilla)",
                min_value=0,
                max_value=999,
                value=42,
                help="Semilla para reproducibilidad de resultados."
            )
        
        # Información sobre parámetros
        with st.expander("ℹ️ ¿Qué significan estos parámetros?"):
            st.markdown("""
            **max_depth**: Limita la profundidad del árbol. Árboles más profundos pueden capturar
            patrones más complejos pero son más propensos al sobreajuste.
            
            **min_samples_split**: Si un nodo tiene menos muestras que este valor, no se divide.
            Ayuda a prevenir divisiones en grupos muy pequeños.
            
            **min_samples_leaf**: Garantiza que cada hoja tenga al menos este número de muestras.
            Previene hojas con muy pocas observaciones.
            
            **random_state**: Fija la semilla aleatoria para que los resultados sean reproducibles.
            """)
        
        st.session_state['dt_params'] = {
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf,
            'random_state': random_state
        }
    
    # Tab 2: Entrenamiento
    with tabs[1]:
        st.subheader("Entrenamiento del Modelo")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Muestras de Entrenamiento", len(X_train))
        with col2:
            st.metric("Muestras de Prueba", len(X_test))
        with col3:
            st.metric("Características", len(preprocessor.feature_names))
        
        st.markdown("---")
        
        if st.button("🚀 Entrenar Modelo", type="primary", use_container_width=True):
            with st.spinner("Entrenando árbol de decisión..."):
                # Crear y entrenar modelo
                dt_model = DecisionTreeModel(**st.session_state['dt_params'])
                dt_model.train(X_train, y_train)
                
                # Guardar modelo en session_state
                st.session_state['dt_model'] = dt_model
                st.session_state['dt_trained'] = True
                
                st.success("✅ Modelo entrenado exitosamente!")
                
                # Mostrar información del árbol
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Profundidad Real del Árbol", dt_model.get_tree_depth())
                with col2:
                    st.metric("Número de Hojas", dt_model.get_n_leaves())
                
                st.balloons()
        
        if st.session_state.get('dt_trained', False):
            st.info("✓ Modelo ya entrenado. Puedes explorar las demás pestañas.")
    
    # Tab 3: Visualización del Árbol
    with tabs[2]:
        st.subheader("Visualización del Árbol de Decisión")
        
        if not st.session_state.get('dt_trained', False):
            st.warning("⚠️ Primero debes entrenar el modelo en la pestaña 'Entrenamiento'")
        else:
            dt_model = st.session_state['dt_model']
            
            st.markdown("""
            Esta visualización muestra cómo el árbol toma decisiones. Cada nodo representa
            una pregunta sobre una característica, y las ramas representan las posibles respuestas.
            """)
            
            with st.spinner("Generando visualización del árbol..."):
                try:
                    fig = plot_decision_tree_matplotlib(
                        dt_model.model,
                        feature_names=st.session_state['feature_names']
                    )
                    st.pyplot(fig)
                    
                    st.success("💡 Colores: Naranja = Mayor probabilidad de morir, Azul = Mayor probabilidad de sobrevivir")
                except Exception as e:
                    st.error(f"Error al visualizar árbol: {e}")
            
            st.markdown("---")
            
            # Importancia de características
            st.subheader("Importancia de Características")
            
            importance_dict = dt_model.get_feature_importance()
            
            st.plotly_chart(
                plot_feature_importance_tree(importance_dict),
                use_container_width=True
            )
            
            with st.expander("📊 Ver valores de importancia"):
                importance_df = pd.DataFrame({
                    'Característica': importance_dict.keys(),
                    'Importancia': importance_dict.values()
                })
                st.dataframe(importance_df, use_container_width=True)
    
    # Tab 4: Métricas
    with tabs[3]:
        st.subheader("Métricas de Evaluación")
        
        if not st.session_state.get('dt_trained', False):
            st.warning("⚠️ Primero debes entrenar el modelo en la pestaña 'Entrenamiento'")
        else:
            dt_model = st.session_state['dt_model']
            
            # Evaluar modelo
            metrics = dt_model.evaluate(X_test, y_test)
            
            # Guardar métricas
            st.session_state['dt_metrics'] = metrics
            
            # Mostrar métricas principales
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Accuracy", f"{metrics['accuracy']:.3f}")
            with col2:
                st.metric("Precision", f"{metrics['precision']:.3f}")
            with col3:
                st.metric("Recall", f"{metrics['recall']:.3f}")
            with col4:
                st.metric("F1-Score", f"{metrics['f1_score']:.3f}")
            
            st.markdown("---")
            
            # Gráficos de métricas
            col1, col2 = st.columns(2)
            
            with col1:
                st.plotly_chart(
                    plot_metrics_comparison(metrics, "Decision Tree"),
                    use_container_width=True
                )
            
            with col2:
                st.plotly_chart(
                    plot_confusion_matrix(metrics['confusion_matrix']),
                    use_container_width=True
                )
            
            # Reporte de clasificación
            st.subheader("Reporte de Clasificación Completo")
            st.text(metrics['classification_report'])
            
            # Explicación de métricas
            with st.expander("📖 ¿Qué significan estas métricas?"):
                st.markdown("""
                **Accuracy**: Proporción de predicciones correctas sobre el total.
                
                **Precision**: De todos los que predijimos como sobrevivientes, ¿cuántos realmente sobrevivieron?
                
                **Recall (Sensibilidad)**: De todos los que realmente sobrevivieron, ¿cuántos identificamos correctamente?
                
                **F1-Score**: Media armónica entre precision y recall, útil cuando las clases están desbalanceadas.
                
                **Matriz de Confusión**:
                - Verdaderos Negativos (TN): Predijimos muerte y murieron
                - Falsos Positivos (FP): Predijimos supervivencia pero murieron
                - Falsos Negativos (FN): Predijimos muerte pero sobrevivieron
                - Verdaderos Positivos (TP): Predijimos supervivencia y sobrevivieron
                """)
    
    # Tab 5: Análisis de Overfitting
    with tabs[4]:
        st.subheader("Análisis de Sobreajuste (Overfitting)")
        
        if not st.session_state.get('dt_trained', False):
            st.warning("⚠️ Primero debes entrenar el modelo en la pestaña 'Entrenamiento'")
        else:
            dt_model = st.session_state['dt_model']
            
            st.markdown("""
            El **overfitting** ocurre cuando un modelo aprende demasiado bien los datos de entrenamiento,
            incluyendo el ruido, lo que resulta en un mal rendimiento en datos nuevos.
            """)
            
            # Análisis de overfitting
            overfitting_analysis = dt_model.analyze_overfitting(X_train, y_train, X_test, y_test)
            
            # Métricas de comparación
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Accuracy en Train",
                    f"{overfitting_analysis['train_accuracy']:.3f}"
                )
            
            with col2:
                st.metric(
                    "Accuracy en Test",
                    f"{overfitting_analysis['test_accuracy']:.3f}"
                )
            
            with col3:
                delta_color = "inverse" if overfitting_analysis['difference'] > 0.05 else "normal"
                st.metric(
                    "Diferencia",
                    f"{overfitting_analysis['difference']:.3f}",
                    delta=overfitting_analysis['overfitting_level'],
                    delta_color=delta_color
                )
            
            # Interpretación
            if overfitting_analysis['overfitting_level'] == 'Alto':
                st.error(f"🔴 {overfitting_analysis['description']}")
            elif overfitting_analysis['overfitting_level'] == 'Moderado':
                st.warning(f"🟡 {overfitting_analysis['description']}")
            else:
                st.success(f"🟢 {overfitting_analysis['description']}")
            
            st.markdown("---")
            
            # Recomendaciones
            st.subheader("💡 Recomendaciones para Reducir Overfitting")
            
            st.markdown("""
            Si tu modelo tiene overfitting alto, prueba:
            
            1. **Reducir max_depth**: Limita la profundidad del árbol
            2. **Aumentar min_samples_split**: Requiere más muestras para dividir nodos
            3. **Aumentar min_samples_leaf**: Asegura hojas con más muestras
            4. **Usar Random Forest**: El ensemble reduce la varianza
            5. **Podar el árbol**: Eliminar ramas que no mejoran significativamente la predicción
            """)
            
            # Comparación visual
            st.info("""
            **📊 Interpretación de la Diferencia:**
            
            - **< 0.05**: Modelo bien balanceado ✅
            - **0.05 - 0.10**: Overfitting moderado ⚠️
            - **> 0.10**: Overfitting alto, ajustar parámetros ❌
            """)
    
    # Sección final
    st.markdown("---")
    st.markdown("""
    ## 📝 Conclusiones del Árbol de Decisión
    
    El modelo de Árbol de Decisión nos permite:
    - ✅ Entender claramente cómo se toman las decisiones
    - ✅ Identificar las características más importantes
    - ✅ Visualizar el proceso de clasificación
    - ⚠️ Monitorear el sobreajuste ajustando hiperparámetros
    
    **Siguiente paso**: Compara este modelo con Random Forest para ver cómo el ensemble
    mejora el rendimiento y reduce el overfitting.
    """)