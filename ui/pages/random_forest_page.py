"""
Página del modelo de Random Forest
Responsable: Tania
"""
import streamlit as st
import pandas as pd
import numpy as np
from src.data.data_loader import load_titanic_data
from src.data.preprocessor import TitanicPreprocessor
from src.models.random_forest import RandomForestModel
from src.visualization.metrics_viz import (
    plot_confusion_matrix,
    plot_metrics_comparison,
    plot_cross_validation_scores,
    plot_feature_importance_comparison
)
from ui.styles.theme import render_header
import plotly.graph_objects as go


def show():
    """Muestra la página de Random Forest"""
    
    render_header(
        "🌲 Random Forest",
        "Implementación y análisis del modelo Random Forest"
    )
    
    st.markdown("""
    ## 📚 Introducción a Random Forest
    
    **Random Forest** es un algoritmo de *ensemble learning* que combina múltiples árboles
    de decisión para crear un modelo más robusto y preciso. Cada árbol se entrena con una
    muestra aleatoria de los datos (bootstrap) y considera solo un subconjunto aleatorio
    de características en cada división.
    
    **Ventajas sobre un solo árbol:**
    - ✅ Reduce el sobreajuste mediante promedio de predicciones
    - ✅ Más estable ante cambios en los datos
    - ✅ Proporciona estimaciones de importancia de características más confiables
    """)
    
    # Cargar y preparar datos
    with st.spinner("⏳ Preparando datos..."):
        if 'X_train' not in st.session_state:
            df = load_titanic_data()
            if df is None:
                st.error("Error al cargar datos")
                return
            
            preprocessor = TitanicPreprocessor()
            X_train, X_test, y_train, y_test, df_clean = preprocessor.full_pipeline(df)
            
            st.session_state['X_train'] = X_train
            st.session_state['X_test'] = X_test
            st.session_state['y_train'] = y_train
            st.session_state['y_test'] = y_test
            st.session_state['feature_names'] = preprocessor.feature_names
        else:
            X_train = st.session_state['X_train']
            X_test = st.session_state['X_test']
            y_train = st.session_state['y_train']
            y_test = st.session_state['y_test']
    
    st.success("✅ Datos preparados exitosamente")
    
    # Tabs
    tabs = st.tabs([
        "⚙️ Configuración",
        "📊 Entrenamiento",
        "🔄 Validación Cruzada",
        "📈 Métricas",
        "🎯 Feature Importance",
        "⚖️ Comparación con DT"
    ])
    
    # Tab 1: Configuración
    with tabs[0]:
        st.subheader("Configuración de Hiperparámetros")
        
        st.markdown("""
        Random Forest tiene varios hiperparámetros que controlan el comportamiento del ensemble.
        Ajusta estos valores para optimizar el rendimiento del modelo.
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🌲 Parámetros del Ensemble")
            
            n_estimators = st.slider(
                "Número de Árboles (n_estimators)",
                min_value=10,
                max_value=500,
                value=100,
                step=10,
                help="Cantidad de árboles en el bosque. Más árboles = más estable pero más lento."
            )
            
            max_depth = st.slider(
                "Profundidad Máxima",
                min_value=2,
                max_value=30,
                value=10,
                help="Profundidad máxima de cada árbol individual."
            )
            
            max_features = st.selectbox(
                "Max Features por Split",
                options=['sqrt', 'log2', None],
                index=0,
                help="Número máximo de características a considerar en cada división."
            )
        
        with col2:
            st.markdown("#### 📊 Parámetros de Control")
            
            min_samples_split = st.slider(
                "Min Samples Split",
                min_value=2,
                max_value=100,
                value=20,
                help="Mínimo de muestras para dividir un nodo."
            )
            
            min_samples_leaf = st.slider(
                "Min Samples Leaf",
                min_value=1,
                max_value=50,
                value=5,
                help="Mínimo de muestras en cada hoja."
            )
            
            random_state = st.number_input(
                "Random State",
                min_value=0,
                max_value=999,
                value=42,
                help="Semilla para reproducibilidad."
            )
        
        # Información adicional
        with st.expander("ℹ️ Explicación de Hiperparámetros"):
            st.markdown("""
            **n_estimators**: Número de árboles en el bosque
            - Más árboles generalmente mejoran el rendimiento
            - Rendimientos decrecientes después de cierto punto
            - Aumenta el tiempo de entrenamiento linealmente
            
            **max_depth**: Profundidad máxima de cada árbol
            - Controla la complejidad de cada árbol individual
            - Valores muy altos pueden causar overfitting
            - Random Forest es menos sensible que un solo árbol
            
            **max_features**: Features a considerar en cada split
            - 'sqrt': Raíz cuadrada del número total de features (recomendado para clasificación)
            - 'log2': Logaritmo base 2 del número de features
            - None: Todas las features (no recomendado)
            
            **min_samples_split/leaf**: Control de tamaño de nodos
            - Previene divisiones en grupos muy pequeños
            - Ayuda a reducir overfitting
            """)
        
        # Guardar configuración
        st.session_state['rf_params'] = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf,
            'max_features': max_features,
            'random_state': random_state
        }
        
        st.success("✓ Configuración guardada")
    
    # Tab 2: Entrenamiento
    with tabs[1]:
        st.subheader("Entrenamiento del Modelo")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Muestras Train", len(X_train))
        with col2:
            st.metric("Muestras Test", len(X_test))
        with col3:
            st.metric("Features", len(st.session_state.get('feature_names', [])))
        
        st.markdown("---")
        
        if st.button("🚀 Entrenar Random Forest", type="primary", use_container_width=True):
            with st.spinner("Entrenando Random Forest... Esto puede tomar unos segundos."):
                # Crear y entrenar modelo
                rf_model = RandomForestModel(**st.session_state['rf_params'])
                rf_model.train(X_train, y_train)
                
                # Guardar en session_state
                st.session_state['rf_model'] = rf_model
                st.session_state['rf_trained'] = True
                
                st.success("✅ Random Forest entrenado exitosamente!")
                st.balloons()
                
                # Información del ensemble
                ensemble_info = rf_model.analyze_ensemble()
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Árboles en Bosque", ensemble_info['n_estimators'])
                with col2:
                    st.metric("Profundidad Máxima", ensemble_info['max_depth'])
                with col3:
                    st.metric("Features Usadas", ensemble_info['n_features'])
                with col4:
                    st.metric("Clases", ensemble_info['n_classes'])
        
        if st.session_state.get('rf_trained', False):
            st.info("✓ Modelo entrenado. Explora las demás pestañas para análisis detallado.")
    
    # Tab 3: Validación Cruzada
    with tabs[2]:
        st.subheader("Validación Cruzada")
        
        if not st.session_state.get('rf_trained', False):
            st.warning("⚠️ Primero entrena el modelo en la pestaña 'Entrenamiento'")
        else:
            st.markdown("""
            La **validación cruzada** divide los datos en K partes (folds) y entrena
            el modelo K veces, usando cada vez una parte diferente como test.
            Esto proporciona una estimación más robusta del rendimiento.
            """)
            
            cv_folds = st.slider(
                "Número de Folds",
                min_value=3,
                max_value=10,
                value=5,
                help="Número de divisiones para validación cruzada (K-Fold CV)"
            )
            
            if st.button("🔄 Ejecutar Validación Cruzada", type="primary"):
                with st.spinner(f"Ejecutando {cv_folds}-Fold Cross Validation..."):
                    rf_model = st.session_state['rf_model']
                    
                    # Realizar validación cruzada
                    cv_results = rf_model.cross_validate(
                        pd.concat([X_train, X_test]),
                        pd.concat([y_train, y_test]),
                        cv=cv_folds
                    )
                    
                    st.session_state['cv_results'] = cv_results
                    
                    st.success("✅ Validación cruzada completada!")
                    
                    # Métricas de CV
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Score Promedio", f"{cv_results['mean_score']:.4f}")
                    with col2:
                        st.metric("Desv. Estándar", f"{cv_results['std_score']:.4f}")
                    with col3:
                        st.metric("Score Mínimo", f"{cv_results['min_score']:.4f}")
                    with col4:
                        st.metric("Score Máximo", f"{cv_results['max_score']:.4f}")
                    
                    # Gráfico de CV
                    st.plotly_chart(
                        plot_cross_validation_scores(cv_results['scores']),
                        use_container_width=True
                    )
                    
                    st.info(f"""
                    **Interpretación:**
                    
                    - Una desviación estándar baja ({cv_results['std_score']:.4f}) indica que el modelo
                      es estable y consistente en diferentes subconjuntos de datos.
                    - El score promedio ({cv_results['mean_score']:.4f}) es una estimación más confiable
                      del rendimiento real que un solo train/test split.
                    """)
    
    # Tab 4: Métricas
    with tabs[3]:
        st.subheader("Métricas de Evaluación")
        
        if not st.session_state.get('rf_trained', False):
            st.warning("⚠️ Primero entrena el modelo en la pestaña 'Entrenamiento'")
        else:
            rf_model = st.session_state['rf_model']
            
            # Evaluar modelo
            metrics = rf_model.evaluate(X_test, y_test)
            st.session_state['rf_metrics'] = metrics
            
            # Métricas principales
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
            
            # Gráficos
            col1, col2 = st.columns(2)
            
            with col1:
                st.plotly_chart(
                    plot_metrics_comparison(metrics, "Random Forest"),
                    use_container_width=True
                )
            
            with col2:
                st.plotly_chart(
                    plot_confusion_matrix(metrics['confusion_matrix'], title="Matriz de Confusión - RF"),
                    use_container_width=True
                )
            
            # Reporte detallado
            st.subheader("Reporte de Clasificación")
            st.text(metrics['classification_report'])
            
            # Análisis Train vs Test
            st.markdown("---")
            st.subheader("Análisis de Generalization")
            
            train_score = rf_model.model.score(X_train, y_train)
            test_score = metrics['accuracy']
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Train Accuracy", f"{train_score:.3f}")
            with col2:
                st.metric("Test Accuracy", f"{test_score:.3f}")
            with col3:
                diff = train_score - test_score
                st.metric("Diferencia", f"{diff:.3f}")
            
            if diff < 0.03:
                st.success("🟢 Excelente generalización! El modelo no está sobreajustado.")
            elif diff < 0.07:
                st.info("🟡 Buena generalización. Diferencia aceptable entre train y test.")
            else:
                st.warning("🟠 Posible sobreajuste. Considera ajustar hiperparámetros.")
    
    # Tab 5: Feature Importance
    with tabs[4]:
        st.subheader("Importancia de Características")
        
        if not st.session_state.get('rf_trained', False):
            st.warning("⚠️ Primero entrena el modelo en la pestaña 'Entrenamiento'")
        else:
            rf_model = st.session_state['rf_model']
            
            st.markdown("""
            Random Forest calcula la importancia de cada característica basándose en
            cuánto reducen la impureza (Gini) en promedio a través de todos los árboles.
            """)
            
            # Obtener importancias
            importance_dict = rf_model.get_feature_importance()
            
            # Gráfico de importancia
            features = list(importance_dict.keys())
            importances = list(importance_dict.values())
            
            fig = go.Figure(go.Bar(
                x=importances,
                y=features,
                orientation='h',
                marker=dict(
                    color=importances,
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Importancia")
                ),
                text=[f'{imp:.4f}' for imp in importances],
                textposition='auto'
            ))
            
            fig.update_layout(
                title='Importancia de Características - Random Forest',
                xaxis_title='Importancia',
                yaxis_title='Característica',
                template='plotly_white',
                height=max(400, len(features) * 30)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Tabla de importancias
            with st.expander("📊 Ver tabla de importancias"):
                importance_df = pd.DataFrame({
                    'Característica': features,
                    'Importancia': importances,
                    'Porcentaje': [f"{imp*100:.2f}%" for imp in importances]
                })
                st.dataframe(importance_df, use_container_width=True)
            
            # Top 3 features
            st.markdown("### 🏆 Top 3 Características Más Importantes")
            top3 = list(importance_dict.items())[:3]
            
            cols = st.columns(3)
            for i, (feature, importance) in enumerate(top3):
                with cols[i]:
                    st.metric(
                        f"#{i+1} {feature}",
                        f"{importance:.4f}",
                        f"{importance*100:.2f}%"
                    )
    
    # Tab 6: Comparación con DT
    with tabs[5]:
        st.subheader("Comparación: Random Forest vs Decision Tree")
        
        if not st.session_state.get('rf_trained', False):
            st.warning("⚠️ Primero entrena el modelo Random Forest")
        elif not st.session_state.get('dt_trained', False):
            st.warning("⚠️ También necesitas entrenar el Decision Tree en su sección")
        else:
            st.markdown("""
            Comparación directa entre el modelo individual (Decision Tree) y el ensemble (Random Forest).
            """)
            
            rf_metrics = st.session_state['rf_metrics']
            dt_metrics = st.session_state['dt_metrics']
            
            # Tabla comparativa
            st.markdown("### 📊 Tabla Comparativa de Métricas")
            
            comparison_df = pd.DataFrame({
                'Métrica': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
                'Decision Tree': [
                    dt_metrics['accuracy'],
                    dt_metrics['precision'],
                    dt_metrics['recall'],
                    dt_metrics['f1_score']
                ],
                'Random Forest': [
                    rf_metrics['accuracy'],
                    rf_metrics['precision'],
                    rf_metrics['recall'],
                    rf_metrics['f1_score']
                ]
            })
            
            comparison_df['Mejora (%)'] = (
                (comparison_df['Random Forest'] - comparison_df['Decision Tree']) / 
                comparison_df['Decision Tree'] * 100
            ).round(2)
            
            st.dataframe(
                comparison_df.style.format({
                    'Decision Tree': '{:.4f}',
                    'Random Forest': '{:.4f}',
                    'Mejora (%)': '{:+.2f}%'
                }).background_gradient(subset=['Mejora (%)'], cmap='RdYlGn'),
                use_container_width=True
            )
            
            # Gráfico comparativo
            from src.visualization.metrics_viz import plot_models_comparison
            st.plotly_chart(
                plot_models_comparison(dt_metrics, rf_metrics),
                use_container_width=True
            )
            
            # Comparación de Feature Importance
            st.markdown("---")
            st.markdown("### 🎯 Comparación de Feature Importance")
            
            dt_importance = st.session_state['dt_model'].get_feature_importance()
            rf_importance = rf_model.get_feature_importance()
            
            st.plotly_chart(
                plot_feature_importance_comparison(dt_importance, rf_importance),
                use_container_width=True
            )
            
            # Conclusiones
            st.markdown("---")
            st.markdown("### 📝 Conclusiones")
            
            winner = "Random Forest" if rf_metrics['accuracy'] > dt_metrics['accuracy'] else "Decision Tree"
            
            st.success(f"""
            **🏆 Modelo Ganador: {winner}**
            
            **Resumen de comparación:**
            - Random Forest {'supera' if rf_metrics['accuracy'] > dt_metrics['accuracy'] else 'es similar a'} Decision Tree en accuracy
            - El ensemble reduce el overfitting y mejora la estabilidad
            - Random Forest proporciona estimaciones más confiables de feature importance
            - El costo es mayor tiempo de entrenamiento y menor interpretabilidad
            """)
            
            st.info("""
            **💡 Recomendaciones:**
            
            - **Usa Decision Tree** si necesitas interpretabilidad máxima y un modelo simple
            - **Usa Random Forest** si priorizas rendimiento y robustez sobre interpretabilidad
            - Para producción, Random Forest suele ser la mejor opción
            """)