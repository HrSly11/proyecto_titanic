"""
Página de comparación de modelos
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from src.visualization.metrics_viz import plot_models_comparison, plot_feature_importance_comparison
from ui.styles.theme import render_header


def show():
    """Muestra la página de comparación"""
    
    render_header(
        "⚖️ Comparación de Modelos",
        "Análisis comparativo entre Decision Tree y Random Forest"
    )
    
    # Verificar que ambos modelos estén entrenados
    if not st.session_state.get('dt_trained', False):
        st.error("❌ Primero entrena el modelo Decision Tree en su sección")
        return
    
    if not st.session_state.get('rf_trained', False):
        st.error("❌ Primero entrena el modelo Random Forest en su sección")
        return
    
    st.success("✅ Ambos modelos están entrenados y listos para comparar")
    
    # Obtener métricas de ambos modelos
    dt_metrics = st.session_state['dt_metrics']
    rf_metrics = st.session_state['rf_metrics']
    
    # Tabs para organizar contenido
    tabs = st.tabs([
        "📊 Resumen Ejecutivo",
        "📈 Métricas Detalladas",
        "🎯 Feature Importance",
        "⏱️ Rendimiento",
        "💡 Conclusiones"
    ])
    
    # Tab 1: Resumen Ejecutivo
    with tabs[0]:
        st.subheader("Resumen Ejecutivo")
        
        # KPIs principales
        st.markdown("### 🎯 Métricas Principales")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🌳 Decision Tree")
            dt_col1, dt_col2 = st.columns(2)
            with dt_col1:
                st.metric("Accuracy", f"{dt_metrics['accuracy']:.3f}")
                st.metric("Precision", f"{dt_metrics['precision']:.3f}")
            with dt_col2:
                st.metric("Recall", f"{dt_metrics['recall']:.3f}")
                st.metric("F1-Score", f"{dt_metrics['f1_score']:.3f}")
        
        with col2:
            st.markdown("#### 🌲 Random Forest")
            rf_col1, rf_col2 = st.columns(2)
            with rf_col1:
                st.metric(
                    "Accuracy", 
                    f"{rf_metrics['accuracy']:.3f}",
                    delta=f"{(rf_metrics['accuracy'] - dt_metrics['accuracy']):.3f}"
                )
                st.metric(
                    "Precision", 
                    f"{rf_metrics['precision']:.3f}",
                    delta=f"{(rf_metrics['precision'] - dt_metrics['precision']):.3f}"
                )
            with rf_col2:
                st.metric(
                    "Recall", 
                    f"{rf_metrics['recall']:.3f}",
                    delta=f"{(rf_metrics['recall'] - dt_metrics['recall']):.3f}"
                )
                st.metric(
                    "F1-Score", 
                    f"{rf_metrics['f1_score']:.3f}",
                    delta=f"{(rf_metrics['f1_score'] - dt_metrics['f1_score']):.3f}"
                )
        
        st.markdown("---")
        
        # Ganador
        st.markdown("### 🏆 Modelo Ganador")
        
        winner_accuracy = "Random Forest" if rf_metrics['accuracy'] > dt_metrics['accuracy'] else "Decision Tree"
        winner_f1 = "Random Forest" if rf_metrics['f1_score'] > dt_metrics['f1_score'] else "Decision Tree"
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if rf_metrics['accuracy'] > dt_metrics['accuracy']:
                st.success(f"🏆 **{winner_accuracy}**\nMejor Accuracy")
            elif dt_metrics['accuracy'] > rf_metrics['accuracy']:
                st.info(f"🏆 **{winner_accuracy}**\nMejor Accuracy")
            else:
                st.warning("🤝 **Empate**\nMisma Accuracy")
        
        with col2:
            if rf_metrics['f1_score'] > dt_metrics['f1_score']:
                st.success(f"🏆 **{winner_f1}**\nMejor F1-Score")
            elif dt_metrics['f1_score'] > rf_metrics['f1_score']:
                st.info(f"🏆 **{winner_f1}**\nMejor F1-Score")
            else:
                st.warning("🤝 **Empate**\nMismo F1-Score")
        
        with col3:
            # Modelo más balanceado
            dt_balance = abs(dt_metrics['precision'] - dt_metrics['recall'])
            rf_balance = abs(rf_metrics['precision'] - rf_metrics['recall'])
            
            if rf_balance < dt_balance:
                st.success("🏆 **Random Forest**\nMás Balanceado")
            else:
                st.info("🏆 **Decision Tree**\nMás Balanceado")
        
        st.markdown("---")
        
        # Quick insights
        st.markdown("### 💡 Insights Rápidos")
        
        mejora_accuracy = ((rf_metrics['accuracy'] - dt_metrics['accuracy']) / dt_metrics['accuracy']) * 100
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.info(f"""
            **📊 Mejora en Accuracy:**
            
            Random Forest {'mejora' if mejora_accuracy > 0 else 'reduce'} el accuracy en **{abs(mejora_accuracy):.2f}%** 
            comparado con Decision Tree.
            """)
        
        with col2:
            if 'cv_results' in st.session_state:
                cv_std = st.session_state['cv_results']['std_score']
                st.info(f"""
                **🔄 Estabilidad (CV):**
                
                Random Forest tiene una desviación estándar de **{cv_std:.4f}** en validación cruzada,
                indicando {'alta' if cv_std < 0.02 else 'moderada'} estabilidad.
                """)
            else:
                st.info("""
                **🔄 Estabilidad:**
                
                Random Forest generalmente es más estable gracias al ensemble de múltiples árboles.
                """)
    
    # Tab 2: Métricas Detalladas
    with tabs[1]:
        st.subheader("Análisis Detallado de Métricas")
        
        # Gráfico comparativo
        st.plotly_chart(
            plot_models_comparison(dt_metrics, rf_metrics),
            use_container_width=True
        )
        
        st.markdown("---")
        
        # Tabla comparativa detallada
        st.markdown("### 📋 Tabla Comparativa Completa")
        
        comparison_data = {
            'Métrica': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 
                       'True Negatives', 'False Positives', 'False Negatives', 'True Positives'],
            'Decision Tree': [
                dt_metrics['accuracy'],
                dt_metrics['precision'],
                dt_metrics['recall'],
                dt_metrics['f1_score'],
                dt_metrics['confusion_matrix'][0][0],
                dt_metrics['confusion_matrix'][0][1],
                dt_metrics['confusion_matrix'][1][0],
                dt_metrics['confusion_matrix'][1][1]
            ],
            'Random Forest': [
                rf_metrics['accuracy'],
                rf_metrics['precision'],
                rf_metrics['recall'],
                rf_metrics['f1_score'],
                rf_metrics['confusion_matrix'][0][0],
                rf_metrics['confusion_matrix'][0][1],
                rf_metrics['confusion_matrix'][1][0],
                rf_metrics['confusion_matrix'][1][1]
            ]
        }
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Calcular diferencias
        comparison_df['Diferencia'] = comparison_df['Random Forest'] - comparison_df['Decision Tree']
        comparison_df['Mejora (%)'] = (
            (comparison_df['Random Forest'] - comparison_df['Decision Tree']) / 
            comparison_df['Decision Tree'] * 100
        ).round(2)
        
        # Aplicar formato
        styled_df = comparison_df.style.format({
            'Decision Tree': lambda x: f'{x:.4f}' if x < 2 else f'{int(x)}',
            'Random Forest': lambda x: f'{x:.4f}' if x < 2 else f'{int(x)}',
            'Diferencia': lambda x: f'{x:+.4f}' if abs(x) < 2 else f'{int(x):+d}',
            'Mejora (%)': '{:+.2f}%'
        })
        
        st.dataframe(styled_df, use_container_width=True)
        
        st.markdown("---")
        
        # Análisis de Confusion Matrix
        st.markdown("### 🔍 Análisis de Matrices de Confusión")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Decision Tree")
            from src.visualization.metrics_viz import plot_confusion_matrix
            st.plotly_chart(
                plot_confusion_matrix(dt_metrics['confusion_matrix'], title="DT - Confusion Matrix"),
                use_container_width=True
            )
        
        with col2:
            st.markdown("#### Random Forest")
            st.plotly_chart(
                plot_confusion_matrix(rf_metrics['confusion_matrix'], title="RF - Confusion Matrix"),
                use_container_width=True
            )
        
        # Interpretación
        st.info("""
        **📖 Interpretación de la Confusion Matrix:**
        
        - **True Negatives (TN)**: Muertes correctamente predichas
        - **False Positives (FP)**: Predijimos supervivencia pero murieron (Error Tipo I)
        - **False Negatives (FN)**: Predijimos muerte pero sobrevivieron (Error Tipo II)
        - **True Positives (TP)**: Supervivencias correctamente predichas
        
        Un buen modelo minimiza FP y FN mientras maximiza TN y TP.
        """)
    
    # Tab 3: Feature Importance
    with tabs[2]:
        st.subheader("Comparación de Feature Importance")
        
        st.markdown("""
        La importancia de características muestra qué variables tienen más influencia
        en las predicciones de cada modelo.
        """)
        
        dt_importance = st.session_state['dt_model'].get_feature_importance()
        rf_importance = st.session_state['rf_model'].get_feature_importance()
        
        # Gráfico comparativo
        st.plotly_chart(
            plot_feature_importance_comparison(dt_importance, rf_importance),
            use_container_width=True
        )
        
        st.markdown("---")
        
        # Tabla de importancias
        st.markdown("### 📊 Tabla de Importancias")
        
        importance_comparison = pd.DataFrame({
            'Feature': list(dt_importance.keys()),
            'DT Importance': list(dt_importance.values()),
            'RF Importance': [rf_importance.get(f, 0) for f in dt_importance.keys()]
        })
        
        importance_comparison['Diferencia'] = (
            importance_comparison['RF Importance'] - importance_comparison['DT Importance']
        )
        
        importance_comparison = importance_comparison.sort_values('RF Importance', ascending=False)
        
        st.dataframe(
            importance_comparison.style.format({
                'DT Importance': '{:.4f}',
                'RF Importance': '{:.4f}',
                'Diferencia': '{:+.4f}'
            }).background_gradient(subset=['RF Importance'], cmap='Greens'),
            use_container_width=True
        )
        
        # Top features de cada modelo
        st.markdown("---")
        st.markdown("### 🏆 Top 3 Features por Modelo")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🌳 Decision Tree")
            for i, (feature, imp) in enumerate(list(dt_importance.items())[:3], 1):
                st.metric(f"{i}. {feature}", f"{imp:.4f}")
        
        with col2:
            st.markdown("#### 🌲 Random Forest")
            for i, (feature, imp) in enumerate(list(rf_importance.items())[:3], 1):
                st.metric(f"{i}. {feature}", f"{imp:.4f}")
        
        # Análisis de consenso
        st.markdown("---")
        st.markdown("### 🤝 Consenso entre Modelos")
        
        dt_top3 = set(list(dt_importance.keys())[:3])
        rf_top3 = set(list(rf_importance.keys())[:3])
        
        consensus = dt_top3.intersection(rf_top3)
        
        if len(consensus) > 0:
            st.success(f"""
            **✅ Ambos modelos coinciden en que estas features son importantes:**
            
            {', '.join(consensus)}
            
            Esto sugiere que estas características son robustamente importantes
            para la predicción de supervivencia.
            """)
        else:
            st.warning("""
            ⚠️ Los modelos tienen diferencias en las features más importantes.
            Esto puede deberse a la forma diferente en que cada algoritmo evalúa la importancia.
            """)
    
    # Tab 4: Rendimiento
    with tabs[3]:
        st.subheader("Análisis de Rendimiento y Complejidad")
        
        st.markdown("""
        Comparación de características técnicas y de rendimiento de ambos modelos.
        """)
        
        # Tabla de características
        characteristics = {
            'Característica': [
                'Complejidad de Entrenamiento',
                'Complejidad de Predicción',
                'Interpretabilidad',
                'Resistencia al Overfitting',
                'Manejo de Ruido',
                'Estabilidad',
                'Uso de Memoria',
                'Paralelización'
            ],
            'Decision Tree': [
                'Baja (O(n log n))',
                'Muy Rápida (O(log n))',
                'Alta - Fácil de visualizar',
                'Baja - Propenso',
                'Baja',
                'Baja - Sensible a cambios',
                'Bajo',
                'No'
            ],
            'Random Forest': [
                'Media-Alta (k × O(n log n))',
                'Media (k × O(log n))',
                'Media-Baja',
                'Alta - Resistente',
                'Alta',
                'Alta - Robusto',
                'Alto (k árboles)',
                'Sí'
            ]
        }
        
        char_df = pd.DataFrame(characteristics)
        st.dataframe(char_df, use_container_width=True, hide_index=True)
        
        st.markdown("---")
        
        # Análisis de complejidad
        st.markdown("### ⚙️ Detalles de Configuración")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🌳 Decision Tree")
            dt_params = st.session_state.get('dt_params', {})
            st.code(f"""
Profundidad Máxima: {dt_params.get('max_depth', 'N/A')}
Min Samples Split: {dt_params.get('min_samples_split', 'N/A')}
Min Samples Leaf: {dt_params.get('min_samples_leaf', 'N/A')}

Árbol Real:
- Profundidad: {st.session_state.get('dt_model').get_tree_depth()}
- Hojas: {st.session_state.get('dt_model').get_n_leaves()}
            """)
        
        with col2:
            st.markdown("#### 🌲 Random Forest")
            rf_params = st.session_state.get('rf_params', {})
            ensemble_info = st.session_state.get('rf_model').analyze_ensemble()
            st.code(f"""
Número de Árboles: {rf_params.get('n_estimators', 'N/A')}
Profundidad Máxima: {rf_params.get('max_depth', 'N/A')}
Min Samples Split: {rf_params.get('min_samples_split', 'N/A')}
Min Samples Leaf: {rf_params.get('min_samples_leaf', 'N/A')}

Ensemble:
- Features: {ensemble_info['n_features']}
- Clases: {ensemble_info['n_classes']}
            """)
        
        st.markdown("---")
        
        # Casos de uso recomendados
        st.markdown("### 💼 Casos de Uso Recomendados")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.info("""
            **🌳 Usa Decision Tree cuando:**
            
            - ✅ Necesitas máxima interpretabilidad
            - ✅ Quieres explicar cada decisión
            - ✅ Tienes recursos limitados
            - ✅ El dataset es pequeño
            - ✅ Necesitas predicciones muy rápidas
            - ✅ La simplicidad es prioritaria
            """)
        
        with col2:
            st.success("""
            **🌲 Usa Random Forest cuando:**
            
            - ✅ Priorizas precisión sobre interpretabilidad
            - ✅ Tienes suficientes recursos computacionales
            - ✅ El dataset es mediano/grande
            - ✅ Necesitas robustez ante ruido
            - ✅ Quieres reducir overfitting
            - ✅ La producción requiere estabilidad
            """)
    
    # Tab 5: Conclusiones
    with tabs[4]:
        st.subheader("Conclusiones y Recomendaciones")
        
        # Resumen final
        st.markdown("### 📝 Resumen Final")
        
        winner = "Random Forest" if rf_metrics['accuracy'] > dt_metrics['accuracy'] else "Decision Tree"
        
        st.success(f"""
        ### 🏆 Modelo Recomendado: **{winner}**
        
        Basándonos en el análisis completo de métricas, estabilidad y características,
        **{winner}** es el modelo más adecuado para este problema de clasificación.
        """)
        
        st.markdown("---")
        
        # Análisis detallado
        st.markdown("### 🔍 Análisis Detallado")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### ✅ Fortalezas Identificadas")
            
            if rf_metrics['accuracy'] > dt_metrics['accuracy']:
                st.write("**Random Forest:**")
                st.write("- Mayor accuracy general")
                st.write("- Mejor generalización")
                st.write("- Más robusto ante overfitting")
            else:
                st.write("**Decision Tree:**")
                st.write("- Buena precisión con simplicidad")
                st.write("- Fácil de interpretar")
                st.write("- Rápido en entrenamiento y predicción")
            
            st.write("\n**Decision Tree:**")
            st.write("- Máxima interpretabilidad")
            st.write("- Visualización clara del proceso")
            st.write("- Bajo uso de recursos")
        
        with col2:
            st.markdown("#### ⚠️ Áreas de Mejora")
            
            st.write("**Decision Tree:**")
            st.write("- Propenso al sobreajuste")
            st.write("- Menos estable ante cambios en datos")
            st.write("- Puede crear árboles muy complejos")
            
            st.write("\n**Random Forest:**")
            st.write("- Menos interpretable que DT")
            st.write("- Mayor costo computacional")
            st.write("- Requiere más recursos de memoria")
        
        st.markdown("---")
        
        # Recomendaciones finales
        st.markdown("### 💡 Recomendaciones para Mejoras Futuras")
        
        st.info("""
        **📈 Próximos Pasos para Mejorar los Modelos:**
        
        1. **Optimización de Hiperparámetros:**
           - Usar GridSearchCV o RandomizedSearchCV
           - Explorar más combinaciones de parámetros
           - Optimizar específicamente para F1-score si las clases están desbalanceadas
        
        2. **Feature Engineering Avanzado:**
           - Crear features de familia (FamilySize, IsAlone)
           - Extraer títulos de los nombres (Mr., Mrs., Master)
           - Binning inteligente de Age y Fare
        
        3. **Ensemble Avanzado:**
           - Probar Gradient Boosting (XGBoost, LightGBM)
           - Implementar Stacking de modelos
           - Voting Classifier combinando ambos modelos
        
        4. **Validación Más Robusta:**
           - Usar StratifiedKFold para mejor validación
           - Implementar validación en datos temporales si aplica
           - Análisis de curvas ROC y AUC
        
        5. **Interpretabilidad:**
           - Usar SHAP values para explicar predicciones
           - Implementar LIME para casos individuales
           - Crear visualizaciones interactivas
        """)
        
        st.markdown("---")
        
        # Aplicación en el mundo real
        st.markdown("### 🌍 Aplicación en el Mundo Real")
        
        st.warning("""
        **⚡ Consideraciones para Producción:**
        
        **Si este fuera un sistema real de predicción:**
        
        1. **Manejo de Datos Nuevos:**
           - Pipeline de preprocesamiento automatizado
           - Validación de datos de entrada
           - Manejo de valores fuera de rango
        
        2. **Monitoreo y Mantenimiento:**
           - Tracking de performance en producción
           - Detección de data drift
           - Re-entrenamiento periódico
        
        3. **Explicabilidad:**
           - Reportes de decisiones para stakeholders
           - Auditoría de predicciones
           - Cumplimiento de regulaciones (GDPR, etc.)
        
        4. **Optimización:**
           - Reducción de tamaño del modelo
           - Optimización de velocidad de inferencia
           - Caching de predicciones frecuentes
        """)
        
        # Métricas finales
        st.markdown("---")
        st.markdown("### 🎯 Métricas Finales del Proyecto")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            best_acc = max(dt_metrics['accuracy'], rf_metrics['accuracy'])
            st.metric("Mejor Accuracy", f"{best_acc:.3f}")
        
        with col2:
            best_f1 = max(dt_metrics['f1_score'], rf_metrics['f1_score'])
            st.metric("Mejor F1-Score", f"{best_f1:.3f}")
        
        with col3:
            improvement = abs(rf_metrics['accuracy'] - dt_metrics['accuracy'])
            st.metric("Mejora Lograda", f"{improvement:.3f}")
        
        with col4:
            avg_acc = (dt_metrics['accuracy'] + rf_metrics['accuracy']) / 2
            st.metric("Accuracy Promedio", f"{avg_acc:.3f}")
        
        st.success("""
        ✅ **Proyecto Completado Exitosamente!**
        
        Hemos implementado, evaluado y comparado dos algoritmos fundamentales de Machine Learning,
        demostrando sus fortalezas y debilidades en un problema real de clasificación.
        """)