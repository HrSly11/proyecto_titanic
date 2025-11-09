"""
Componente de sidebar para navegación
"""
import streamlit as st


def render_sidebar():
    """
    Renderiza el sidebar de navegación
    
    Returns:
        str: Página seleccionada
    """
    with st.sidebar:
        st.title("🚢 Análisis Titanic")
        st.markdown("---")
        
        # Navegación
        st.subheader("Navegación")
        
        page = st.radio(
            "Selecciona una sección:",
            [
                "🏠 Inicio",
                "📊 Exploración de Datos",
                "🛠️ Preparación de Datos",
                "🌳 Árbol de Decisión",
                "🌲 Random Forest",
                "⚖️ Comparación de Modelos",
                "🔮 Predictor Interactivo"
            ],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        
        # Información del proyecto
        st.subheader("📋 Proyecto")
        st.info("""
        **Análisis de Supervivencia**  
        Dataset: Titanic  
        Modelos: Decision Tree & Random Forest
        """)
        
        st.markdown("---")
        
        # Integrantes
        st.subheader("👥 Desarrolladores")
        st.markdown("""
        - **Harry**  
          Árbol de Decisión
        - **Tania**  
          Random Forest
        """)
        
        st.markdown("---")
        
        # Footer
        st.caption("Machine Learning - 2025 - UNT")
        
    return page