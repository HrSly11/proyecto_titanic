"""
Aplicación Principal - Análisis del Titanic con ML
Autores: Harry Style (Decision Tree) y Tania (Random Forest)
"""
import streamlit as st

# Importar componentes de UI
from ui.components.sidebar import render_sidebar
from ui.styles.theme import apply_theme

# Importar páginas
from ui.pages import (
    home,
    exploracion,
    preparacion,
    decision_tree_page,
    random_forest_page,
    comparacion,
    predictor
)


# Configuración de la página
st.set_page_config(
    page_title="Titanic ML Analysis",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded"
)


def main():
    """Función principal de la aplicación"""
    
    # Aplicar tema personalizado
    apply_theme()
    
    # Inicializar session_state
    if 'dt_trained' not in st.session_state:
        st.session_state['dt_trained'] = False
    if 'rf_trained' not in st.session_state:
        st.session_state['rf_trained'] = False
    
    # Renderizar sidebar y obtener página seleccionada
    page = render_sidebar()
    
    # Routing de páginas
    if page == "🏠 Inicio":
        home.show()
    
    elif page == "📊 Exploración de Datos":
        exploracion.show()
    
    elif page == "🛠️ Preparación de Datos":
        preparacion.show()
    
    elif page == "🌳 Árbol de Decisión":
        decision_tree_page.show()
    
    elif page == "🌲 Random Forest":
        random_forest_page.show()
    
    elif page == "⚖️ Comparación de Modelos":
        comparacion.show()
    
    elif page == "🔮 Predictor Interactivo":
        predictor.show()


if __name__ == "__main__":
    main()