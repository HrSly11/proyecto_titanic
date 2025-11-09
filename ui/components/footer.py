"""
Componente de footer
"""
import streamlit as st
from datetime import datetime


def render_footer():
    """Renderiza el footer de la aplicación"""
    
    current_year = datetime.now().year
    
    st.markdown("---")
    st.markdown(f"""
    <div style='
        text-align: center;
        color: #7f8c8d;
        padding: 2rem 0;
        font-size: 0.9rem;
    '>
        <p style='margin: 0;'>
            🚢 <strong>Titanic ML Analysis Project</strong>
        </p>
        <p style='margin: 0.5rem 0;'>
            Desarrollado con ❤️ usando Streamlit y scikit-learn
        </p>
        <p style='margin: 0.5rem 0;'>
            👥 <strong>Integrantes:</strong> Harry Style (Decision Tree) | Tania (Random Forest)
        </p>
        <p style='margin: 0.5rem 0;'>
            📅 Machine Learning - {current_year}
        </p>
        <p style='margin: 1rem 0 0 0; font-size: 0.8rem; opacity: 0.7;'>
            Dataset: <a href='https://www.kaggle.com/c/titanic' target='_blank' style='color: #667eea;'>Kaggle Titanic Competition</a>
        </p>
    </div>
    """, unsafe_allow_html=True)


def render_mini_footer(text=None):
    """
    Renderiza un mini footer simple
    
    Args:
        text: Texto personalizado opcional
    """
    default_text = "🚢 Titanic ML Analysis | Machine Learning Project"
    display_text = text if text else default_text
    
    st.markdown(f"""
    <div style='
        text-align: center;
        color: #95a5a6;
        padding: 1rem 0;
        font-size: 0.85rem;
        border-top: 1px solid #ecf0f1;
        margin-top: 2rem;
    '>
        <p style='margin: 0;'>{display_text}</p>
    </div>
    """, unsafe_allow_html=True)