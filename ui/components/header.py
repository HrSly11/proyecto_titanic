"""
Componente de header personalizado
"""
import streamlit as st


def render_page_header(icon, title, subtitle=None, description=None):
    """
    Renderiza un header de página con estilo
    
    Args:
        icon: Emoji o icono
        title: Título principal
        subtitle: Subtítulo opcional
        description: Descripción adicional opcional
    """
    st.markdown(f"""
    <div style='
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    '>
        <h1 style='margin: 0; color: white; font-size: 2.5rem; font-weight: 700;'>
            {icon} {title}
        </h1>
        {f"<p style='margin: 0.5rem 0 0 0; font-size: 1.1rem; opacity: 0.9;'>{subtitle}</p>" if subtitle else ""}
        {f"<p style='margin: 1rem 0 0 0; font-size: 0.95rem; opacity: 0.85;'>{description}</p>" if description else ""}
    </div>
    """, unsafe_allow_html=True)


def render_section_header(icon, title, description=None):
    """
    Renderiza un header de sección
    
    Args:
        icon: Emoji o icono
        title: Título de la sección
        description: Descripción opcional
    """
    st.markdown(f"""
    <div style='
        padding: 1rem;
        margin: 2rem 0 1rem 0;
        border-left: 4px solid #667eea;
        background: linear-gradient(90deg, #f8f9fa 0%, transparent 100%);
    '>
        <h2 style='margin: 0; color: #2c3e50; font-size: 1.8rem;'>
            {icon} {title}
        </h2>
        {f"<p style='margin: 0.5rem 0 0 0; color: #7f8c8d;'>{description}</p>" if description else ""}
    </div>
    """, unsafe_allow_html=True)


def render_info_box(title, content, box_type="info"):
    """
    Renderiza una caja de información estilizada
    
    Args:
        title: Título de la caja
        content: Contenido de la caja
        box_type: Tipo (info, success, warning, error)
    """
    colors = {
        "info": {"bg": "#d1ecf1", "border": "#3498db", "icon": "ℹ️"},
        "success": {"bg": "#d4edda", "border": "#2ecc71", "icon": "✅"},
        "warning": {"bg": "#fff3cd", "border": "#f39c12", "icon": "⚠️"},
        "error": {"bg": "#f8d7da", "border": "#e74c3c", "icon": "❌"}
    }
    
    color = colors.get(box_type, colors["info"])
    
    st.markdown(f"""
    <div style='
        background-color: {color["bg"]};
        border-left: 4px solid {color["border"]};
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    '>
        <h4 style='margin: 0 0 0.5rem 0; color: #2c3e50;'>
            {color["icon"]} {title}
        </h4>
        <p style='margin: 0; color: #2c3e50;'>
            {content}
        </p>
    </div>
    """, unsafe_allow_html=True)