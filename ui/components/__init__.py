"""
Componentes reutilizables de la interfaz de usuario
"""
from .sidebar import render_sidebar
from .header import render_page_header, render_section_header, render_info_box
from .footer import render_footer, render_mini_footer

__all__ = [
    'render_sidebar',
    'render_page_header',
    'render_section_header',
    'render_info_box',
    'render_footer',
    'render_mini_footer'
]