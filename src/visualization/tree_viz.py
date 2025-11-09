"""
Módulo para visualización de árboles de decisión
"""
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
import plotly.graph_objects as go


def plot_decision_tree_matplotlib(model, feature_names, class_names=['Died', 'Survived']):
    """
    Visualiza el árbol de decisión con matplotlib
    
    Args:
        model: Modelo de árbol entrenado
        feature_names: Nombres de las características
        class_names: Nombres de las clases
        
    Returns:
        matplotlib.Figure
    """
    fig, ax = plt.subplots(figsize=(20, 10))
    
    plot_tree(
        model,
        feature_names=feature_names,
        class_names=class_names,
        filled=True,
        rounded=True,
        fontsize=10,
        ax=ax
    )
    
    plt.title('Árbol de Decisión - Titanic', fontsize=16, pad=20)
    plt.tight_layout()
    
    return fig


def plot_tree_depth_performance(depths, train_scores, test_scores):
    """
    Gráfico de rendimiento vs profundidad del árbol
    
    Args:
        depths: Lista de profundidades
        train_scores: Scores de entrenamiento
        test_scores: Scores de prueba
        
    Returns:
        plotly.Figure
    """
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=depths,
        y=train_scores,
        mode='lines+markers',
        name='Train Score',
        line=dict(color='blue', width=2),
        marker=dict(size=8)
    ))
    
    fig.add_trace(go.Scatter(
        x=depths,
        y=test_scores,
        mode='lines+markers',
        name='Test Score',
        line=dict(color='red', width=2),
        marker=dict(size=8)
    ))
    
    fig.update_layout(
        title='Rendimiento del Modelo vs Profundidad del Árbol',
        xaxis_title='Profundidad Máxima',
        yaxis_title='Accuracy',
        template='plotly_white',
        height=400,
        hovermode='x unified'
    )
    
    return fig


def plot_feature_importance_tree(importance_dict):
    """
    Gráfico de importancia de características del árbol
    
    Args:
        importance_dict: Diccionario con importancias
        
    Returns:
        plotly.Figure
    """
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
        text=[f'{imp:.3f}' for imp in importances],
        textposition='auto',
    ))
    
    fig.update_layout(
        title='Importancia de Características - Árbol de Decisión',
        xaxis_title='Importancia',
        yaxis_title='Característica',
        template='plotly_white',
        height=max(400, len(features) * 30),
        showlegend=False
    )
    
    return fig