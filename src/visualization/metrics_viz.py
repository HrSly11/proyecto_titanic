"""
Módulo para visualización de métricas de modelos
"""
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np


def plot_confusion_matrix(cm, labels=['Died', 'Survived'], title='Matriz de Confusión'):
    """
    Visualiza la matriz de confusión
    
    Args:
        cm: Matriz de confusión
        labels: Etiquetas de las clases
        title: Título del gráfico
        
    Returns:
        plotly.Figure
    """
    # Normalizar para obtener porcentajes
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Crear anotaciones
    annotations = []
    for i in range(len(labels)):
        for j in range(len(labels)):
            annotations.append(
                dict(
                    x=j,
                    y=i,
                    text=f'{cm[i, j]}<br>({cm_normalized[i, j]:.1%})',
                    showarrow=False,
                    font=dict(color='white' if cm_normalized[i, j] > 0.5 else 'black', size=12)
                )
            )
    
    fig = go.Figure(data=go.Heatmap(
        z=cm_normalized,
        x=labels,
        y=labels,
        colorscale='Blues',
        showscale=True,
        colorbar=dict(title="Proporción")
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Predicción',
        yaxis_title='Real',
        annotations=annotations,
        template='plotly_white',
        height=400,
        width=500
    )
    
    return fig


def plot_metrics_comparison(metrics_dict, model_name='Modelo'):
    """
    Gráfico de barras con métricas principales
    
    Args:
        metrics_dict: Diccionario con métricas
        model_name: Nombre del modelo
        
    Returns:
        plotly.Figure
    """
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    values = [metrics_dict.get(m, 0) for m in metrics]
    
    colors = ['#3498db', '#2ecc71', '#f39c12', '#9b59b6']
    
    fig = go.Figure(data=[
        go.Bar(
            x=['Accuracy', 'Precision', 'Recall', 'F1-Score'],
            y=values,
            marker_color=colors,
            text=[f'{v:.3f}' for v in values],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title=f'Métricas de Evaluación - {model_name}',
        yaxis_title='Score',
        yaxis=dict(range=[0, 1]),
        template='plotly_white',
        height=400
    )
    
    return fig


def plot_models_comparison(dt_metrics, rf_metrics):
    """
    Comparación lado a lado de dos modelos
    
    Args:
        dt_metrics: Métricas del árbol de decisión
        rf_metrics: Métricas del random forest
        
    Returns:
        plotly.Figure
    """
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    
    dt_values = [dt_metrics.get(m, 0) for m in metrics]
    rf_values = [rf_metrics.get(m, 0) for m in metrics]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Decision Tree',
        x=metric_names,
        y=dt_values,
        marker_color='lightblue',
        text=[f'{v:.3f}' for v in dt_values],
        textposition='auto',
    ))
    
    fig.add_trace(go.Bar(
        name='Random Forest',
        x=metric_names,
        y=rf_values,
        marker_color='lightgreen',
        text=[f'{v:.3f}' for v in rf_values],
        textposition='auto',
    ))
    
    fig.update_layout(
        title='Comparación de Modelos: Decision Tree vs Random Forest',
        yaxis_title='Score',
        yaxis=dict(range=[0, 1]),
        barmode='group',
        template='plotly_white',
        height=450
    )
    
    return fig


def plot_cross_validation_scores(cv_scores):
    """
    Visualiza los scores de validación cruzada
    
    Args:
        cv_scores: Array con scores de CV
        
    Returns:
        plotly.Figure
    """
    folds = list(range(1, len(cv_scores) + 1))
    
    fig = go.Figure()
    
    # Barras con los scores
    fig.add_trace(go.Bar(
        x=folds,
        y=cv_scores,
        marker_color='lightcoral',
        text=[f'{s:.3f}' for s in cv_scores],
        textposition='auto',
        name='Fold Score'
    ))
    
    # Línea con la media
    mean_score = cv_scores.mean()
    fig.add_trace(go.Scatter(
        x=[0.5, len(folds) + 0.5],
        y=[mean_score, mean_score],
        mode='lines',
        line=dict(color='red', width=2, dash='dash'),
        name=f'Media: {mean_score:.3f}'
    ))
    
    fig.update_layout(
        title='Resultados de Validación Cruzada',
        xaxis_title='Fold',
        yaxis_title='Accuracy',
        yaxis=dict(range=[0, 1]),
        template='plotly_white',
        height=400
    )
    
    return fig


def plot_feature_importance_comparison(dt_importance, rf_importance):
    """
    Compara la importancia de características entre modelos
    
    Args:
        dt_importance: Importancias del Decision Tree
        rf_importance: Importancias del Random Forest
        
    Returns:
        plotly.Figure
    """
    # Obtener features comunes
    features = list(dt_importance.keys())
    
    dt_values = [dt_importance[f] for f in features]
    rf_values = [rf_importance.get(f, 0) for f in features]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Decision Tree',
        x=features,
        y=dt_values,
        marker_color='lightblue'
    ))
    
    fig.add_trace(go.Bar(
        name='Random Forest',
        x=features,
        y=rf_values,
        marker_color='lightgreen'
    ))
    
    fig.update_layout(
        title='Comparación de Importancia de Características',
        xaxis_title='Característica',
        yaxis_title='Importancia',
        barmode='group',
        template='plotly_white',
        height=450
    )
    
    return fig