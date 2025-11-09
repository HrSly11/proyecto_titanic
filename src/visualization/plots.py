"""
Módulo para visualizaciones generales
"""
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from src.utils.config import COLOR_SURVIVED, COLOR_NOT_SURVIVED


def plot_survival_distribution(df):
    """
    Gráfico de distribución de supervivencia
    
    Args:
        df: DataFrame con columna 'Survived'
        
    Returns:
        plotly.Figure
    """
    survival_counts = df['Survived'].value_counts()
    
    fig = go.Figure(data=[
        go.Bar(
            x=['Died', 'Survived'],
            y=[survival_counts[0], survival_counts[1]],
            marker_color=[COLOR_NOT_SURVIVED, COLOR_SURVIVED],
            text=[f'{survival_counts[0]} ({survival_counts[0]/len(df)*100:.1f}%)',
                  f'{survival_counts[1]} ({survival_counts[1]/len(df)*100:.1f}%)'],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title='Distribución de Supervivencia',
        xaxis_title='Estado',
        yaxis_title='Número de Pasajeros',
        template='plotly_white',
        height=400
    )
    
    return fig


def plot_survival_by_feature(df, feature, title=None):
    """
    Gráfico de supervivencia por característica
    
    Args:
        df: DataFrame
        feature: Característica a analizar
        title: Título personalizado
        
    Returns:
        plotly.Figure
    """
    survival_by_feature = df.groupby([feature, 'Survived']).size().reset_index(name='count')
    
    fig = px.bar(
        survival_by_feature,
        x=feature,
        y='count',
        color='Survived',
        barmode='group',
        color_discrete_map={0: COLOR_NOT_SURVIVED, 1: COLOR_SURVIVED},
        labels={'Survived': 'Estado', 'count': 'Cantidad'},
        title=title or f'Supervivencia por {feature}'
    )
    
    fig.update_layout(template='plotly_white', height=400)
    
    return fig


def plot_age_distribution(df):
    """
    Distribución de edades con supervivencia
    
    Args:
        df: DataFrame con columnas 'Age' y 'Survived'
        
    Returns:
        plotly.Figure
    """
    fig = go.Figure()
    
    # Histograma para sobrevivientes
    fig.add_trace(go.Histogram(
        x=df[df['Survived'] == 1]['Age'],
        name='Sobrevivieron',
        marker_color=COLOR_SURVIVED,
        opacity=0.7,
        nbinsx=30
    ))
    
    # Histograma para no sobrevivientes
    fig.add_trace(go.Histogram(
        x=df[df['Survived'] == 0]['Age'],
        name='Murieron',
        marker_color=COLOR_NOT_SURVIVED,
        opacity=0.7,
        nbinsx=30
    ))
    
    fig.update_layout(
        title='Distribución de Edad por Supervivencia',
        xaxis_title='Edad',
        yaxis_title='Frecuencia',
        barmode='overlay',
        template='plotly_white',
        height=400
    )
    
    return fig


def plot_fare_distribution(df):
    """
    Distribución de tarifas con supervivencia
    
    Args:
        df: DataFrame con columnas 'Fare' y 'Survived'
        
    Returns:
        plotly.Figure
    """
    fig = px.box(
        df,
        x='Survived',
        y='Fare',
        color='Survived',
        color_discrete_map={0: COLOR_NOT_SURVIVED, 1: COLOR_SURVIVED},
        labels={'Survived': 'Estado', 'Fare': 'Tarifa'},
        title='Distribución de Tarifa por Supervivencia'
    )
    
    fig.update_layout(template='plotly_white', height=400)
    fig.update_xaxes(tickvals=[0, 1], ticktext=['Died', 'Survived'])
    
    return fig


def plot_correlation_heatmap(df):
    """
    Mapa de calor de correlaciones
    
    Args:
        df: DataFrame con variables numéricas
        
    Returns:
        plotly.Figure
    """
    # Seleccionar solo columnas numéricas
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr_matrix = df[numeric_cols].corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmid=0,
        text=corr_matrix.values.round(2),
        texttemplate='%{text}',
        textfont={"size": 10},
        colorbar=dict(title="Correlación")
    ))
    
    fig.update_layout(
        title='Matriz de Correlación',
        template='plotly_white',
        height=500,
        width=700
    )
    
    return fig


def plot_train_test_split(y_train, y_test):
    """
    Visualiza la distribución de clases en train/test
    
    Args:
        y_train: Target de entrenamiento
        y_test: Target de prueba
        
    Returns:
        plotly.Figure
    """
    train_counts = pd.Series(y_train).value_counts()
    test_counts = pd.Series(y_test).value_counts()
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Train',
        x=['Died', 'Survived'],
        y=[train_counts[0], train_counts[1]],
        marker_color='lightblue'
    ))
    
    fig.add_trace(go.Bar(
        name='Test',
        x=['Died', 'Survived'],
        y=[test_counts[0], test_counts[1]],
        marker_color='lightcoral'
    ))
    
    fig.update_layout(
        title='Distribución de Clases en Train/Test',
        xaxis_title='Clase',
        yaxis_title='Cantidad',
        barmode='group',
        template='plotly_white',
        height=400
    )
    
    return fig