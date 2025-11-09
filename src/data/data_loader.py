"""
Módulo para cargar y validar datos del Titanic
"""
import pandas as pd
import streamlit as st
from src.utils.config import DATA_URL


@st.cache_data
def load_titanic_data():
    """
    Carga el dataset del Titanic desde URL o archivo local
    
    Returns:
        pd.DataFrame: Dataset del Titanic
    """
    try:
        df = pd.read_csv(DATA_URL)
        return df
    except Exception as e:
        st.error(f"Error al cargar datos: {e}")
        return None


def get_data_info(df):
    """
    Obtiene información descriptiva del dataset
    
    Args:
        df (pd.DataFrame): Dataset
        
    Returns:
        dict: Información del dataset
    """
    info = {
        'n_filas': len(df),
        'n_columnas': len(df.columns),
        'columnas': df.columns.tolist(),
        'tipos': df.dtypes.to_dict(),
        'nulos': df.isnull().sum().to_dict(),
        'duplicados': df.duplicated().sum()
    }
    return info


def get_survival_stats(df):
    """
    Calcula estadísticas de supervivencia
    
    Args:
        df (pd.DataFrame): Dataset
        
    Returns:
        dict: Estadísticas de supervivencia
    """
    total = len(df)
    survived = df['Survived'].sum()
    died = total - survived
    
    stats = {
        'total_pasajeros': total,
        'sobrevivieron': survived,
        'murieron': died,
        'tasa_supervivencia': (survived / total) * 100,
        'tasa_mortalidad': (died / total) * 100
    }
    return stats


def get_feature_stats(df, feature):
    """
    Obtiene estadísticas de una característica específica
    
    Args:
        df (pd.DataFrame): Dataset
        feature (str): Nombre de la característica
        
    Returns:
        dict: Estadísticas de la característica
    """
    if feature in df.columns:
        if df[feature].dtype in ['int64', 'float64']:
            stats = {
                'media': df[feature].mean(),
                'mediana': df[feature].median(),
                'std': df[feature].std(),
                'min': df[feature].min(),
                'max': df[feature].max(),
                'nulos': df[feature].isnull().sum()
            }
        else:
            stats = {
                'valores_unicos': df[feature].nunique(),
                'valores': df[feature].value_counts().to_dict(),
                'nulos': df[feature].isnull().sum()
            }
        return stats
    return None