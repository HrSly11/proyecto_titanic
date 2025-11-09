"""
Funciones auxiliares y utilidades generales
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import streamlit as st


def format_metric(value: float, metric_type: str = "percentage") -> str:
    """
    Formatea un valor métrico para visualización
    
    Args:
        value: Valor a formatear
        metric_type: Tipo de métrica (percentage, decimal, integer)
        
    Returns:
        str: Valor formateado
    """
    if metric_type == "percentage":
        return f"{value * 100:.2f}%"
    elif metric_type == "decimal":
        return f"{value:.4f}"
    elif metric_type == "integer":
        return f"{int(value)}"
    else:
        return str(value)


def calculate_improvement(baseline: float, new_value: float) -> Dict[str, Any]:
    """
    Calcula la mejora entre dos valores
    
    Args:
        baseline: Valor base
        new_value: Nuevo valor
        
    Returns:
        dict: Diccionario con mejora absoluta y porcentual
    """
    absolute_improvement = new_value - baseline
    if baseline != 0:
        percentage_improvement = (absolute_improvement / baseline) * 100
    else:
        percentage_improvement = 0
    
    return {
        'absolute': absolute_improvement,
        'percentage': percentage_improvement,
        'improved': absolute_improvement > 0
    }


def check_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analiza valores faltantes en un DataFrame
    
    Args:
        df: DataFrame a analizar
        
    Returns:
        pd.DataFrame: Resumen de valores faltantes
    """
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    
    missing_df = pd.DataFrame({
        'Column': missing.index,
        'Missing_Count': missing.values,
        'Percentage': missing_pct.values
    })
    
    return missing_df[missing_df['Missing_Count'] > 0].sort_values('Missing_Count', ascending=False)


def get_feature_types(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Clasifica features por tipo
    
    Args:
        df: DataFrame a analizar
        
    Returns:
        dict: Diccionario con listas de features por tipo
    """
    numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    return {
        'numeric': numeric_features,
        'categorical': categorical_features
    }


def validate_data_split(y_train: pd.Series, y_test: pd.Series) -> Dict[str, Any]:
    """
    Valida que la división de datos sea adecuada
    
    Args:
        y_train: Target de entrenamiento
        y_test: Target de prueba
        
    Returns:
        dict: Información de validación
    """
    train_counts = y_train.value_counts(normalize=True)
    test_counts = y_test.value_counts(normalize=True)
    
    # Calcular diferencia en proporciones
    max_diff = abs(train_counts - test_counts).max()
    
    return {
        'train_distribution': train_counts.to_dict(),
        'test_distribution': test_counts.to_dict(),
        'max_difference': max_diff,
        'balanced': max_diff < 0.05,
        'train_size': len(y_train),
        'test_size': len(y_test)
    }


def create_confusion_matrix_labels(cm: np.ndarray, labels: List[str]) -> List[str]:
    """
    Crea etiquetas formateadas para matriz de confusión
    
    Args:
        cm: Matriz de confusión
        labels: Etiquetas de clases
        
    Returns:
        list: Etiquetas formateadas
    """
    total = cm.sum()
    formatted_labels = []
    
    for i in range(len(labels)):
        for j in range(len(labels)):
            count = cm[i, j]
            percentage = (count / total) * 100
            formatted_labels.append(f'{count}\n({percentage:.1f}%)')
    
    return formatted_labels


def get_model_summary(model, X_train, y_train, X_test, y_test) -> Dict[str, Any]:
    """
    Obtiene un resumen completo del modelo
    
    Args:
        model: Modelo entrenado
        X_train, y_train: Datos de entrenamiento
        X_test, y_test: Datos de prueba
        
    Returns:
        dict: Resumen del modelo
    """
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    
    return {
        'train_score': train_score,
        'test_score': test_score,
        'overfitting': train_score - test_score,
        'n_train': len(X_train),
        'n_test': len(X_test),
        'n_features': X_train.shape[1]
    }


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    División segura que maneja división por cero
    
    Args:
        numerator: Numerador
        denominator: Denominador
        default: Valor por defecto si denominador es 0
        
    Returns:
        float: Resultado de la división o valor por defecto
    """
    try:
        if denominator == 0:
            return default
        return numerator / denominator
    except:
        return default


def get_categorical_encoding_info(original_df: pd.DataFrame, encoded_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Obtiene información sobre la codificación de variables categóricas
    
    Args:
        original_df: DataFrame original
        encoded_df: DataFrame codificado
        
    Returns:
        dict: Información de codificación
    """
    original_cols = set(original_df.columns)
    encoded_cols = set(encoded_df.columns)
    
    new_cols = encoded_cols - original_cols
    removed_cols = original_cols - encoded_cols
    
    return {
        'new_columns': list(new_cols),
        'removed_columns': list(removed_cols),
        'original_shape': original_df.shape,
        'encoded_shape': encoded_df.shape,
        'features_added': len(new_cols),
        'features_removed': len(removed_cols)
    }


@st.cache_data
def load_and_cache_data(data_loader_func):
    """
    Decorator para cachear la carga de datos
    
    Args:
        data_loader_func: Función que carga datos
        
    Returns:
        Datos cacheados
    """
    return data_loader_func()


def generate_feature_summary(df: pd.DataFrame, feature: str) -> Dict[str, Any]:
    """
    Genera un resumen completo de una feature
    
    Args:
        df: DataFrame
        feature: Nombre de la feature
        
    Returns:
        dict: Resumen de la feature
    """
    summary = {
        'name': feature,
        'dtype': str(df[feature].dtype),
        'missing': df[feature].isnull().sum(),
        'missing_pct': (df[feature].isnull().sum() / len(df)) * 100,
        'unique_values': df[feature].nunique()
    }
    
    if df[feature].dtype in ['int64', 'float64']:
        summary.update({
            'mean': df[feature].mean(),
            'median': df[feature].median(),
            'std': df[feature].std(),
            'min': df[feature].min(),
            'max': df[feature].max(),
            'q25': df[feature].quantile(0.25),
            'q75': df[feature].quantile(0.75)
        })
    else:
        summary.update({
            'mode': df[feature].mode()[0] if not df[feature].mode().empty else None,
            'top_values': df[feature].value_counts().head(5).to_dict()
        })
    
    return summary


def compare_models_summary(model1_metrics: Dict, model2_metrics: Dict, 
                          model1_name: str = "Model 1", 
                          model2_name: str = "Model 2") -> pd.DataFrame:
    """
    Crea una tabla comparativa de dos modelos
    
    Args:
        model1_metrics: Métricas del primer modelo
        model2_metrics: Métricas del segundo modelo
        model1_name: Nombre del primer modelo
        model2_name: Nombre del segundo modelo
        
    Returns:
        pd.DataFrame: Tabla comparativa
    """
    comparison_data = {
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
        model1_name: [
            model1_metrics['accuracy'],
            model1_metrics['precision'],
            model1_metrics['recall'],
            model1_metrics['f1_score']
        ],
        model2_name: [
            model2_metrics['accuracy'],
            model2_metrics['precision'],
            model2_metrics['recall'],
            model2_metrics['f1_score']
        ]
    }
    
    df = pd.DataFrame(comparison_data)
    df['Difference'] = df[model2_name] - df[model1_name]
    df['Winner'] = df.apply(
        lambda row: model2_name if row['Difference'] > 0 else model1_name if row['Difference'] < 0 else 'Tie',
        axis=1
    )
    
    return df