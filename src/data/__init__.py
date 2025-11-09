"""
Módulo de manejo de datos
"""
from .data_loader import load_titanic_data, get_data_info, get_survival_stats
from .preprocessor import TitanicPreprocessor
from .feature_engineering import TitanicFeatureEngineer, engineer_features_simple

__all__ = [
    'load_titanic_data',
    'get_data_info',
    'get_survival_stats',
    'TitanicPreprocessor',
    'TitanicFeatureEngineer',
    'engineer_features_simple'
]