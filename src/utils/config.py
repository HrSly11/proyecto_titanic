"""
Configuraciones globales del proyecto
"""

# Configuración de datos
DATA_URL = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Configuración de modelos
DECISION_TREE_PARAMS = {
    'max_depth': 5,
    'min_samples_split': 20,
    'min_samples_leaf': 10,
    'random_state': RANDOM_STATE
}

RANDOM_FOREST_PARAMS = {
    'n_estimators': 100,
    'max_depth': 10,
    'min_samples_split': 20,
    'min_samples_leaf': 5,
    'random_state': RANDOM_STATE,
    'n_jobs': -1
}

# Features a utilizar
FEATURES_NUMERICAS = ['Age', 'SibSp', 'Parch', 'Fare']
FEATURES_CATEGORICAS = ['Pclass', 'Sex', 'Embarked']
TARGET = 'Survived'

# Configuración de visualización
COLOR_SURVIVED = '#2ecc71'  # Verde
COLOR_NOT_SURVIVED = '#e74c3c'  # Rojo
COLOR_PRIMARY = '#3498db'  # Azul
COLOR_SECONDARY = '#9b59b6'  # Morado

# Rangos para el simulador
AGE_RANGE = (0, 80)
FARE_RANGE = (0, 500)
SIBSP_RANGE = (0, 8)
PARCH_RANGE = (0, 6)