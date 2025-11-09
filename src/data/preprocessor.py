"""
Módulo para preprocesamiento de datos
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from src.utils.config import RANDOM_STATE, TEST_SIZE


class TitanicPreprocessor:
    """Clase para preprocesar datos del Titanic"""
    
    def __init__(self):
        self.feature_names = None
        self.encodings = {}
    
    def clean_data(self, df):
        """
        Limpia y prepara los datos
        
        Args:
            df (pd.DataFrame): Dataset original
            
        Returns:
            pd.DataFrame: Dataset limpio
        """
        df_clean = df.copy()
        
        # Eliminar columnas innecesarias
        columns_to_drop = ['PassengerId', 'Name', 'Ticket', 'Cabin']
        df_clean = df_clean.drop(columns=columns_to_drop, errors='ignore')
        
        # Imputar valores nulos en Age con la mediana
        df_clean['Age'].fillna(df_clean['Age'].median(), inplace=True)
        
        # Imputar valores nulos en Embarked con la moda
        df_clean['Embarked'].fillna(df_clean['Embarked'].mode()[0], inplace=True)
        
        # Imputar valores nulos en Fare con la mediana
        df_clean['Fare'].fillna(df_clean['Fare'].median(), inplace=True)
        
        return df_clean
    
    def encode_features(self, df):
        """
        Codifica variables categóricas
        
        Args:
            df (pd.DataFrame): Dataset con variables categóricas
            
        Returns:
            pd.DataFrame: Dataset con variables codificadas
        """
        df_encoded = df.copy()
        
        # Codificar Sex (0: male, 1: female)
        if 'Sex' in df_encoded.columns:
            df_encoded['Sex'] = df_encoded['Sex'].map({'male': 0, 'female': 1})
            self.encodings['Sex'] = {'male': 0, 'female': 1}
        
        # Codificar Embarked con One-Hot Encoding
        if 'Embarked' in df_encoded.columns:
            embarked_dummies = pd.get_dummies(df_encoded['Embarked'], prefix='Embarked')
            df_encoded = pd.concat([df_encoded, embarked_dummies], axis=1)
            df_encoded.drop('Embarked', axis=1, inplace=True)
            self.encodings['Embarked'] = df_encoded.columns.tolist()
        
        return df_encoded
    
    def prepare_features(self, df):
        """
        Prepara features y target para el modelo
        
        Args:
            df (pd.DataFrame): Dataset preprocesado
            
        Returns:
            tuple: (X, y) Features y target
        """
        # Separar target
        if 'Survived' in df.columns:
            y = df['Survived']
            X = df.drop('Survived', axis=1)
        else:
            y = None
            X = df
        
        self.feature_names = X.columns.tolist()
        
        return X, y
    
    def split_data(self, X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE):
        """
        Divide datos en train y test
        
        Args:
            X: Features
            y: Target
            test_size: Proporción del test set
            random_state: Semilla aleatoria
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        return train_test_split(X, y, test_size=test_size, 
                                random_state=random_state, stratify=y)
    
    def full_pipeline(self, df):
        """
        Pipeline completo de preprocesamiento
        
        Args:
            df (pd.DataFrame): Dataset original
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test, df_clean)
        """
        # Limpiar datos
        df_clean = self.clean_data(df)
        
        # Codificar features
        df_encoded = self.encode_features(df_clean)
        
        # Preparar features y target
        X, y = self.prepare_features(df_encoded)
        
        # Dividir en train/test
        X_train, X_test, y_train, y_test = self.split_data(X, y)
        
        return X_train, X_test, y_train, y_test, df_clean