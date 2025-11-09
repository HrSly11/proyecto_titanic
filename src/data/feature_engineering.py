"""
Módulo para Feature Engineering avanzado
"""
import pandas as pd
import numpy as np
import re


class TitanicFeatureEngineer:
    """Clase para crear features avanzadas del Titanic"""
    
    def __init__(self):
        self.title_mapping = {
            'Mr': 0, 'Miss': 1, 'Mrs': 2, 'Master': 3,
            'Dr': 4, 'Rev': 4, 'Col': 4, 'Major': 4, 'Mlle': 1,
            'Countess': 2, 'Ms': 1, 'Lady': 2, 'Jonkheer': 4,
            'Don': 4, 'Dona': 2, 'Mme': 2, 'Capt': 4, 'Sir': 4
        }
    
    def create_family_size(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Crea feature de tamaño de familia
        
        Args:
            df: DataFrame con columnas SibSp y Parch
            
        Returns:
            pd.DataFrame: DataFrame con nueva columna FamilySize
        """
        df_new = df.copy()
        df_new['FamilySize'] = df_new['SibSp'] + df_new['Parch'] + 1
        return df_new
    
    def create_is_alone(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Crea feature indicando si viaja solo
        
        Args:
            df: DataFrame con columna FamilySize
            
        Returns:
            pd.DataFrame: DataFrame con nueva columna IsAlone
        """
        df_new = df.copy()
        if 'FamilySize' not in df_new.columns:
            df_new = self.create_family_size(df_new)
        
        df_new['IsAlone'] = (df_new['FamilySize'] == 1).astype(int)
        return df_new
    
    def extract_title(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extrae título del nombre
        
        Args:
            df: DataFrame con columna Name
            
        Returns:
            pd.DataFrame: DataFrame con nueva columna Title
        """
        if 'Name' not in df.columns:
            return df
        
        df_new = df.copy()
        
        # Extraer título
        df_new['Title'] = df_new['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
        
        # Mapear títulos raros a categorías comunes
        df_new['Title'] = df_new['Title'].map(self.title_mapping)
        df_new['Title'].fillna(0, inplace=True)
        
        return df_new
    
    def create_age_groups(self, df: pd.DataFrame, bins: list = None, labels: list = None) -> pd.DataFrame:
        """
        Crea grupos de edad
        
        Args:
            df: DataFrame con columna Age
            bins: Límites de los bins
            labels: Etiquetas de los grupos
            
        Returns:
            pd.DataFrame: DataFrame con nueva columna AgeGroup
        """
        df_new = df.copy()
        
        if bins is None:
            bins = [0, 12, 18, 35, 60, 100]
        if labels is None:
            labels = ['Child', 'Teen', 'Adult', 'Middle', 'Senior']
        
        df_new['AgeGroup'] = pd.cut(df_new['Age'], bins=bins, labels=labels)
        
        # Convertir a numérico
        age_group_mapping = {label: i for i, label in enumerate(labels)}
        df_new['AgeGroup'] = df_new['AgeGroup'].map(age_group_mapping)
        
        return df_new
    
    def create_fare_bins(self, df: pd.DataFrame, n_bins: int = 4) -> pd.DataFrame:
        """
        Crea bins de tarifa
        
        Args:
            df: DataFrame con columna Fare
            n_bins: Número de bins
            
        Returns:
            pd.DataFrame: DataFrame con nueva columna FareBin
        """
        df_new = df.copy()
        
        df_new['FareBin'] = pd.qcut(
            df_new['Fare'], 
            q=n_bins, 
            labels=range(n_bins),
            duplicates='drop'
        )
        
        return df_new
    
    def create_deck(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extrae el deck del número de cabina
        
        Args:
            df: DataFrame con columna Cabin
            
        Returns:
            pd.DataFrame: DataFrame con nueva columna Deck
        """
        if 'Cabin' not in df.columns:
            return df
        
        df_new = df.copy()
        df_new['Deck'] = df_new['Cabin'].str[0]
        
        # Mapear decks a números
        deck_mapping = {
            'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'T': 7
        }
        df_new['Deck'] = df_new['Deck'].map(deck_mapping)
        df_new['Deck'].fillna(-1, inplace=True)  # -1 para missing
        
        return df_new
    
    def create_ticket_frequency(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Crea feature de frecuencia de ticket (personas con mismo ticket)
        
        Args:
            df: DataFrame con columna Ticket
            
        Returns:
            pd.DataFrame: DataFrame con nueva columna TicketFrequency
        """
        if 'Ticket' not in df.columns:
            return df
        
        df_new = df.copy()
        ticket_counts = df_new['Ticket'].value_counts()
        df_new['TicketFrequency'] = df_new['Ticket'].map(ticket_counts)
        
        return df_new
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Crea features de interacción
        
        Args:
            df: DataFrame
            
        Returns:
            pd.DataFrame: DataFrame con nuevas columnas de interacción
        """
        df_new = df.copy()
        
        # Interacción Age * Class
        if 'Age' in df_new.columns and 'Pclass' in df_new.columns:
            df_new['Age_Class'] = df_new['Age'] * df_new['Pclass']
        
        # Interacción Fare * Class
        if 'Fare' in df_new.columns and 'Pclass' in df_new.columns:
            df_new['Fare_Class'] = df_new['Fare'] / (df_new['Pclass'] + 1)
        
        # Interacción Sex * Class
        if 'Sex' in df_new.columns and 'Pclass' in df_new.columns:
            df_new['Sex_Class'] = df_new['Sex'] * df_new['Pclass']
        
        return df_new
    
    def apply_all_features(self, df: pd.DataFrame, 
                          include_title: bool = False,
                          include_deck: bool = False) -> pd.DataFrame:
        """
        Aplica todas las transformaciones de feature engineering
        
        Args:
            df: DataFrame original
            include_title: Si incluir extracción de título (requiere columna Name)
            include_deck: Si incluir extracción de deck (requiere columna Cabin)
            
        Returns:
            pd.DataFrame: DataFrame con todas las nuevas features
        """
        df_new = df.copy()
        
        # Features básicas siempre disponibles
        df_new = self.create_family_size(df_new)
        df_new = self.create_is_alone(df_new)
        
        if 'Age' in df_new.columns:
            df_new = self.create_age_groups(df_new)
        
        if 'Fare' in df_new.columns:
            df_new = self.create_fare_bins(df_new)
        
        # Features opcionales
        if include_title and 'Name' in df_new.columns:
            df_new = self.extract_title(df_new)
        
        if include_deck and 'Cabin' in df_new.columns:
            df_new = self.create_deck(df_new)
        
        # Features de interacción
        df_new = self.create_interaction_features(df_new)
        
        return df_new
    
    def get_engineered_features_list(self) -> list:
        """
        Retorna lista de features creadas
        
        Returns:
            list: Lista de nombres de features
        """
        return [
            'FamilySize',
            'IsAlone',
            'AgeGroup',
            'FareBin',
            'Age_Class',
            'Fare_Class',
            'Sex_Class'
        ]


def engineer_features_simple(df: pd.DataFrame) -> pd.DataFrame:
    """
    Función simple para crear features básicas sin clases
    
    Args:
        df: DataFrame original
        
    Returns:
        pd.DataFrame: DataFrame con features básicas agregadas
    """
    df_new = df.copy()
    
    # Tamaño de familia
    if 'SibSp' in df_new.columns and 'Parch' in df_new.columns:
        df_new['FamilySize'] = df_new['SibSp'] + df_new['Parch'] + 1
        df_new['IsAlone'] = (df_new['FamilySize'] == 1).astype(int)
    
    # Grupos de edad
    if 'Age' in df_new.columns:
        df_new['AgeGroup'] = pd.cut(
            df_new['Age'],
            bins=[0, 12, 18, 35, 60, 100],
            labels=[0, 1, 2, 3, 4]
        )
        df_new['AgeGroup'] = df_new['AgeGroup'].astype(float)
    
    return df_new