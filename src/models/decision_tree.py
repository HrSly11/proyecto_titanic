"""
Módulo para el modelo de Árbol de Decisión
"""
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np


class DecisionTreeModel:
    """Clase para entrenar y evaluar Árbol de Decisión"""
    
    def __init__(self, max_depth=5, min_samples_split=20, min_samples_leaf=10, 
                 random_state=42):
        """
        Inicializa el modelo
        
        Args:
            max_depth: Profundidad máxima del árbol
            min_samples_split: Mínimo de muestras para dividir
            min_samples_leaf: Mínimo de muestras en hoja
            random_state: Semilla aleatoria
        """
        self.model = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state
        )
        self.is_trained = False
        self.feature_names = None
    
    def train(self, X_train, y_train):
        """
        Entrena el modelo
        
        Args:
            X_train: Features de entrenamiento
            y_train: Target de entrenamiento
        """
        self.model.fit(X_train, y_train)
        self.is_trained = True
        self.feature_names = X_train.columns.tolist() if hasattr(X_train, 'columns') else None
    
    def predict(self, X):
        """
        Realiza predicciones
        
        Args:
            X: Features para predecir
            
        Returns:
            np.array: Predicciones
        """
        if not self.is_trained:
            raise ValueError("El modelo no ha sido entrenado")
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """
        Obtiene probabilidades de predicción
        
        Args:
            X: Features para predecir
            
        Returns:
            np.array: Probabilidades
        """
        if not self.is_trained:
            raise ValueError("El modelo no ha sido entrenado")
        return self.model.predict_proba(X)
    
    def evaluate(self, X_test, y_test):
        """
        Evalúa el modelo
        
        Args:
            X_test: Features de prueba
            y_test: Target de prueba
            
        Returns:
            dict: Métricas de evaluación
        """
        y_pred = self.predict(X_test)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='binary'),
            'recall': recall_score(y_test, y_pred, average='binary'),
            'f1_score': f1_score(y_test, y_pred, average='binary'),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred, 
                                                           target_names=['Died', 'Survived'])
        }
        
        return metrics
    
    def get_feature_importance(self):
        """
        Obtiene la importancia de las características
        
        Returns:
            dict: Importancia de cada característica
        """
        if not self.is_trained:
            raise ValueError("El modelo no ha sido entrenado")
        
        importances = self.model.feature_importances_
        
        if self.feature_names:
            importance_dict = dict(zip(self.feature_names, importances))
            # Ordenar por importancia
            importance_dict = dict(sorted(importance_dict.items(), 
                                         key=lambda x: x[1], reverse=True))
        else:
            importance_dict = {f'Feature_{i}': imp 
                              for i, imp in enumerate(importances)}
        
        return importance_dict
    
    def get_tree_depth(self):
        """
        Obtiene la profundidad real del árbol
        
        Returns:
            int: Profundidad del árbol
        """
        if not self.is_trained:
            raise ValueError("El modelo no ha sido entrenado")
        return self.model.get_depth()
    
    def get_n_leaves(self):
        """
        Obtiene el número de hojas del árbol
        
        Returns:
            int: Número de hojas
        """
        if not self.is_trained:
            raise ValueError("El modelo no ha sido entrenado")
        return self.model.get_n_leaves()
    
    def analyze_overfitting(self, X_train, y_train, X_test, y_test):
        """
        Analiza el sobreajuste comparando train vs test
        
        Args:
            X_train: Features de entrenamiento
            y_train: Target de entrenamiento
            X_test: Features de prueba
            y_test: Target de prueba
            
        Returns:
            dict: Análisis de sobreajuste
        """
        train_score = self.model.score(X_train, y_train)
        test_score = self.model.score(X_test, y_test)
        
        diff = train_score - test_score
        
        analysis = {
            'train_accuracy': train_score,
            'test_accuracy': test_score,
            'difference': diff,
            'overfitting_level': 'Alto' if diff > 0.1 else 'Moderado' if diff > 0.05 else 'Bajo',
            'description': self._get_overfitting_description(diff)
        }
        
        return analysis
    
    def _get_overfitting_description(self, diff):
        """
        Genera descripción del nivel de sobreajuste
        
        Args:
            diff: Diferencia entre train y test accuracy
            
        Returns:
            str: Descripción
        """
        if diff > 0.1:
            return "El modelo tiene sobreajuste alto. Considera reducir la profundidad o aumentar min_samples."
        elif diff > 0.05:
            return "El modelo tiene sobreajuste moderado. El rendimiento es aceptable pero podría mejorarse."
        else:
            return "El modelo tiene buen balance entre sesgo y varianza."