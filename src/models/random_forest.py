"""
Módulo para el modelo de Random Forest
"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np


class RandomForestModel:
    """Clase para entrenar y evaluar Random Forest"""
    
    def __init__(self, n_estimators=100, max_depth=10, min_samples_split=20, 
                 min_samples_leaf=5, max_features='sqrt', random_state=42, n_jobs=-1):
        """
        Inicializa el modelo
        
        Args:
            n_estimators: Número de árboles
            max_depth: Profundidad máxima
            min_samples_split: Mínimo de muestras para dividir
            min_samples_leaf: Mínimo de muestras en hoja
            max_features: Features a considerar en cada split
            random_state: Semilla aleatoria
            n_jobs: Número de cores a usar
        """
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=random_state,
            n_jobs=n_jobs
        )
        self.is_trained = False
        self.feature_names = None
        self.cv_scores = None
    
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
    
    def cross_validate(self, X, y, cv=5):
        """
        Realiza validación cruzada
        
        Args:
            X: Features
            y: Target
            cv: Número de folds
            
        Returns:
            dict: Resultados de validación cruzada
        """
        scores = cross_val_score(self.model, X, y, cv=cv, scoring='accuracy')
        self.cv_scores = scores
        
        cv_results = {
            'scores': scores,
            'mean_score': scores.mean(),
            'std_score': scores.std(),
            'min_score': scores.min(),
            'max_score': scores.max()
        }
        
        return cv_results
    
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
    
    def get_oob_score(self):
        """
        Obtiene el Out-of-Bag score si está disponible
        
        Returns:
            float: OOB score
        """
        if not self.is_trained:
            raise ValueError("El modelo no ha sido entrenado")
        
        if hasattr(self.model, 'oob_score_'):
            return self.model.oob_score_
        else:
            return None
    
    def analyze_ensemble(self):
        """
        Analiza las características del ensemble
        
        Returns:
            dict: Información del ensemble
        """
        if not self.is_trained:
            raise ValueError("El modelo no ha sido entrenado")
        
        ensemble_info = {
            'n_estimators': self.model.n_estimators,
            'max_depth': self.model.max_depth,
            'n_features': self.model.n_features_in_,
            'n_classes': self.model.n_classes_
        }
        
        return ensemble_info