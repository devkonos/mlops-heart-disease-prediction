"""
Model training and evaluation utilities
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)
import joblib
import os
from typing import Dict, Tuple, Any
import matplotlib.pyplot as plt
import seaborn as sns

class ModelTrainer:
    """Handles model training and evaluation"""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models = {}
        self.best_model = None
        self.best_model_name = None
    
    def train_logistic_regression(self, X_train: np.ndarray, y_train: np.ndarray, 
                                  **kwargs) -> LogisticRegression:
        """Train Logistic Regression model"""
        params = {'random_state': self.random_state, 'max_iter': 1000}
        params.update(kwargs)
        
        model = LogisticRegression(**params)
        model.fit(X_train, y_train)
        self.models['Logistic Regression'] = model
        
        return model
    
    def train_random_forest(self, X_train: np.ndarray, y_train: np.ndarray,
                           **kwargs) -> RandomForestClassifier:
        """Train Random Forest model"""
        params = {'n_estimators': 100, 'random_state': self.random_state, 
                 'max_depth': 10, 'min_samples_split': 5}
        params.update(kwargs)
        
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
        self.models['Random Forest'] = model
        
        return model
    
    def evaluate_model(self, model, X_test: np.ndarray, y_test: np.ndarray,
                      model_name: str = "Model") -> Dict[str, Any]:
        """
        Evaluate model performance
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test labels
            model_name: Name of the model
            
        Returns:
            Dictionary of metrics
        """
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        metrics = {
            'model_name': model_name,
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred),
        }
        
        return metrics
    
    def cross_validate_model(self, model, X: np.ndarray, y: np.ndarray,
                            cv_folds: int = 5) -> Dict[str, np.ndarray]:
        """
        Perform cross-validation
        
        Args:
            model: Model to evaluate
            X: Features
            y: Labels
            cv_folds: Number of cross-validation folds
            
        Returns:
            Cross-validation results
        """
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        
        scoring = {
            'accuracy': 'accuracy',
            'precision': 'precision',
            'recall': 'recall',
            'f1': 'f1',
            'roc_auc': 'roc_auc'
        }
        
        cv_results = cross_validate(model, X, y, cv=cv, scoring=scoring, return_train_score=True)
        
        return cv_results
    
    def save_model(self, model, filepath: str):
        """Save model to disk"""
        os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
        joblib.dump(model, filepath)
        print(f"Model saved to {filepath}")
    
    @staticmethod
    def load_model(filepath: str):
        """Load model from disk"""
        return joblib.load(filepath)
    
    def plot_confusion_matrix(self, confusion_mat: np.ndarray, 
                             model_name: str = "Model",
                             filepath: str = None):
        """Plot confusion matrix"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(confusion_mat, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if filepath:
            os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved to {filepath}")
        
        plt.close()


def compare_models(models_dict: Dict[str, Any], X_test: np.ndarray, 
                  y_test: np.ndarray) -> pd.DataFrame:
    """
    Compare multiple models
    
    Args:
        models_dict: Dictionary of models
        X_test: Test features
        y_test: Test labels
        
    Returns:
        DataFrame with comparison results
    """
    results = []
    
    for model_name, model in models_dict.items():
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        results.append({
            'Model': model_name,
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred),
            'Recall': recall_score(y_test, y_pred),
            'F1-Score': f1_score(y_test, y_pred),
            'ROC-AUC': roc_auc_score(y_test, y_pred_proba),
        })
    
    return pd.DataFrame(results)
