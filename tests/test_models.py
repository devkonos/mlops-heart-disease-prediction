"""
Unit tests for model training and evaluation
"""
import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.models.train import ModelTrainer, compare_models


class TestModelTrainer:
    """Test cases for ModelTrainer class"""
    
    @pytest.fixture
    def sample_classification_data(self):
        """Create sample classification data"""
        X, y = make_classification(n_samples=200, n_features=13, n_informative=10,
                                  n_redundant=3, random_state=42)
        X_train, X_test = X[:160], X[160:]
        y_train, y_test = y[:160], y[160:]
        return X_train, X_test, y_train, y_test
    
    def test_trainer_initialization(self):
        """Test trainer initialization"""
        trainer = ModelTrainer(random_state=42)
        assert trainer.random_state == 42
        assert len(trainer.models) == 0
    
    def test_train_logistic_regression(self, sample_classification_data):
        """Test Logistic Regression training"""
        X_train, _, y_train, _ = sample_classification_data
        
        trainer = ModelTrainer()
        model = trainer.train_logistic_regression(X_train, y_train)
        
        assert model is not None
        assert 'Logistic Regression' in trainer.models
    
    def test_train_random_forest(self, sample_classification_data):
        """Test Random Forest training"""
        X_train, _, y_train, _ = sample_classification_data
        
        trainer = ModelTrainer()
        model = trainer.train_random_forest(X_train, y_train)
        
        assert model is not None
        assert 'Random Forest' in trainer.models
    
    def test_model_prediction(self, sample_classification_data):
        """Test model prediction"""
        X_train, X_test, y_train, _ = sample_classification_data
        
        trainer = ModelTrainer()
        model = trainer.train_logistic_regression(X_train, y_train)
        
        predictions = model.predict(X_test)
        
        assert predictions.shape == (len(X_test),)
        assert np.all((predictions == 0) | (predictions == 1))
    
    def test_evaluate_model(self, sample_classification_data):
        """Test model evaluation"""
        X_train, X_test, y_train, y_test = sample_classification_data
        
        trainer = ModelTrainer()
        model = trainer.train_logistic_regression(X_train, y_train)
        
        metrics = trainer.evaluate_model(model, X_test, y_test)
        
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1' in metrics
        assert 'roc_auc' in metrics
        assert 0 <= metrics['accuracy'] <= 1
        assert 0 <= metrics['roc_auc'] <= 1
    
    def test_cross_validate_model(self, sample_classification_data):
        """Test cross-validation"""
        X_train, _, y_train, _ = sample_classification_data
        
        trainer = ModelTrainer()
        model = trainer.train_logistic_regression(X_train, y_train)
        
        cv_results = trainer.cross_validate_model(model, X_train, y_train, cv_folds=5)
        
        assert 'test_accuracy' in cv_results
        assert len(cv_results['test_accuracy']) == 5
        assert np.all((cv_results['test_accuracy'] >= 0) & (cv_results['test_accuracy'] <= 1))
    
    def test_model_save_and_load(self, sample_classification_data, tmp_path):
        """Test model saving and loading"""
        X_train, _, y_train, _ = sample_classification_data
        
        trainer = ModelTrainer()
        model = trainer.train_logistic_regression(X_train, y_train)
        
        # Save model
        model_path = str(tmp_path / "test_model.pkl")
        trainer.save_model(model, model_path)
        
        # Load model
        loaded_model = trainer.load_model(model_path)
        
        # Check that loaded model makes same predictions
        test_pred_original = model.predict(X_train[:10])
        test_pred_loaded = loaded_model.predict(X_train[:10])
        
        assert np.array_equal(test_pred_original, test_pred_loaded)


class TestCompareModels:
    """Test cases for model comparison"""
    
    @pytest.fixture
    def sample_models_and_data(self):
        """Create sample models and data"""
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier
        
        X, y = make_classification(n_samples=200, n_features=13, random_state=42)
        X_train, X_test = X[:160], X[160:]
        y_train, y_test = y[:160], y[160:]
        
        lr = LogisticRegression(random_state=42, max_iter=1000)
        rf = RandomForestClassifier(random_state=42, n_estimators=50)
        
        lr.fit(X_train, y_train)
        rf.fit(X_train, y_train)
        
        models = {'LogisticRegression': lr, 'RandomForest': rf}
        
        return models, X_test, y_test
    
    def test_compare_models_output(self, sample_models_and_data):
        """Test compare_models function output"""
        models, X_test, y_test = sample_models_and_data
        
        comparison_df = compare_models(models, X_test, y_test)
        
        assert isinstance(comparison_df, pd.DataFrame)
        assert len(comparison_df) == 2
        assert 'Model' in comparison_df.columns
        assert 'Accuracy' in comparison_df.columns
        assert 'Precision' in comparison_df.columns
        assert 'Recall' in comparison_df.columns
        assert 'F1-Score' in comparison_df.columns
        assert 'ROC-AUC' in comparison_df.columns


class TestModelMetrics:
    """Test cases for model metrics"""
    
    def test_metrics_range(self):
        """Test that metrics are within valid ranges"""
        from sklearn.linear_model import LogisticRegression
        
        X, y = make_classification(n_samples=200, n_features=13, random_state=42)
        X_train, X_test = X[:160], X[160:]
        y_train, y_test = y[:160], y[160:]
        
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X_train, y_train)
        
        trainer = ModelTrainer()
        metrics = trainer.evaluate_model(model, X_test, y_test)
        
        # All metrics should be between 0 and 1
        assert 0 <= metrics['accuracy'] <= 1
        assert 0 <= metrics['precision'] <= 1
        assert 0 <= metrics['recall'] <= 1
        assert 0 <= metrics['f1'] <= 1
        assert 0 <= metrics['roc_auc'] <= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
