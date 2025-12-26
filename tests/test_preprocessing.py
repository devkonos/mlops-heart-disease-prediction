"""
Unit tests for data processing module
"""
import pytest
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data.preprocessing import DataPreprocessor, split_features_target


class TestDataPreprocessor:
    """Test cases for DataPreprocessor class"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing"""
        np.random.seed(42)
        data = {
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100),
            'feature3': np.random.randn(100),
            'target': np.random.randint(0, 2, 100)
        }
        return pd.DataFrame(data)
    
    def test_preprocessor_initialization(self):
        """Test preprocessor initialization"""
        preprocessor = DataPreprocessor()
        assert preprocessor.is_fitted is False
        assert preprocessor.feature_names is None
    
    def test_fit_method(self, sample_data):
        """Test fit method"""
        X = sample_data.drop('target', axis=1)
        y = sample_data['target']
        
        preprocessor = DataPreprocessor()
        result = preprocessor.fit(X, y)
        
        assert result is preprocessor  # Check that fit returns self
        assert preprocessor.is_fitted is True
        assert preprocessor.feature_names == list(X.columns)
    
    def test_transform_before_fit_raises_error(self, sample_data):
        """Test that transform raises error if not fitted"""
        X = sample_data.drop('target', axis=1)
        preprocessor = DataPreprocessor()
        
        with pytest.raises(ValueError):
            preprocessor.transform(X)
    
    def test_fit_transform(self, sample_data):
        """Test fit_transform method"""
        X = sample_data.drop('target', axis=1)
        y = sample_data['target']
        
        preprocessor = DataPreprocessor()
        X_transformed = preprocessor.fit_transform(X, y)
        
        assert X_transformed.shape == X.shape
        assert isinstance(X_transformed, np.ndarray)
        assert preprocessor.is_fitted is True
    
    def test_transform_shape_preservation(self, sample_data):
        """Test that transform preserves shape"""
        X = sample_data.drop('target', axis=1)
        
        preprocessor = DataPreprocessor()
        preprocessor.fit(X)
        X_transformed = preprocessor.transform(X)
        
        assert X_transformed.shape == X.shape
    
    def test_scaling_normalization(self, sample_data):
        """Test that data is properly scaled"""
        X = sample_data.drop('target', axis=1)
        
        preprocessor = DataPreprocessor()
        X_transformed = preprocessor.fit_transform(X)
        
        # Check that transformed data has mean close to 0 and std close to 1
        assert np.abs(X_transformed.mean()) < 1e-10
        assert np.abs(X_transformed.std() - 1.0) < 0.1


class TestSplitFeaturesTarget:
    """Test cases for split_features_target function"""
    
    @pytest.fixture
    def sample_dataframe(self):
        """Create sample DataFrame"""
        return pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [5, 4, 3, 2, 1],
            'target': [0, 1, 0, 1, 0]
        })
    
    def test_split_with_default_target_column(self, sample_dataframe):
        """Test split with default target column name"""
        X, y = split_features_target(sample_dataframe)
        
        assert X.shape == (5, 2)
        assert y.shape == (5,)
        assert list(X.columns) == ['feature1', 'feature2']
        assert list(y) == [0, 1, 0, 1, 0]
    
    def test_split_with_custom_target_column(self, sample_dataframe):
        """Test split with custom target column name"""
        sample_dataframe.rename(columns={'target': 'label'}, inplace=True)
        X, y = split_features_target(sample_dataframe, target_col='label')
        
        assert X.shape == (5, 2)
        assert y.shape == (5,)
    
    def test_split_preserves_data_types(self, sample_dataframe):
        """Test that split preserves data types"""
        X, y = split_features_target(sample_dataframe)
        
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)


class TestDataCleaning:
    """Test cases for data cleaning functions"""
    
    def test_missing_value_handling(self):
        """Test handling of missing values"""
        preprocessor = DataPreprocessor()
        
        # Create data with missing values
        X = pd.DataFrame({
            'feature1': [1, 2, np.nan, 4, 5],
            'feature2': [5, np.nan, 3, 2, 1],
            'feature3': [1, 2, 3, 4, 5]
        })
        
        # Fit and transform
        X_transformed = preprocessor.fit_transform(X)
        
        # Check no NaN values in output
        assert not np.isnan(X_transformed).any()
        assert X_transformed.shape == X.shape
    
    def test_preprocessing_consistency(self):
        """Test that preprocessing is consistent"""
        preprocessor = DataPreprocessor()
        
        X = pd.DataFrame(np.random.randn(100, 5))
        X_train = X[:80]
        X_test = X[80:]
        
        preprocessor.fit(X_train)
        X_train_transformed = preprocessor.transform(X_train)
        X_test_transformed = preprocessor.transform(X_test)
        
        # Both should have same number of features
        assert X_train_transformed.shape[1] == X_test_transformed.shape[1]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
