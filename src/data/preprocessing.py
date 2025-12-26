"""
Data preprocessing and cleaning utilities
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from typing import Tuple, Dict, Any
import joblib
import os

class DataPreprocessor:
    """Handles data cleaning and preprocessing"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='median')
        self.encoders: Dict[str, LabelEncoder] = {}
        self.feature_names = None
        self.is_fitted = False
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and prepare data
        
        Args:
            df: Input DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        df_clean = df.copy()
        
        # Remove rows with missing values for critical features
        # For heart disease data, remove rows with '?' or NaN
        df_clean = df_clean.dropna()
        
        # Convert target to binary (0 = no disease, 1+ = disease present)
        if 'target' in df_clean.columns:
            df_clean['target'] = (df_clean['target'] > 0).astype(int)
        
        # Handle categorical features
        categorical_cols = df_clean.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            # Try to convert to numeric if possible
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
        
        return df_clean
    
    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> 'DataPreprocessor':
        """
        Fit preprocessor on training data
        
        Args:
            X: Feature data
            y: Target data (not used but included for consistency)
            
        Returns:
            Self
        """
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        # Fit scaler on numerical features
        self.scaler.fit(X)
        
        # Fit imputer
        self.imputer.fit(X)
        
        self.is_fitted = True
        return self
    
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Transform data using fitted preprocessor
        
        Args:
            X: Feature data
            
        Returns:
            Transformed numpy array
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transform")
        
        X_transformed = self.imputer.transform(X)
        X_scaled = self.scaler.transform(X_transformed)
        
        return X_scaled
    
    def fit_transform(self, X: pd.DataFrame, y: pd.Series = None) -> np.ndarray:
        """
        Fit and transform data
        
        Args:
            X: Feature data
            y: Target data
            
        Returns:
            Transformed numpy array
        """
        return self.fit(X, y).transform(X)
    
    def save(self, filepath: str):
        """Save preprocessor to disk"""
        os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
        joblib.dump(self, filepath)
        print(f"Preprocessor saved to {filepath}")
    
    @staticmethod
    def load(filepath: str) -> 'DataPreprocessor':
        """Load preprocessor from disk"""
        return joblib.load(filepath)


def split_features_target(df: pd.DataFrame, target_col: str = 'target') -> Tuple[pd.DataFrame, pd.Series]:
    """
    Split DataFrame into features and target
    
    Args:
        df: Input DataFrame
        target_col: Name of target column
        
    Returns:
        Tuple of (features, target)
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y
