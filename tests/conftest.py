"""
Test configuration and fixtures
"""
import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification


@pytest.fixture(scope="session")
def sample_dataset():
    """Create sample classification dataset"""
    X, y = make_classification(
        n_samples=500, 
        n_features=13,
        n_informative=10,
        n_redundant=3,
        random_state=42
    )
    return X, y


@pytest.fixture
def train_test_data(sample_dataset):
    """Split data into train and test sets"""
    X, y = sample_dataset
    train_size = int(0.8 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    return X_train, X_test, y_train, y_test
