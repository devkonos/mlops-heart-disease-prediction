"""
Configuration settings for the MLOps project
"""
import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.absolute()
DATA_RAW_PATH = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_PATH = PROJECT_ROOT / "data" / "processed"
MODELS_PATH = PROJECT_ROOT / "models" / "artifacts"
LOGS_PATH = PROJECT_ROOT / "logs"
SCREENSHOTS_PATH = PROJECT_ROOT / "screenshots"

# Create directories if they don't exist
for path in [DATA_RAW_PATH, DATA_PROCESSED_PATH, MODELS_PATH, LOGS_PATH, SCREENSHOTS_PATH]:
    path.mkdir(parents=True, exist_ok=True)

# Model configuration
MODEL_CONFIG = {
    'random_state': 42,
    'test_size': 0.2,
    'cv_folds': 5,
    'logistic_regression': {
        'C_values': [0.1, 1.0, 10.0],
        'solvers': ['lbfgs', 'liblinear'],
        'max_iter': 1000,
    },
    'random_forest': {
        'n_estimators': [50, 100, 200],
        'max_depths': [5, 10, 15],
        'min_samples_split': 5,
    }
}

# Data configuration
DATA_CONFIG = {
    'target_column': 'target',
    'test_size': 0.2,
    'random_state': 42,
    'missing_value_strategy': 'median',
}

# API configuration
API_CONFIG = {
    'host': os.getenv('API_HOST', '0.0.0.0'),
    'port': int(os.getenv('API_PORT', 8000)),
    'debug': os.getenv('API_DEBUG', 'False').lower() == 'true',
    'reload': os.getenv('API_RELOAD', 'False').lower() == 'true',
}

# MLflow configuration
MLFLOW_CONFIG = {
    'tracking_uri': os.getenv('MLFLOW_TRACKING_URI', 'file:mlruns'),
    'experiment_name': 'heart_disease_prediction',
    'backend_store_uri': os.getenv('MLFLOW_BACKEND_STORE_URI', 'file:mlruns'),
    'default_artifact_root': os.getenv('MLFLOW_ARTIFACT_ROOT', 'mlruns/artifacts'),
}

# Logging configuration
LOGGING_CONFIG = {
    'level': os.getenv('LOG_LEVEL', 'INFO'),
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'log_file': str(LOGS_PATH / 'app.log'),
}

# Feature names for the model
FEATURE_NAMES = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
    'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
]

# Target mapping
TARGET_MAPPING = {
    0: 'No Disease',
    1: 'Disease Present'
}

# Environment
ENVIRONMENT = os.getenv('ENVIRONMENT', 'development')
DEBUG = ENVIRONMENT == 'development'
