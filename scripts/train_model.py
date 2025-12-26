"""
Main model training script
"""
import os
import sys
import logging
import pandas as pd
import numpy as np
from pathlib import Path

# Setup paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data.download_data import download_heart_disease_data, load_and_prepare_data
from src.data.preprocessing import DataPreprocessor, split_features_target
from src.models.train import ModelTrainer, compare_models
from src.monitoring import log_model_metrics

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

import mlflow
import mlflow.sklearn

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main training pipeline"""
    
    logger.info("=" * 60)
    logger.info("STARTING MODEL TRAINING PIPELINE")
    logger.info("=" * 60)
    
    # Step 1: Data Download and Loading
    logger.info("\n[1/7] Downloading and loading data...")
    csv_path = download_heart_disease_data(output_dir='data/raw')
    
    if not csv_path or not os.path.exists(csv_path):
        logger.error("Failed to download dataset")
        return False
    
    df = load_and_prepare_data(csv_path)
    logger.info(f"Data loaded: {df.shape}")
    
    # Step 2: Data Cleaning and Preprocessing
    logger.info("\n[2/7] Cleaning and preprocessing data...")
    df_clean = df.dropna()
    df_clean['target'] = (df_clean['target'] > 0).astype(int)
    
    X, y = split_features_target(df_clean, target_col='target')
    logger.info(f"Features shape: {X.shape}, Target shape: {y.shape}")
    
    # Step 3: Create and Fit Preprocessor
    logger.info("\n[3/7] Fitting preprocessor...")
    preprocessor = DataPreprocessor()
    X_preprocessed = preprocessor.fit_transform(X, y)
    
    # Save preprocessor
    os.makedirs('models/artifacts', exist_ok=True)
    preprocessor.save('models/artifacts/preprocessor.pkl')
    logger.info("Preprocessor saved")
    
    # Step 4: Train-Test Split
    logger.info("\n[4/7] Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_preprocessed, y, test_size=0.2, random_state=42, stratify=y
    )
    logger.info(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    
    # Step 5: Model Training with MLflow
    logger.info("\n[5/7] Training models with MLflow tracking...")
    
    # Setup MLflow
    mlflow.set_tracking_uri(uri='file:mlruns')
    experiment_name = "heart_disease_prediction"
    mlflow.set_experiment(experiment_name)
    
    trainer = ModelTrainer(random_state=42)
    
    # Logistic Regression
    logger.info("Training Logistic Regression...")
    lr_params = {'C': [0.1, 1.0, 10.0], 'solver': ['lbfgs', 'liblinear']}
    lr_grid = GridSearchCV(LogisticRegression(random_state=42, max_iter=1000),
                          lr_params, cv=5, scoring='roc_auc', n_jobs=-1)
    lr_grid.fit(X_train, y_train)
    lr_best = lr_grid.best_estimator_
    
    with mlflow.start_run(run_name="LogisticRegression_v1"):
        mlflow.log_params({
            'model_type': 'LogisticRegression',
            'C': lr_grid.best_params_['C'],
            'solver': lr_grid.best_params_['solver'],
            'max_iter': 1000,
        })
        
        lr_metrics = trainer.evaluate_model(lr_best, X_test, y_test)
        mlflow.log_metrics({
            'accuracy': lr_metrics['accuracy'],
            'precision': lr_metrics['precision'],
            'recall': lr_metrics['recall'],
            'f1_score': lr_metrics['f1'],
            'roc_auc': lr_metrics['roc_auc'],
        })
        mlflow.sklearn.log_model(lr_best, artifact_path="model")
        
        log_model_metrics('LogisticRegression',
                         lr_metrics['accuracy'],
                         lr_metrics['precision'],
                         lr_metrics['recall'],
                         lr_metrics['f1'])
        
        logger.info(f"LR Metrics - Accuracy: {lr_metrics['accuracy']:.4f}, "
                   f"ROC-AUC: {lr_metrics['roc_auc']:.4f}")
    
    # Random Forest
    logger.info("Training Random Forest...")
    rf_params = {'n_estimators': [50, 100, 200], 'max_depth': [5, 10, 15]}
    rf_grid = GridSearchCV(RandomForestClassifier(random_state=42),
                          rf_params, cv=5, scoring='roc_auc', n_jobs=-1)
    rf_grid.fit(X_train, y_train)
    rf_best = rf_grid.best_estimator_
    
    with mlflow.start_run(run_name="RandomForest_v1"):
        mlflow.log_params({
            'model_type': 'RandomForest',
            'n_estimators': rf_grid.best_params_['n_estimators'],
            'max_depth': rf_grid.best_params_['max_depth'],
        })
        
        rf_metrics = trainer.evaluate_model(rf_best, X_test, y_test)
        mlflow.log_metrics({
            'accuracy': rf_metrics['accuracy'],
            'precision': rf_metrics['precision'],
            'recall': rf_metrics['recall'],
            'f1_score': rf_metrics['f1'],
            'roc_auc': rf_metrics['roc_auc'],
        })
        mlflow.sklearn.log_model(rf_best, artifact_path="model")
        
        log_model_metrics('RandomForest',
                         rf_metrics['accuracy'],
                         rf_metrics['precision'],
                         rf_metrics['recall'],
                         rf_metrics['f1'])
        
        logger.info(f"RF Metrics - Accuracy: {rf_metrics['accuracy']:.4f}, "
                   f"ROC-AUC: {rf_metrics['roc_auc']:.4f}")
    
    # Step 6: Save Models
    logger.info("\n[6/7] Saving models...")
    trainer.save_model(lr_best, 'models/artifacts/logistic_regression_model.pkl')
    trainer.save_model(rf_best, 'models/artifacts/random_forest_model.pkl')
    
    # Step 7: Model Comparison
    logger.info("\n[7/7] Model comparison...")
    models_dict = {'Logistic Regression': lr_best, 'Random Forest': rf_best}
    comparison_df = compare_models(models_dict, X_test, y_test)
    
    logger.info("\nModel Comparison Results:")
    logger.info(comparison_df.to_string(index=False))
    
    # Save comparison
    os.makedirs('data/processed', exist_ok=True)
    comparison_df.to_csv('data/processed/model_comparison.csv', index=False)
    
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING PIPELINE COMPLETED SUCCESSFULLY")
    logger.info("=" * 60)
    logger.info(f"Models saved to: models/artifacts/")
    logger.info(f"MLflow URI: file:mlruns")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
