"""
Data download script for Heart Disease UCI Dataset
Downloads from UCI Machine Learning Repository
"""
import os
import urllib.request
import pandas as pd
from pathlib import Path

def download_heart_disease_data(output_dir: str = "data/raw") -> str:
    """
    Download Heart Disease UCI Dataset
    
    Args:
        output_dir: Directory to save the dataset
        
    Returns:
        Path to the downloaded CSV file
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # UCI Heart Disease dataset URL
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
    output_path = os.path.join(output_dir, "heart_disease.csv")
    
    print(f"Downloading Heart Disease dataset from {url}...")
    try:
        urllib.request.urlretrieve(url, output_path)
        print(f"Dataset successfully downloaded to {output_path}")
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        print("Alternative: Download manually from https://archive.ics.uci.edu/ml/datasets/heart+disease")
        return None
    
    return output_path

def load_and_prepare_data(csv_path: str) -> pd.DataFrame:
    """
    Load and prepare the Heart Disease dataset
    
    Args:
        csv_path: Path to the CSV file
        
    Returns:
        Prepared DataFrame
    """
    # Column names from UCI repository
    column_names = [
        'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
        'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
    ]
    
    # Load data with column names and handle missing values (marked as '?')
    df = pd.read_csv(csv_path, header=None, names=column_names, na_values='?')
    
    print(f"Dataset shape: {df.shape}")
    print(f"\nMissing values:\n{df.isnull().sum()}")
    print(f"\nFirst few rows:\n{df.head()}")
    print(f"\nData types:\n{df.dtypes}")
    print(f"\nTarget distribution:\n{df['target'].value_counts()}")
    
    return df

if __name__ == "__main__":
    # Download the dataset
    csv_path = download_heart_disease_data()
    
    if csv_path and os.path.exists(csv_path):
        # Load and display data info
        df = load_and_prepare_data(csv_path)
        print("\nData preparation complete!")
    else:
        print("Failed to download dataset. Please download manually.")
