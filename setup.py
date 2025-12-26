from setuptools import setup, find_packages

setup(
    name="heart-disease-mlops",
    version="1.0.0",
    description="MLOps End-to-End ML Model Development for Heart Disease Prediction",
    author="MLOps Team",
    packages=find_packages(),
    install_requires=[
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "mlflow>=2.7.0",
        "fastapi>=0.103.0",
        "pytest>=7.4.0",
    ],
    python_requires=">=3.9",
)
