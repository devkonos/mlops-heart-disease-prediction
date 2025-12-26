# Installation & Setup Guide

## Quick Start (5 minutes)

### Prerequisites
- Python 3.9+
- pip (Python package manager)
- Git

### Steps

1. **Clone Repository**
```bash
git clone <repository-url>
cd heart-disease-prediction
```

2. **Create Virtual Environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install Dependencies**
```bash
pip install -r requirements.txt
```

4. **Download Dataset**
```bash
python src/data/download_data.py
```

5. **Train Models**
```bash
python scripts/train_model.py
```

6. **Start API**
```bash
python -m uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```

7. **Access Services**
- API: http://localhost:8000
- API Docs: http://localhost:8000/docs
- Health: http://localhost:8000/health

---

## Full Installation Guide

### 1. Environment Setup

#### Using venv (Recommended for development)
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# OR
venv\Scripts\activate  # Windows
```

#### Using conda (Alternative)
```bash
conda create -n heart-disease python=3.9
conda activate heart-disease
```

### 2. Install Dependencies

**For production:**
```bash
pip install -r requirements.txt
```

**For development (includes testing tools):**
```bash
pip install -r requirements.txt
pip install pytest pytest-cov flake8 black jupyter
```

### 3. Verify Installation

```bash
# Check Python version
python --version

# Check key packages
python -c "import pandas; print(f'Pandas {pandas.__version__}')"
python -c "import sklearn; print(f'scikit-learn {sklearn.__version__}')"
python -c "import mlflow; print(f'MLflow {mlflow.__version__}')"
python -c "import fastapi; print(f'FastAPI installed')"
```

### 4. Data Setup

#### Automatic Download
```bash
python src/data/download_data.py
```

#### Manual Download
1. Visit: https://archive.ics.uci.edu/ml/datasets/heart+disease
2. Download `processed.cleveland.data`
3. Save to `data/raw/heart_disease.csv`

### 5. Model Training

```bash
# Quick training
python scripts/train_model.py

# Or use Jupyter notebook for interactive training
jupyter notebook notebooks/01_EDA_and_Model_Training.ipynb
```

Expected output:
- Trained models in `models/artifacts/`
- Preprocessing pipeline in `models/artifacts/`
- MLflow runs in `mlruns/`

### 6. Testing Setup

```bash
# Run unit tests
pytest tests/ -v

# With coverage report
pytest tests/ --cov=src --cov-report=html
```

### 7. API Setup

#### Local Development
```bash
# Using uvicorn directly
python -m uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload

# Using Python
python src/api/app.py
```

#### Using Docker
```bash
# Build image
docker build -f docker/Dockerfile -t heart-disease-api:latest .

# Run container
docker run -p 8000:8000 heart-disease-api:latest
```

#### Using Docker Compose
```bash
docker-compose -f docker/docker-compose.yml up -d
```

### 8. Experiment Tracking

```bash
# Start MLflow UI
mlflow ui --backend-store-uri file:mlruns

# Access at http://localhost:5000
```

### 9. Kubernetes Setup (Optional)

#### Prerequisites
- kubectl installed
- Minikube or cloud cluster access

#### Local Deployment (Minikube)
```bash
# Start Minikube
minikube start

# Build image in Minikube
eval $(minikube docker-env)
docker build -f docker/Dockerfile -t heart-disease-api:latest .

# Deploy
kubectl apply -f k8s/deployment.yaml

# Access
kubectl port-forward svc/heart-disease-api 8000:80 -n heart-disease-prediction
```

---

## Troubleshooting

### Issue: Python version mismatch
**Solution:**
```bash
# Check Python version
python --version

# If version < 3.9, install Python 3.9+
# Then create new virtual environment
```

### Issue: Package installation fails
**Solution:**
```bash
# Upgrade pip
pip install --upgrade pip

# Retry installation
pip install -r requirements.txt --force-reinstall
```

### Issue: Cannot download dataset
**Solution:**
```bash
# Check internet connection
ping archive.ics.uci.edu

# Manual download and placement
# 1. Download from https://archive.ics.uci.edu/ml/datasets/heart+disease
# 2. Save to data/raw/heart_disease.csv
```

### Issue: Model files not found
**Solution:**
```bash
# Retrain models
python scripts/train_model.py

# Verify files exist
ls -la models/artifacts/
```

### Issue: Port already in use
**Solution:**
```bash
# Use different port
python -m uvicorn src.api.app:app --port 8001

# Or kill existing process
# Linux/Mac: lsof -i :8000 | kill -9 <PID>
# Windows: netstat -ano | findstr :8000
```

### Issue: Docker build fails
**Solution:**
```bash
# Clear Docker cache
docker system prune

# Rebuild
docker build -f docker/Dockerfile -t heart-disease-api:latest --no-cache .
```

---

## Development Setup

### IDE Setup

**VS Code:**
1. Install Python extension
2. Select Python interpreter from virtual environment
3. Install Pylance for better code completion

**PyCharm:**
1. Open project folder
2. Configure interpreter: Preferences > Project > Python Interpreter
3. Select virtual environment

### Pre-commit Hooks (Optional)
```bash
# Install pre-commit
pip install pre-commit

# Setup hooks
pre-commit install

# Run hooks manually
pre-commit run --all-files
```

### Development Workflow
```bash
# Make changes to code
# Run tests
make test

# Format code
make format

# Run linting
make lint

# Train models
make train

# Start API
make serve-api
```

---

## Project Structure

```
.
├── README.md                    # Project overview
├── INSTALLATION.md              # This file
├── DEPLOYMENT.md                # Deployment guide
├── requirements.txt             # Python dependencies
├── setup.py                     # Package setup
├── Makefile                     # Build automation
│
├── src/                         # Source code
│   ├── data/                   # Data loading & preprocessing
│   ├── models/                 # Model training & evaluation
│   ├── features/               # Feature engineering
│   ├── api/                    # FastAPI application
│   ├── config.py               # Configuration
│   └── monitoring.py           # Monitoring & logging
│
├── notebooks/                  # Jupyter notebooks
│   └── 01_EDA_and_Model_Training.ipynb
│
├── tests/                      # Unit tests
│   ├── test_preprocessing.py
│   ├── test_models.py
│   ├── test_api.py
│   └── conftest.py
│
├── scripts/                    # Utility scripts
│   ├── train_model.py
│   └── test_api.sh
│
├── docker/                     # Docker files
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── run_container.sh
│
├── k8s/                        # Kubernetes manifests
│   ├── deployment.yaml
│   ├── ingress.yaml
│   └── configmap.yaml
│
├── monitoring/                 # Monitoring configs
│   └── prometheus-grafana.yaml
│
├── data/                       # Data directory
│   ├── raw/                   # Original dataset
│   └── processed/             # Processed datasets
│
├── models/                     # Model directory
│   └── artifacts/             # Trained models
│
├── logs/                       # Application logs
│
├── screenshots/                # Deployment screenshots
│
└── .github/workflows/          # CI/CD pipelines
    └── mlops_pipeline.yml
```

---

## Next Steps

After installation:

1. **Run Tests**
   ```bash
   make test
   ```

2. **Train Models**
   ```bash
   make train
   ```

3. **Start API**
   ```bash
   make serve-api
   ```

4. **View Documentation**
   - Open http://localhost:8000/docs in browser

5. **Check MLflow**
   ```bash
   make mlflow-ui
   ```

6. **Read Documentation**
   - [README.md](README.md) - Project overview
   - [DEPLOYMENT.md](DEPLOYMENT.md) - Deployment guide

---

## Support

For issues or questions:
1. Check troubleshooting section
2. Review GitHub Issues
3. Check project documentation
4. Contact maintainers

---

## Additional Resources

- Python Virtual Environments: https://docs.python.org/3/venv/
- pip Documentation: https://pip.pypa.io/
- scikit-learn: https://scikit-learn.org/
- MLflow: https://mlflow.org/docs/
- FastAPI: https://fastapi.tiangolo.com/
- Docker: https://docs.docker.com/
- Kubernetes: https://kubernetes.io/docs/
