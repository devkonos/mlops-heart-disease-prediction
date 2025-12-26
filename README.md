# Heart Disease Prediction - MLOps End-to-End Pipeline

A MLOps solution for predicting heart disease risk using machine learning, featuring data pipelines, experiment tracking, CI/CD, containerization, and cloud deployment.

## Project Overview

This project demonstrates MLOps including:
- **Data Management**: Automated data acquisition, cleaning, and preprocessing
- **Model Development**: Multiple classification models with hyperparameter tuning
- **Experiment Tracking**: MLflow integration for tracking runs and artifacts
- **Testing**: Comprehensive unit tests with pytest
- **CI/CD**: GitHub Actions automated pipeline
- **Containerization**: Docker and Docker Compose for reproducible environments
- **Deployment**: Kubernetes manifests and Helm charts
- **Monitoring**: Prometheus and Grafana monitoring stack

## Dataset

**Heart Disease UCI Dataset**
- Source: UCI Machine Learning Repository
- Features: 14+ patient health indicators (age, sex, blood pressure, cholesterol, etc.)
- Target: Binary classification (presence/absence of heart disease)
- Link: https://archive.ics.uci.edu/ml/datasets/heart+disease

## Project Structure

```
.
├── .github/workflows/        # GitHub Actions CI/CD pipelines
├── src/
│   ├── data/                # Data loading and preprocessing
│   ├── models/              # Model training and evaluation
│   ├── features/            # Feature engineering
│   ├── api/                 # FastAPI application
│   └── monitoring.py        # Monitoring and logging
├── notebooks/
│   └── 01_EDA_and_Model_Training.ipynb
├── tests/                   # Unit tests
├── docker/                  # Dockerfile and docker-compose
├── k8s/                     # Kubernetes manifests
├── monitoring/              # Prometheus and Grafana configs
├── scripts/                 # Training and utility scripts
├── data/
│   ├── raw/                # Original dataset
│   └── processed/          # Processed datasets
├── models/
│   └── artifacts/          # Trained models and preprocessors
├── requirements.txt        # Python dependencies
├── setup.py               # Package setup
└── README.md              # This file
```

## Installation & Setup

### Prerequisites
- Python 3.9+
- Docker & Docker Compose (for containerization)
- Kubernetes cluster or Minikube (for K8s deployment)
- Git

### Local Development Setup

```bash
# 1. Clone the repository
git clone <repository-url>
cd heart-disease-prediction

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download dataset
python src/data/download_data.py

# 5. Run model training
python scripts/train_model.py

# 6. View MLflow UI
mlflow ui --backend-store-uri file:mlruns
# Access at http://localhost:5000
```

### Using Docker

```bash
# Build image
docker build -f docker/Dockerfile -t heart-disease-api:latest .

# Run container
docker-compose -f docker/docker-compose.yml up -d

# Access API at http://localhost:8000
# MLflow UI at http://localhost:5000
```

### Using Kubernetes

```bash
# Deploy to Kubernetes
kubectl apply -f k8s/deployment.yaml

# Deploy monitoring stack
kubectl apply -f monitoring/prometheus-grafana.yaml

# Port forward to access services
kubectl port-forward svc/heart-disease-api 8000:80 -n heart-disease-prediction
kubectl port-forward svc/grafana 3000:3000 -n heart-disease-prediction
```

## Usage

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test file
pytest tests/test_preprocessing.py -v
```

### Running Code Quality Checks

```bash
# Linting with flake8
flake8 src/ tests/

# Format check with black
black --check src/ tests/

# Auto-format with black
black src/ tests/
```

### Model Training

```bash
# Full training pipeline
python scripts/train_model.py

# Train within Jupyter notebook
jupyter notebook notebooks/01_EDA_and_Model_Training.ipynb
```

### API Usage

```bash
# Single prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 63, "sex": 1, "cp": 3, "trestbps": 145,
    "chol": 233, "fbs": 1, "restecg": 0, "thalach": 150,
    "exang": 0, "oldpeak": 2.3, "slope": 0, "ca": 0, "thal": 1
  }'

# Batch prediction
curl -X POST http://localhost:8000/batch_predict \
  -H "Content-Type: application/json" \
  -d '[{...}, {...}]'

# Health check
curl http://localhost:8000/health

# API documentation
# Open http://localhost:8000/docs in browser
```

## Model Development

### Models Implemented
1. **Logistic Regression**
   - Hyperparameters: C, solver
   - Training time: Fast
   - Interpretability: High

2. **Random Forest**
   - Hyperparameters: n_estimators, max_depth
   - Training time: Medium
   - Performance: High

### Model Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC

### Experiment Tracking

All experiments are logged to MLflow with:
- Model parameters
- Training metrics
- Model artifacts
- Evaluation plots

View experiments:
```bash
mlflow ui --backend-store-uri file:mlruns
```

## CI/CD Pipeline

### GitHub Actions Workflow

The pipeline automatically runs on push/PR:
1. **Linting**: flake8, black
2. **Testing**: pytest with coverage
3. **Docker Build**: Build and test container
4. **Model Training**: Automated training pipeline
5. **Artifact Upload**: Store results as artifacts

Workflow file: `.github/workflows/mlops_pipeline.yml`

### Local CI/CD Simulation

```bash
# Run linting
flake8 src/ tests/

# Run tests
pytest tests/ -v --cov=src

# Build Docker
docker build -f docker/Dockerfile -t heart-disease-api:latest .

# Run training
python scripts/train_model.py
```

## Deployment

### Docker Deployment

```bash
# Build image
docker build -f docker/Dockerfile -t heart-disease-api:latest .

# Run locally
docker run -p 8000:8000 heart-disease-api:latest

# Push to registry
docker tag heart-disease-api:latest <registry>/heart-disease-api:latest
docker push <registry>/heart-disease-api:latest
```

### Kubernetes Deployment

```bash
# Create namespace
kubectl create namespace heart-disease-prediction

# Deploy application
kubectl apply -f k8s/deployment.yaml

# Verify deployment
kubectl get pods -n heart-disease-prediction
kubectl get svc -n heart-disease-prediction

# Access via service
kubectl port-forward svc/heart-disease-api 8000:80 -n heart-disease-prediction
```

### Cloud Deployment (GKE, EKS, AKS)

```bash
# Push image to cloud registry
docker push <cloud-registry>/heart-disease-api:latest

# Update deployment image
kubectl set image deployment/heart-disease-api \
  api=<cloud-registry>/heart-disease-api:latest \
  -n heart-disease-prediction

# Check rollout status
kubectl rollout status deployment/heart-disease-api -n heart-disease-prediction
```

## Monitoring & Logging

### Prometheus Metrics

Metrics tracked:
- `http_requests_total`: Total HTTP requests
- `http_request_duration_seconds`: Request latency
- `predictions_total`: Total predictions made
- `prediction_time_seconds`: Prediction latency
- `active_requests`: Currently active requests
- `model_accuracy/precision/recall`: Model performance

### Grafana Dashboards

Dashboard URLs (after deployment):
- Grafana: http://localhost:3000
- Prometheus: http://localhost:9090

Deploy monitoring stack:
```bash
kubectl apply -f monitoring/prometheus-grafana.yaml
```

### Structured Logging

API logs are stored in JSON format in `logs/api.log` with fields:
- timestamp
- level
- message
- event_type
- additional metadata

## Features

### Data Pipeline
- Automated data download
- Missing value imputation
- Feature scaling (StandardScaler)
- Train-test split with stratification

### Model Pipeline
- Multiple model implementations
- Hyperparameter tuning (GridSearchCV)
- Cross-validation
- Comprehensive evaluation metrics

### API Features
- RESTful endpoints
- Input validation (Pydantic)
- Batch predictions
- Interactive documentation (Swagger/OpenAPI)
- Health checks

### Production Features
- Model versioning
- Reproducible preprocessing
- Docker containerization
- Kubernetes orchestration
- Automated testing
- Experiment tracking
- Comprehensive monitoring

## Performance Benchmarks

### Model Performance (Test Set)
| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | TBD | TBD | TBD | TBD | TBD |
| Random Forest | TBD | TBD | TBD | TBD | TBD |

(Populate after running training)

## API Endpoints

### Health & Status
- `GET /` - API status
- `GET /health` - Health check
- `GET /model-info` - Model information

### Predictions
- `POST /predict` - Single prediction
- `POST /batch_predict` - Batch predictions

### Documentation
- `GET /docs` - Swagger UI
- `GET /redoc` - ReDoc

## Environment Variables

```
PYTHONUNBUFFERED=1          # Python unbuffered output
MLFLOW_TRACKING_URI=file:mlruns  # MLflow tracking
LOG_LEVEL=INFO              # Logging level
ENVIRONMENT=production      # Environment
```

## Troubleshooting

### Model not loading
```bash
# Ensure model files exist
ls -la models/artifacts/

# Retrain models if missing
python scripts/train_model.py
```

### Docker build issues
```bash
# Clear Docker cache and rebuild
docker system prune
docker build --no-cache -f docker/Dockerfile -t heart-disease-api:latest .
```

### Kubernetes deployment issues
```bash
# Check pod logs
kubectl logs -f <pod-name> -n heart-disease-prediction

# Describe pod for events
kubectl describe pod <pod-name> -n heart-disease-prediction
```

## Contributing

1. Create feature branch: `git checkout -b feature/new-feature`
2. Make changes and test: `pytest tests/`
3. Format code: `black src/ tests/`
4. Push and create PR: `git push origin feature/new-feature`

## License

MIT License

## Contact & Support

For questions or issues:
- GitHub Issues: <repository-issues>
- Email: <contact-email>

## References

- MLflow Documentation: https://mlflow.org/docs/latest/
- FastAPI Documentation: https://fastapi.tiangolo.com/
- Kubernetes Documentation: https://kubernetes.io/docs/
- Docker Documentation: https://docs.docker.com/
- scikit-learn Documentation: https://scikit-learn.org/stable/

## Changelog

### Version 1.0.0 (2025)
- Initial release
- Data pipeline implementation
- Model training and evaluation
- MLflow integration
- API development
- Docker containerization
- Kubernetes deployment
- Monitoring and logging
- CI/CD pipeline
