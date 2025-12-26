# Quick Reference Guide

## Common Commands

### Setup & Installation
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Install dev dependencies
pip install -r requirements.txt pytest pytest-cov flake8 black jupyter
```

### Data & Training
```bash
# Download dataset
python src/data/download_data.py

# Train models
python scripts/train_model.py

# Open Jupyter notebook
jupyter notebook notebooks/01_EDA_and_Model_Training.ipynb

# View MLflow UI
mlflow ui --backend-store-uri file:mlruns
```

### Testing & Code Quality
```bash
# Run all tests
pytest tests/ -v

# Run tests with coverage
pytest tests/ -v --cov=src --cov-report=html

# Lint code
flake8 src/ tests/

# Format code
black src/ tests/

# Check formatting
black --check src/ tests/
```

### API & Development
```bash
# Start API locally
python -m uvicorn src.api.app:app --port 8000

# Start API with auto-reload
python -m uvicorn src.api.app:app --port 8000 --reload

# Test API endpoints
bash scripts/test_api.sh

# Access API docs
# http://localhost:8000/docs
```

### Docker
```bash
# Build image
docker build -f docker/Dockerfile -t heart-disease-api:latest .

# Run container
docker run -p 8000:8000 heart-disease-api:latest

# Stop container
docker stop heart-disease-api

# Use docker-compose
docker-compose -f docker/docker-compose.yml up -d
docker-compose -f docker/docker-compose.yml down
```

### Kubernetes
```bash
# Deploy application
kubectl apply -f k8s/deployment.yaml

# Deploy monitoring stack
kubectl apply -f monitoring/prometheus-grafana.yaml

# Check pods
kubectl get pods -n heart-disease-prediction

# Check services
kubectl get svc -n heart-disease-prediction

# View pod logs
kubectl logs -f <pod-name> -n heart-disease-prediction

# Port forward
kubectl port-forward svc/heart-disease-api 8000:80 -n heart-disease-prediction

# Scale deployment
kubectl scale deployment heart-disease-api --replicas=5 -n heart-disease-prediction

# Delete deployment
kubectl delete -f k8s/deployment.yaml
```

### Makefile Commands
```bash
# Show help
make help

# Install dependencies
make install
make install-dev

# Clean project
make clean

# Testing
make test
make test-cov
make lint
make format

# Machine Learning
make train
make train-notebook
make mlflow-ui

# Docker
make docker-build
make docker-run
make docker-stop
make docker-compose

# Kubernetes
make k8s-deploy
make k8s-destroy
make k8s-logs

# API
make serve-api
make test-api
```

---

## API Endpoints

### Health Checks
```bash
curl http://localhost:8000/
curl http://localhost:8000/health
```

### Model Information
```bash
curl http://localhost:8000/model-info
```

### Single Prediction
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 63, "sex": 1, "cp": 3, "trestbps": 145,
    "chol": 233, "fbs": 1, "restecg": 0, "thalach": 150,
    "exang": 0, "oldpeak": 2.3, "slope": 0, "ca": 0, "thal": 1
  }'
```

### Batch Prediction
```bash
curl -X POST http://localhost:8000/batch_predict \
  -H "Content-Type: application/json" \
  -d '[{...}, {...}]'
```

### API Documentation
```
Swagger UI: http://localhost:8000/docs
ReDoc: http://localhost:8000/redoc
```

---

## Key File Locations

| Component | Files |
|-----------|-------|
| **Data** | `src/data/download_data.py`, `src/data/preprocessing.py` |
| **Models** | `src/models/train.py` |
| **API** | `src/api/app.py` |
| **Tests** | `tests/test_*.py` |
| **Training** | `scripts/train_model.py` |
| **CI/CD** | `.github/workflows/mlops_pipeline.yml` |
| **Docker** | `docker/Dockerfile`, `docker/docker-compose.yml` |
| **Kubernetes** | `k8s/deployment.yaml`, `k8s/ingress.yaml` |
| **Monitoring** | `monitoring/prometheus-grafana.yaml` |
| **Config** | `src/config.py` |

---

## Troubleshooting Quick Fixes

### Cannot download dataset
```bash
# Manually place data/raw/heart_disease.csv
# Then run preprocessing
python src/data/preprocessing.py
```

### Port already in use (8000)
```bash
# Use different port
python -m uvicorn src.api.app:app --port 8001

# Or kill process
# Linux/Mac: lsof -i :8000 | kill -9 <PID>
# Windows: netstat -ano | findstr :8000
```

### Docker build fails
```bash
# Clear cache
docker system prune

# Rebuild
docker build -f docker/Dockerfile -t heart-disease-api:latest --no-cache .
```

### Model not found
```bash
# Retrain models
python scripts/train_model.py

# Verify
ls models/artifacts/
```

### Tests fail
```bash
# Ensure dependencies installed
pip install -r requirements.txt

# Run with verbose output
pytest tests/ -v -s
```

---

## Service URLs (Local Development)

| Service | URL | Default Credentials |
|---------|-----|-------------------|
| API | http://localhost:8000 | - |
| API Docs | http://localhost:8000/docs | - |
| MLflow | http://localhost:5000 | - |
| Prometheus | http://localhost:9090 | - |
| Grafana | http://localhost:3000 | admin/admin |

---

## Feature Flags & Configuration

### Environment Variables
```bash
# Development
export ENVIRONMENT=development
export LOG_LEVEL=DEBUG

# Production
export ENVIRONMENT=production
export LOG_LEVEL=INFO

# MLflow
export MLFLOW_TRACKING_URI=file:mlruns

# API
export API_PORT=8000
export API_HOST=0.0.0.0
```

### Model Configuration (src/config.py)
```python
# Adjust hyperparameters
MODEL_CONFIG = {
    'logistic_regression': {
        'C_values': [0.1, 1.0, 10.0],  # Regularization strength
        'solvers': ['lbfgs', 'liblinear'],
    },
    'random_forest': {
        'n_estimators': [50, 100, 200],  # Number of trees
        'max_depths': [5, 10, 15],  # Tree depth
    }
}
```

---

## Performance Tips

### Optimize Training
- Increase `n_jobs=-1` in GridSearchCV
- Reduce cross-validation folds if time-constrained
- Use smaller hyperparameter grid for quick iterations

### Optimize API
- Enable caching for repeated predictions
- Use batch endpoints for multiple predictions
- Monitor metrics on Grafana dashboard

### Optimize Kubernetes
- Adjust CPU/memory requests based on actual usage
- Configure HPA based on your traffic patterns
- Use PersistentVolumes for model storage

---

## Common Workflows

### Daily Development
```bash
# 1. Start environment
source venv/bin/activate
make install-dev

# 2. Make changes
# (edit code)

# 3. Test
make test
make lint

# 4. Train (if model changes)
make train

# 5. Start API
make serve-api

# 6. Test API
make test-api
```

### Before Committing
```bash
make format
make lint
make test
pytest tests/ --cov=src
```

### Deployment Workflow
```bash
# Local testing
docker-compose -f docker/docker-compose.yml up -d
bash scripts/test_api.sh

# Push to registry
docker tag heart-disease-api:latest <registry>/heart-disease-api:v1
docker push <registry>/heart-disease-api:v1

# Deploy to K8s
kubectl set image deployment/heart-disease-api \
  api=<registry>/heart-disease-api:v1 \
  -n heart-disease-prediction

# Verify
kubectl rollout status deployment/heart-disease-api -n heart-disease-prediction
```

---

## Learning Resources

- **MLOps**: https://ml-ops.systems/
- **scikit-learn**: https://scikit-learn.org/
- **MLflow**: https://mlflow.org/docs/
- **FastAPI**: https://fastapi.tiangolo.com/
- **Docker**: https://docs.docker.com/
- **Kubernetes**: https://kubernetes.io/docs/
- **Prometheus**: https://prometheus.io/docs/
- **Grafana**: https://grafana.com/docs/

---

## Checklist for Deployment

- [ ] All tests passing
- [ ] Code formatted with black
- [ ] No linting errors
- [ ] Models trained and validated
- [ ] Docker image builds successfully
- [ ] Container runs locally without errors
- [ ] API endpoints respond correctly
- [ ] Kubernetes manifests validated
- [ ] Monitoring stack deployed
- [ ] Documentation updated
- [ ] Git changes committed and pushed

---

## Support

- **Issues**: Check GitHub Issues or PROJECT_SUMMARY.md
- **Questions**: Review README.md and documentation files
- **Bugs**: Submit detailed issue with logs and reproduction steps

---

**Last Updated**: December 25, 2024  
**Project Status**: ✅ Production Ready
