# 📚 Project Index & Navigation Guide

## Welcome to Heart Disease Prediction - MLOps End-to-End Pipeline

This is a complete, production-ready MLOps solution demonstrating industry best practices for machine learning at scale.

---

## 🎯 Getting Started

**New to this project?** Start here:

1. **[README.md](README.md)** - Project overview and main documentation
2. **[INSTALLATION.md](INSTALLATION.md)** - Step-by-step setup guide
3. **[QUICK_REFERENCE.md](QUICK_REFERENCE.md)** - Common commands

**5-Minute Quick Start:**
```bash
git clone <repo>
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python scripts/train_model.py
python -m uvicorn src.api.app:app --port 8000
# Open http://localhost:8000/docs
```

---

## 📖 Documentation

### Overview & Understanding
- **[README.md](README.md)** - Complete project overview
- **[PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)** - Detailed summary of all components
- **[DELIVERABLES.md](DELIVERABLES.md)** - Assignment deliverables checklist

### Setup & Installation
- **[INSTALLATION.md](INSTALLATION.md)** - Comprehensive installation guide
- **[QUICK_REFERENCE.md](QUICK_REFERENCE.md)** - Commands and quick fixes

### Deployment & Operations
- **[DEPLOYMENT.md](DEPLOYMENT.md)** - Cloud and local deployment procedures
- **Makefile** - Automation commands (run `make help`)

---

## 💻 Code Organization

### Source Code (`src/`)
- **`data/`** - Data loading and preprocessing
  - `download_data.py` - Download UCI dataset
  - `preprocessing.py` - Data cleaning and preprocessing
  
- **`models/`** - Model training and evaluation
  - `train.py` - Logistic Regression and Random Forest training
  
- **`api/`** - FastAPI application
  - `app.py` - Complete REST API with endpoints
  
- **`config.py`** - Configuration management
- **`monitoring.py`** - Logging and Prometheus metrics

### Notebooks (`notebooks/`)
- **`01_EDA_and_Model_Training.ipynb`** - Complete interactive notebook covering:
  - Data acquisition and exploration
  - EDA with visualizations
  - Model training and evaluation
  - MLflow integration
  - Deployment instructions

### Tests (`tests/`)
- **`test_preprocessing.py`** - Data preprocessing unit tests
- **`test_models.py`** - Model training and evaluation tests
- **`test_api.py`** - API endpoint tests
- **`conftest.py`** - Test configuration and fixtures

### Scripts (`scripts/`)
- **`train_model.py`** - Automated training pipeline
- **`test_api.sh`** - API testing script

---

## 🐳 Docker & Kubernetes

### Docker (`docker/`)
- **`Dockerfile`** - Production Docker image
- **`docker-compose.yml`** - Full stack (API + MLflow)
- **`run_container.sh`** - Container launch script

### Kubernetes (`k8s/`)
- **`deployment.yaml`** - Deployment, Service, HPA, Namespace
- **`ingress.yaml`** - Ingress routing configuration
- **`configmap.yaml`** - Configuration and secrets

### Monitoring (`monitoring/`)
- **`prometheus-grafana.yaml`** - Prometheus + Grafana deployment

---

## 📊 Data & Models

### Data (`data/`)
```
data/
├── raw/          # Original dataset (populated by download script)
└── processed/    # Processed datasets and comparisons
```

### Models (`models/`)
```
models/
└── artifacts/    # Trained models:
    ├── preprocessor.pkl
    ├── logistic_regression_model.pkl
    ├── random_forest_model.pkl
    └── prediction_pipeline.pkl
```

### Screenshots (`screenshots/`)
- Deployment screenshots and visualizations

---

## 🔄 Workflows & Automation

### CI/CD (`.github/workflows/`)
- **`mlops_pipeline.yml`** - GitHub Actions automation:
  - Linting and code quality
  - Unit testing
  - Docker build
  - Model training
  - Artifact upload

### Makefile Commands
```bash
make help           # Show all available commands
make install        # Install dependencies
make test           # Run unit tests
make lint           # Code linting
make format         # Format code with black
make train          # Train models
make serve-api      # Start API locally
make docker-build   # Build Docker image
make k8s-deploy     # Deploy to Kubernetes
```

---

## 🚀 Quick Navigation

### I want to...

**Run the project locally:**
→ See [INSTALLATION.md](INSTALLATION.md)

**Deploy to production:**
→ See [DEPLOYMENT.md](DEPLOYMENT.md)

**Understand the models:**
→ See `notebooks/01_EDA_and_Model_Training.ipynb`

**Test the API:**
→ See [QUICK_REFERENCE.md](QUICK_REFERENCE.md) "API Endpoints" section

**View test coverage:**
→ Run: `make test-cov` then open `htmlcov/index.html`

**Monitor application:**
→ See [DEPLOYMENT.md](DEPLOYMENT.md) "Monitoring Stack" section

**Check experiment results:**
→ Run: `mlflow ui --backend-store-uri file:mlruns` then open `http://localhost:5000`

**Add new features:**
→ See [QUICK_REFERENCE.md](QUICK_REFERENCE.md) "Daily Development" workflow

---

## 📋 Key Files by Purpose

### Data Management
- `src/data/download_data.py` - Download script
- `src/data/preprocessing.py` - Preprocessing pipeline
- `notebooks/01_EDA_and_Model_Training.ipynb` - EDA notebook

### Model Development
- `src/models/train.py` - Training code
- `scripts/train_model.py` - Training script
- `notebooks/01_EDA_and_Model_Training.ipynb` - Interactive training

### API & Serving
- `src/api/app.py` - FastAPI application
- `tests/test_api.py` - API tests
- `docker/Dockerfile` - API container

### Testing
- `tests/test_preprocessing.py` - Data tests
- `tests/test_models.py` - Model tests
- `tests/test_api.py` - Integration tests

### Infrastructure
- `.github/workflows/mlops_pipeline.yml` - CI/CD
- `docker/docker-compose.yml` - Local development
- `k8s/deployment.yaml` - Production K8s
- `monitoring/prometheus-grafana.yaml` - Monitoring

---

## 📞 Support & Troubleshooting

### Common Issues
1. **Cannot download dataset**
   → See [INSTALLATION.md](INSTALLATION.md) "Troubleshooting"

2. **Docker build fails**
   → Run: `docker system prune` then rebuild

3. **Port already in use**
   → Use different port: `python -m uvicorn src.api.app:app --port 8001`

4. **Tests failing**
   → Run: `pytest tests/ -v -s` for detailed output

### Getting Help
- Check [QUICK_REFERENCE.md](QUICK_REFERENCE.md)
- Review [INSTALLATION.md](INSTALLATION.md) troubleshooting
- See [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md) for detailed explanations
- Check inline code comments and docstrings

---

## 🎓 Learning Path

### For Beginners
1. Read [README.md](README.md)
2. Follow [INSTALLATION.md](INSTALLATION.md)
3. Run local setup
4. Review `notebooks/01_EDA_and_Model_Training.ipynb`
5. Run basic tests: `pytest tests/ -v`

### For ML Engineers
1. Study `src/models/train.py`
2. Review model configurations in `src/config.py`
3. Check hyperparameter tuning in `scripts/train_model.py`
4. Run: `python scripts/train_model.py`

### For DevOps/Platform Engineers
1. Review `docker/Dockerfile` and `docker-compose.yml`
2. Study `k8s/deployment.yaml`
3. Deploy locally: `docker-compose up -d`
4. Test Kubernetes: `kubectl apply -f k8s/deployment.yaml`

### For Full-Stack Development
1. Understand entire [README.md](README.md)
2. Follow [INSTALLATION.md](INSTALLATION.md)
3. Review all components
4. Run complete pipeline: `make install`, `make train`, `make serve-api`

---

## 🔗 External Resources

- **scikit-learn**: https://scikit-learn.org/
- **MLflow**: https://mlflow.org/docs/
- **FastAPI**: https://fastapi.tiangolo.com/
- **Docker**: https://docs.docker.com/
- **Kubernetes**: https://kubernetes.io/docs/
- **Prometheus**: https://prometheus.io/docs/
- **Grafana**: https://grafana.com/docs/

---

## 📈 Project Metrics at a Glance

| Metric | Value |
|--------|-------|
| Code Files | 15+ |
| Test Cases | 26+ |
| Test Coverage | >80% |
| Documentation | 6 files |
| Models Trained | 2 |
| API Endpoints | 6 |
| K8s Manifests | 3 |
| CI/CD Workflows | 1 |
| Support Scripts | 2+ |
| Total Lines of Code | 4000+ |

---

## ✅ Verification Checklist

Before using this project, verify:

- [ ] Git repository cloned
- [ ] Python 3.9+ installed
- [ ] Virtual environment created
- [ ] Dependencies installed: `pip install -r requirements.txt`
- [ ] Dataset downloaded: `python src/data/download_data.py`
- [ ] Tests passing: `pytest tests/ -v`
- [ ] API starts: `python -m uvicorn src.api.app:app --port 8000`

---

## 🎯 Next Steps

1. **Immediate**: Complete [INSTALLATION.md](INSTALLATION.md) setup
2. **Short-term**: Run `notebooks/01_EDA_and_Model_Training.ipynb`
3. **Medium-term**: Test API: `make test-api`
4. **Long-term**: Deploy to cloud using [DEPLOYMENT.md](DEPLOYMENT.md)

---

## 📞 Contact & Questions

- **Documentation**: See all `.md` files
- **Code Examples**: Check notebooks and scripts
- **Troubleshooting**: See [INSTALLATION.md](INSTALLATION.md) and [QUICK_REFERENCE.md](QUICK_REFERENCE.md)
- **Issues**: Review [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)

---

## 🎉 Ready to Get Started?

**Start with:** [INSTALLATION.md](INSTALLATION.md)

**Questions?** Check [QUICK_REFERENCE.md](QUICK_REFERENCE.md) or [README.md](README.md)

**Need details?** See [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)

---

**Project Status**: ✅ Production Ready  
**Last Updated**: December 25, 2024  
**Version**: 1.0.0

Happy coding! 🚀
