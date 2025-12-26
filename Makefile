.PHONY: help install install-dev clean test lint format train docker-build docker-run k8s-deploy k8s-destroy docs serve-api

help:
	@echo "Heart Disease Prediction MLOps Project - Available Commands"
	@echo "=========================================================="
	@echo ""
	@echo "Setup:"
	@echo "  make install          - Install production dependencies"
	@echo "  make install-dev      - Install development dependencies"
	@echo "  make clean           - Clean temporary files and caches"
	@echo ""
	@echo "Testing & Code Quality:"
	@echo "  make test            - Run unit tests"
	@echo "  make test-cov        - Run tests with coverage report"
	@echo "  make lint            - Run code linting (flake8)"
	@echo "  make format          - Format code with black"
	@echo "  make format-check    - Check code formatting without changes"
	@echo ""
	@echo "Machine Learning:"
	@echo "  make train           - Run model training pipeline"
	@echo "  make train-notebook  - Open Jupyter notebook for training"
	@echo "  make mlflow-ui       - Start MLflow UI"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-build    - Build Docker image"
	@echo "  make docker-run      - Run Docker container locally"
	@echo "  make docker-stop     - Stop Docker container"
	@echo "  make docker-compose  - Run full stack with docker-compose"
	@echo ""
	@echo "Kubernetes:"
	@echo "  make k8s-deploy      - Deploy to Kubernetes"
	@echo "  make k8s-destroy     - Remove Kubernetes deployment"
	@echo "  make k8s-logs        - View pod logs"
	@echo ""
	@echo "API:"
	@echo "  make serve-api       - Start API server locally"
	@echo "  make test-api        - Test API endpoints"
	@echo ""

install:
	pip install --upgrade pip
	pip install -r requirements.txt

install-dev:
	pip install --upgrade pip
	pip install -r requirements.txt
	pip install pytest pytest-cov flake8 black jupyter

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".coverage" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	rm -f .coverage

test:
	pytest tests/ -v

test-cov:
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term-missing
	@echo "Coverage report generated in htmlcov/index.html"

lint:
	flake8 src/ tests/ --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics

format:
	black src/ tests/ notebooks/ scripts/

format-check:
	black --check src/ tests/ notebooks/ scripts/

train:
	python scripts/train_model.py

train-notebook:
	jupyter notebook notebooks/01_EDA_and_Model_Training.ipynb

mlflow-ui:
	mlflow ui --backend-store-uri file:mlruns

docker-build:
	docker build -f docker/Dockerfile -t heart-disease-api:latest .

docker-run: docker-build
	docker run -d -p 8000:8000 \
		-v $$(pwd)/logs:/app/logs \
		-v $$(pwd)/models:/app/models \
		--name heart-disease-api \
		heart-disease-api:latest
	@echo "API running at http://localhost:8000"
	@echo "API Docs at http://localhost:8000/docs"

docker-stop:
	docker stop heart-disease-api 2>/dev/null || true
	docker rm heart-disease-api 2>/dev/null || true

docker-compose: docker-build
	docker-compose -f docker/docker-compose.yml up -d
	@echo "Services running:"
	@echo "  API: http://localhost:8000"
	@echo "  MLflow: http://localhost:5000"
	@echo "  API Docs: http://localhost:8000/docs"

k8s-deploy:
	kubectl apply -f k8s/deployment.yaml
	kubectl apply -f monitoring/prometheus-grafana.yaml
	@echo "Deployment complete. Services deployed to: heart-disease-prediction namespace"

k8s-destroy:
	kubectl delete -f k8s/deployment.yaml
	kubectl delete -f monitoring/prometheus-grafana.yaml

k8s-logs:
	@POD=$$(kubectl get pods -n heart-disease-prediction | grep heart-disease-api | head -1 | awk '{print $$1}'); \
	if [ -z "$$POD" ]; then \
		echo "No running pods found"; \
	else \
		kubectl logs -f $$POD -n heart-disease-prediction; \
	fi

serve-api:
	python -m uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload

test-api:
	bash scripts/test_api.sh
