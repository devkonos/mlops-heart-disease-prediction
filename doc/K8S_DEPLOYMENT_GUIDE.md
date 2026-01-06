# Local Kubernetes Deployment Guide

## Overview
This guide covers deploying the Heart Disease Prediction MLOps application to a local Kubernetes cluster using Minikube or Docker Desktop Kubernetes.

## Prerequisites

### System Requirements
- 4GB RAM minimum
- 20GB disk space
- Docker installed

### Option A: Docker Desktop Kubernetes (Recommended)
1. Install [Docker Desktop](https://www.docker.com/products/docker-desktop)
2. Enable Kubernetes in Settings:
   - Docker Desktop → Settings → Kubernetes → Enable Kubernetes
   - Wait 2-3 minutes for cluster to start

### Option B: Minikube
```bash
# Install Minikube (Windows)
choco install minikube

# Start cluster with sufficient resources
minikube start --cpus=4 --memory=4096 --vm-driver=hyperv

# Set Docker environment to Minikube
& minikube -p minikube docker-env | Invoke-Expression
```

## Deployment Steps

### Step 1: Build Docker Images

```bash
cd d:\Workspace\BITS MTECH\Sem3\MLOps\Assignment\mlops-assign-1

# Build API image
docker build -f docker/Dockerfile -t heart-disease-api:latest .

# Build Dashboard image
docker build -f docker/Dockerfile.dashboard -t heart-disease-dashboard:latest .

# Verify images
docker images | grep heart-disease
```

**For Minikube**, load images into Minikube's Docker:
```bash
minikube image load heart-disease-api:latest
minikube image load heart-disease-dashboard:latest
```

### Step 2: Verify Kubernetes Connection

```bash
# Check cluster info
kubectl cluster-info

# Get nodes
kubectl get nodes

# Create namespace (optional)
kubectl create namespace mlops-app
```

### Step 3: Deploy API

```bash
# Apply API deployment
kubectl apply -f k8s/deployment.yaml

# Verify deployment
kubectl get deployments
kubectl get pods
kubectl get svc

# Check pod status
kubectl describe pod <pod-name>

# View logs
kubectl logs <pod-name>
```

### Step 4: Deploy Streamlit Dashboard

```bash
# Apply dashboard deployment
kubectl apply -f k8s/dashboard-deployment.yaml

# Verify dashboard deployment
kubectl get deployments
kubectl get pods -l app=streamlit-dashboard
kubectl get svc streamlit-dashboard
```

### Step 5: Access Services

#### For Docker Desktop Kubernetes:
- **API**: http://localhost:8000
- **Dashboard**: http://localhost:8501
- **API Docs**: http://localhost:8000/docs

#### For Minikube:
```bash
# Get service URL
minikube service heart-disease-api
minikube service streamlit-dashboard

# OR use port forwarding
kubectl port-forward svc/heart-disease-api 8000:8000
kubectl port-forward svc/streamlit-dashboard 8501:8501
```

### Step 6: Monitor Deployment

```bash
# Watch pods (real-time)
kubectl get pods --watch

# Check events
kubectl describe deployment heart-disease-api
kubectl describe deployment streamlit-dashboard

# View metrics (if metrics-server installed)
kubectl top nodes
kubectl top pods
```

### Step 7: Scale Deployments

```bash
# Scale API to 3 replicas
kubectl scale deployment heart-disease-api --replicas=3

# Scale dashboard to 2 replicas
kubectl scale deployment streamlit-dashboard --replicas=2

# Monitor
kubectl get pods --watch
```

## Testing

### Test API Health
```bash
curl http://localhost:8000/health
```

### Test Prediction
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"age":50,"sex":1,"cp":3,"trestbps":120,"chol":240,"fbs":0,"restecg":0,"thalach":150,"exang":0,"oldpeak":1.0,"slope":2,"ca":0,"thal":2}'
```

### Test Dashboard
Open browser: http://localhost:8501

## Monitoring

### View Pod Logs
```bash
# API logs
kubectl logs -f deployment/heart-disease-api

# Dashboard logs
kubectl logs -f deployment/streamlit-dashboard

# Previous logs
kubectl logs -p deployment/heart-disease-api
```

### Describe Resources
```bash
kubectl describe deployment heart-disease-api
kubectl describe service heart-disease-api
kubectl describe pod <pod-name>
```

### Check Events
```bash
kubectl get events --sort-by='.lastTimestamp'
```

## Troubleshooting

### Issue: Pods not starting
```bash
# Check pod status
kubectl describe pod <pod-name>

# Check events
kubectl get events --sort-by='.lastTimestamp'

# Check logs
kubectl logs <pod-name>
```

### Issue: Image not found
```bash
# For Docker Desktop: Rebuild images
docker build -f docker/Dockerfile -t heart-disease-api:latest .

# For Minikube: Rebuild and load
docker build -f docker/Dockerfile -t heart-disease-api:latest .
minikube image load heart-disease-api:latest
```

### Issue: Cannot connect to service
```bash
# Check service status
kubectl describe service heart-disease-api

# Test connectivity
kubectl run -it --rm debug --image=busybox --restart=Never -- sh
# Inside pod: wget -O- http://heart-disease-api:8000/health
```

### Issue: Port already in use
```bash
# For Docker Desktop: Service is exposed on localhost directly
# For Minikube: Use different port forwarding
kubectl port-forward svc/heart-disease-api 9000:8000
```

## Clean Up

```bash
# Delete all deployments
kubectl delete deployment heart-disease-api
kubectl delete deployment streamlit-dashboard

# Delete services
kubectl delete svc heart-disease-api
kubectl delete svc streamlit-dashboard

# Delete namespace (if created)
kubectl delete namespace mlops-app
```

## Advanced: Using Helm (Optional)

Create `chart/values.yaml`:
```yaml
api:
  image: heart-disease-api:latest
  replicas: 2
  port: 8000

dashboard:
  image: heart-disease-dashboard:latest
  replicas: 1
  port: 8501
```

Deploy with Helm:
```bash
helm install mlops-app ./chart
```

## Performance Tuning

### Resource Optimization
```yaml
resources:
  requests:
    memory: "256Mi"
    cpu: "250m"
  limits:
    memory: "512Mi"
    cpu: "500m"
```

### Auto-scaling
```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: heart-disease-api-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: heart-disease-api
  minReplicas: 1
  maxReplicas: 5
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 80
```

## Next Steps

1. **Monitoring**: Setup Prometheus + Grafana
2. **Ingress**: Configure Nginx ingress for external access
3. **CI/CD**: Automate deployments with GitHub Actions
4. **Storage**: Add persistent volumes for models and data
5. **Secrets**: Manage sensitive data with Kubernetes Secrets

## References
- [Kubernetes Documentation](https://kubernetes.io/docs/)
- [Docker Desktop Kubernetes](https://docs.docker.com/desktop/kubernetes/)
- [Minikube Documentation](https://minikube.sigs.k8s.io/)
