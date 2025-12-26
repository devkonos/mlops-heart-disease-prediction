#!/bin/bash

# Build Docker image
echo "Building Docker image..."
docker build -f docker/Dockerfile -t heart-disease-api:latest .

# Run container
echo "Running container..."
docker run -d \
  -p 8000:8000 \
  --name heart-disease-api \
  -v $(pwd)/logs:/app/logs \
  -v $(pwd)/models:/app/models \
  heart-disease-api:latest

# Wait for container to start
sleep 3

# Test the API
echo "Testing API health..."
curl -s http://localhost:8000/health | python -m json.tool

echo "API is running on http://localhost:8000"
echo "API Documentation: http://localhost:8000/docs"
