#!/bin/bash

# API Testing Script for Heart Disease Prediction

BASE_URL="http://localhost:8000"

echo "================================================"
echo "Heart Disease Prediction API - Testing Script"
echo "================================================"

# Test 1: Health Check
echo -e "\n[Test 1] Health Check"
curl -s -X GET "$BASE_URL/health" | python -m json.tool
echo ""

# Test 2: Root Endpoint
echo -e "\n[Test 2] Root Endpoint"
curl -s -X GET "$BASE_URL/" | python -m json.tool
echo ""

# Test 3: Model Info
echo -e "\n[Test 3] Model Information"
curl -s -X GET "$BASE_URL/model-info" | python -m json.tool
echo ""

# Test 4: Single Prediction
echo -e "\n[Test 4] Single Prediction (Disease Case)"
curl -s -X POST "$BASE_URL/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "age": 63,
    "sex": 1,
    "cp": 3,
    "trestbps": 145,
    "chol": 233,
    "fbs": 1,
    "restecg": 0,
    "thalach": 150,
    "exang": 0,
    "oldpeak": 2.3,
    "slope": 0,
    "ca": 0,
    "thal": 1
  }' | python -m json.tool
echo ""

# Test 5: Single Prediction (Healthy Case)
echo -e "\n[Test 5] Single Prediction (Healthy Case)"
curl -s -X POST "$BASE_URL/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "age": 35,
    "sex": 0,
    "cp": 1,
    "trestbps": 110,
    "chol": 150,
    "fbs": 0,
    "restecg": 1,
    "thalach": 130,
    "exang": 0,
    "oldpeak": 0.2,
    "slope": 1,
    "ca": 0,
    "thal": 0
  }' | python -m json.tool
echo ""

# Test 6: Batch Prediction
echo -e "\n[Test 6] Batch Prediction"
curl -s -X POST "$BASE_URL/batch_predict" \
  -H "Content-Type: application/json" \
  -d '[
    {
      "age": 63,
      "sex": 1,
      "cp": 3,
      "trestbps": 145,
      "chol": 233,
      "fbs": 1,
      "restecg": 0,
      "thalach": 150,
      "exang": 0,
      "oldpeak": 2.3,
      "slope": 0,
      "ca": 0,
      "thal": 1
    },
    {
      "age": 35,
      "sex": 0,
      "cp": 1,
      "trestbps": 110,
      "chol": 150,
      "fbs": 0,
      "restecg": 1,
      "thalach": 130,
      "exang": 0,
      "oldpeak": 0.2,
      "slope": 1,
      "ca": 0,
      "thal": 0
    }
  ]' | python -m json.tool
echo ""

# Test 7: Invalid Input
echo -e "\n[Test 7] Invalid Input (Missing field)"
curl -s -X POST "$BASE_URL/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "age": 63,
    "sex": 1
  }' | python -m json.tool
echo ""

echo -e "\n================================================"
echo "API Testing Complete"
echo "================================================"
echo -e "\nAPI Documentation: $BASE_URL/docs"
echo -e "Alternative Docs: $BASE_URL/redoc"
