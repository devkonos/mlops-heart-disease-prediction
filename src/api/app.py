"""
FastAPI application for Heart Disease Prediction model serving
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import joblib
import numpy as np
import logging
import json
from datetime import datetime
import os
from prometheus_client import Counter, Gauge, Histogram, generate_latest
from fastapi.responses import Response

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Heart Disease Prediction API",
    description="ML Model API for predicting heart disease risk",
    version="1.0.0"
)

# Prometheus Metrics
REQUEST_COUNT = Counter(
    'api_requests_total',
    'Total API requests',
    ['method', 'endpoint', 'status']
)

PREDICTION_COUNT = Counter(
    'predictions_total',
    'Total predictions made',
    ['prediction_class']
)

REQUEST_LATENCY = Histogram(
    'request_latency_seconds',
    'Request latency in seconds',
    ['endpoint']
)

MODEL_ACCURACY = Gauge(
    'model_accuracy',
    'Model accuracy metric'
)

PREDICTIONS_POSITIVE = Gauge(
    'predictions_positive_total',
    'Total positive predictions'
)

PREDICTIONS_NEGATIVE = Gauge(
    'predictions_negative_total',
    'Total negative predictions'
)

# Load model and preprocessor
MODEL_PATH = "models/artifacts/random_forest_model.pkl"
PREPROCESSOR_PATH = "models/artifacts/preprocessor.pkl"

# Try to load models, use None if not available (for testing)
try:
    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    logger.info("Model and preprocessor loaded successfully")
except FileNotFoundError as e:
    logger.warning(f"Model files not found: {e}. Running in demo mode.")
    model = None
    preprocessor = None


class PredictionInput(BaseModel):
    """Input model for prediction request"""
    age: float
    sex: float
    cp: float
    trestbps: float
    chol: float
    fbs: float
    restecg: float
    thalach: float
    exang: float
    oldpeak: float
    slope: float
    ca: float
    thal: float
    
    class Config:
        example = {
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
        }


class PredictionOutput(BaseModel):
    """Output model for prediction response"""
    prediction: int
    confidence: float
    prediction_label: str
    probabilities: Dict[str, float]
    timestamp: str


@app.get("/", tags=["Health"])
async def root():
    """Root endpoint for API health check"""
    return {
        "message": "Heart Disease Prediction API",
        "status": "healthy",
        "version": "1.0.0"
    }


@app.get("/health", tags=["Health"])
async def health():
    """Health check endpoint"""
    REQUEST_COUNT.labels(method="GET", endpoint="/health", status=200).inc()
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }


@app.post("/predict", response_model=PredictionOutput, tags=["Predictions"])
async def predict(input_data: PredictionInput):
    """
    Make prediction for heart disease risk
    
    Returns:
        Prediction with confidence score and probabilities
    """
    
    if model is None or preprocessor is None:
        logger.error("Model not loaded")
        REQUEST_COUNT.labels(method="POST", endpoint="/predict", status=500).inc()
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Convert input to numpy array
        input_dict = input_data.dict()
        X = np.array([list(input_dict.values())])
        
        # Preprocess
        X_preprocessed = preprocessor.transform(X)
        
        # Predict
        prediction = model.predict(X_preprocessed)[0]
        probabilities = model.predict_proba(X_preprocessed)[0]
        confidence = float(max(probabilities))
        
        # Update metrics
        REQUEST_COUNT.labels(method="POST", endpoint="/predict", status=200).inc()
        PREDICTION_COUNT.labels(prediction_class="positive" if prediction == 1 else "negative").inc()
        if prediction == 1:
            PREDICTIONS_POSITIVE.inc()
        else:
            PREDICTIONS_NEGATIVE.inc()
        
        # Log prediction
        logger.info(f"Prediction made: {prediction} with confidence {confidence:.4f}")
        
        # Format response
        response = PredictionOutput(
            prediction=int(prediction),
            confidence=confidence,
            prediction_label="Disease Present" if prediction == 1 else "No Disease",
            probabilities={
                "no_disease": float(probabilities[0]),
                "disease_present": float(probabilities[1])
            },
            timestamp=datetime.now().isoformat()
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        REQUEST_COUNT.labels(method="POST", endpoint="/predict", status=400).inc()
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")


@app.post("/batch_predict", tags=["Predictions"])
async def batch_predict(inputs: List[PredictionInput]):
    """
    Make batch predictions
    
    Returns:
        List of predictions
    """
    
    # Validate input
    if not inputs or len(inputs) == 0:
        REQUEST_COUNT.labels(method="POST", endpoint="/batch_predict", status=400).inc()
        raise HTTPException(status_code=400, detail="Batch input cannot be empty")
    
    if model is None or preprocessor is None:
        logger.error("Model not loaded")
        REQUEST_COUNT.labels(method="POST", endpoint="/batch_predict", status=500).inc()
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Convert inputs to numpy array
        X_list = [list(inp.dict().values()) for inp in inputs]
        X = np.array(X_list)
        
        # Preprocess
        X_preprocessed = preprocessor.transform(X)
        
        # Predict
        predictions = model.predict(X_preprocessed)
        probabilities = model.predict_proba(X_preprocessed)
        
        # Format responses
        responses = []
        for i, pred in enumerate(predictions):
            proba = probabilities[i]
            PREDICTION_COUNT.labels(prediction_class="positive" if pred == 1 else "negative").inc()
            if pred == 1:
                PREDICTIONS_POSITIVE.inc()
            else:
                PREDICTIONS_NEGATIVE.inc()
            
            responses.append({
                "prediction": int(pred),
                "confidence": float(max(proba)),
                "prediction_label": "Disease Present" if pred == 1 else "No Disease",
                "probabilities": {
                    "no_disease": float(proba[0]),
                    "disease_present": float(proba[1])
                },
                "timestamp": datetime.now().isoformat()
            })
        
        REQUEST_COUNT.labels(method="POST", endpoint="/batch_predict", status=200).inc()
        logger.info(f"Batch prediction made for {len(inputs)} samples")
        return responses
        
    except Exception as e:
        logger.error(f"Error during batch prediction: {str(e)}")
        REQUEST_COUNT.labels(method="POST", endpoint="/batch_predict", status=400).inc()
        raise HTTPException(status_code=400, detail=f"Batch prediction error: {str(e)}")


@app.get("/model-info", tags=["Model"])
async def model_info():
    """Get model information"""
    REQUEST_COUNT.labels(method="GET", endpoint="/model-info", status=200).inc()
    return {
        "model_type": "Random Forest Classifier",
        "features": [
            "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
            "thalach", "exang", "oldpeak", "slope", "ca", "thal"
        ],
        "n_features": 13,
        "target": "heart_disease",
        "target_classes": ["No Disease", "Disease Present"]
    }


@app.get("/metrics", tags=["Monitoring"])
async def metrics():
    """Prometheus metrics endpoint"""
    return Response(generate_latest(), media_type="text/plain")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
