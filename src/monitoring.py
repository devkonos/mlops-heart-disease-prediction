"""
Monitoring and logging configuration for the API
"""
import logging
import json
from pythonjsonlogger import jsonlogger
from prometheus_client import Counter, Histogram, Gauge
import time
from functools import wraps
import os

# Create logs directory if not exists
os.makedirs('logs', exist_ok=True)

# Configure JSON logging
def setup_logging(log_level: str = "INFO"):
    """Setup structured logging with JSON format"""
    
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, log_level))
    
    # File handler with JSON formatting
    file_handler = logging.FileHandler('logs/api.log')
    file_handler.setLevel(getattr(logging, log_level))
    
    # JSON formatter
    formatter = jsonlogger.JsonFormatter()
    file_handler.setFormatter(formatter)
    
    # Add handler
    logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level))
    console_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ))
    logger.addHandler(console_handler)
    
    return logger


# Prometheus metrics
REQUEST_COUNT = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status']
)

REQUEST_DURATION = Histogram(
    'http_request_duration_seconds',
    'HTTP request duration in seconds',
    ['method', 'endpoint']
)

PREDICTION_COUNTER = Counter(
    'predictions_total',
    'Total predictions made',
    ['model', 'prediction_class']
)

PREDICTION_TIME = Histogram(
    'prediction_time_seconds',
    'Time taken to make predictions in seconds',
    ['model']
)

ACTIVE_REQUESTS = Gauge(
    'active_requests',
    'Number of active requests'
)

MODEL_ACCURACY = Gauge(
    'model_accuracy',
    'Model accuracy metric',
    ['model']
)

MODEL_PRECISION = Gauge(
    'model_precision',
    'Model precision metric',
    ['model']
)

MODEL_RECALL = Gauge(
    'model_recall',
    'Model recall metric',
    ['model']
)


def track_metrics(endpoint: str, model_name: str = "default"):
    """Decorator to track metrics for model predictions"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            ACTIVE_REQUESTS.inc()
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                
                PREDICTION_TIME.labels(model=model_name).observe(duration)
                REQUEST_DURATION.labels(method="POST", endpoint=endpoint).observe(duration)
                REQUEST_COUNT.labels(method="POST", endpoint=endpoint, status=200).inc()
                
                return result
            except Exception as e:
                duration = time.time() - start_time
                REQUEST_COUNT.labels(method="POST", endpoint=endpoint, status=500).inc()
                raise
            finally:
                ACTIVE_REQUESTS.dec()
        
        return wrapper
    return decorator


def log_prediction(prediction: int, confidence: float, patient_id: str = None):
    """Log prediction details"""
    logger = logging.getLogger(__name__)
    
    PREDICTION_COUNTER.labels(
        model="random_forest",
        prediction_class=str(prediction)
    ).inc()
    
    logger.info(json.dumps({
        "event": "prediction",
        "patient_id": patient_id,
        "prediction": prediction,
        "confidence": confidence,
        "prediction_label": "Disease Present" if prediction == 1 else "No Disease"
    }))


def log_model_metrics(model_name: str, accuracy: float, precision: float, 
                     recall: float, f1_score: float):
    """Log model evaluation metrics"""
    logger = logging.getLogger(__name__)
    
    MODEL_ACCURACY.labels(model=model_name).set(accuracy)
    MODEL_PRECISION.labels(model=model_name).set(precision)
    MODEL_RECALL.labels(model=model_name).set(recall)
    
    logger.info(json.dumps({
        "event": "model_evaluation",
        "model": model_name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score
    }))


# Initialize logger
logger = setup_logging("INFO")
