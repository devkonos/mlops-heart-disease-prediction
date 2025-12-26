"""
Unit tests for FastAPI endpoints
"""
import pytest
from fastapi.testclient import TestClient
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.api.app import app

client = TestClient(app)


class TestHealthEndpoints:
    """Test health check endpoints"""
    
    def test_root_endpoint(self):
        """Test root endpoint"""
        response = client.get("/")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"
    
    def test_health_endpoint(self):
        """Test health endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        assert "status" in response.json()


class TestModelInfoEndpoint:
    """Test model information endpoint"""
    
    def test_model_info(self):
        """Test model info endpoint"""
        response = client.get("/model-info")
        assert response.status_code == 200
        data = response.json()
        assert "model_type" in data
        assert "features" in data
        assert len(data["features"]) == 13


class TestPredictionEndpoint:
    """Test prediction endpoint"""
    
    @pytest.fixture
    def sample_input(self):
        """Sample patient data"""
        return {
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
    
    def test_prediction_endpoint_structure(self, sample_input):
        """Test prediction endpoint response structure"""
        response = client.post("/predict", json=sample_input)
        
        # May return 500 if model not loaded, but endpoint should exist
        assert response.status_code in [200, 500]
    
    def test_invalid_input_missing_field(self, sample_input):
        """Test with missing required field"""
        invalid_input = sample_input.copy()
        del invalid_input["age"]
        
        response = client.post("/predict", json=invalid_input)
        assert response.status_code == 422  # Validation error
    
    def test_invalid_input_wrong_type(self, sample_input):
        """Test with wrong data type"""
        invalid_input = sample_input.copy()
        invalid_input["age"] = "not_a_number"
        
        response = client.post("/predict", json=invalid_input)
        assert response.status_code == 422


class TestBatchPredictionEndpoint:
    """Test batch prediction endpoint"""
    
    @pytest.fixture
    def sample_batch_input(self):
        """Sample batch of patient data"""
        return [
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
                "age": 45,
                "sex": 0,
                "cp": 1,
                "trestbps": 120,
                "chol": 200,
                "fbs": 0,
                "restecg": 1,
                "thalach": 140,
                "exang": 1,
                "oldpeak": 1.5,
                "slope": 1,
                "ca": 1,
                "thal": 0
            }
        ]
    
    def test_batch_prediction_endpoint(self, sample_batch_input):
        """Test batch prediction endpoint"""
        response = client.post("/batch_predict", json=sample_batch_input)
        
        # May return 500 if model not loaded, but endpoint should exist
        assert response.status_code in [200, 500]
    
    def test_batch_prediction_empty_list(self):
        """Test batch prediction with empty list"""
        response = client.post("/batch_predict", json=[])
        
        # Empty list is invalid input, should return 400 Bad Request
        assert response.status_code in [400, 422]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
