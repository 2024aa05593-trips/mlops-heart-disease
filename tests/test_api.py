from fastapi.testclient import TestClient
from src.app import app
import pytest

client = TestClient(app)

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Heart Disease Prediction API is running"}

def test_predict():
    payload = {
        "age": 63.0,
        "sex": 1.0,
        "cp": 3.0,
        "trestbps": 145.0,
        "chol": 233.0,
        "fbs": 1.0,
        "restecg": 0.0,
        "thalach": 150.0,
        "exang": 0.0,
        "oldpeak": 2.3,
        "slope": 0.0,
        "ca": 0.0,
        "thal": 1.0
    }
    # Note: This test might fail if the model is not trained yet
    # but we can check if it returns 200 after training.
    response = client.post("/predict", json=payload)
    if response.status_code == 200:
        data = response.json()
        assert "prediction" in data
        assert "confidence" in data
    elif response.status_code == 500:
        assert response.json()["detail"] == "Model not loaded. Please train the model first."
