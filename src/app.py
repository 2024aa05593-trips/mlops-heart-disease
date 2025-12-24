from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import pickle
import os
from prometheus_fastapi_instrumentator import Instrumentator

app = FastAPI(title="Heart Disease Prediction API")

# Define the input schema
class PatientData(BaseModel):
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

# Load the model
MODEL_PATH = "models/model.pkl"
if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
else:
    model = None

@app.get("/")
def read_root():
    return {"message": "Heart Disease Prediction API is running"}

@app.post("/predict")
def predict(data: PatientData):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded. Please train the model first.")
    
    # Convert input data to DataFrame
    input_df = pd.DataFrame([data.dict()])
    
    # Make prediction
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0].tolist()
    
    return {
        "prediction": int(prediction),
        "confidence": max(probability),
        "probabilities": {
            "no_disease": probability[0],
            "disease": probability[1]
        }
    }

# Instrument the app
Instrumentator().instrument(app).expose(app)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
