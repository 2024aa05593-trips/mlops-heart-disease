from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import pickle
import os
import traceback
from prometheus_fastapi_instrumentator import Instrumentator

app = FastAPI(title="Heart Disease Prediction API")


# ------------------------------
# Input Schema
# ------------------------------
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


# ------------------------------
# Load Model
# ------------------------------
MODEL_PATH = "models/model.pkl"
if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
else:
    model = None


# ------------------------------
# Routes
# ------------------------------
@app.get("/")
def read_root():
    return {"message": "Heart Disease Prediction API is running"}


@app.post("/predict")
def predict(data: PatientData):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded. Please train the model first.")

    try:
        input_df = pd.DataFrame([data.dict()])

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

    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


# ------------------------------
# Monitoring Metrics
# ------------------------------
@app.on_event("startup")
async def startup():
    Instrumentator().instrument(app).expose(app)


# ------------------------------
# Run Local
# ------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
