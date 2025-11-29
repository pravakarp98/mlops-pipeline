import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
import pandas as pd
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from app.schemas import StudentFeatures, PredictionResponse

app = FastAPI(
    title="Student Academic Risk Predictor",
    description="An MLOps production API to predict student dropout risk.",
    version="1.0.0"
)

try:
    pipeline = joblib.load('artifacts/pipeline.pkl')
    label_encoder = joblib.load('artifacts/label_encoder.pkl')
    print("Model and Encoder loaded successfully.")

except FileNotFoundError:
    print("Error: Artifacts not found. Did you run src/train.py?")
    pipeline = None
    label_encoder = None

@app.get("/", tags=["Health Check"])
def index():

    return {"message": "Student Risk API is running!", "status": "OK"}

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(input_data: StudentFeatures):
    """
    Receives student data and returns the predicted academic outcome.
    """
    if not pipeline or not label_encoder:
        raise HTTPException(status_code=500, detail="Model not loaded.")
    
    try:
        data_dict = input_data.model_dump(by_alias=True)
        df = pd.DataFrame([data_dict])
        prediction_idx = pipeline.predict(df)[0]
        probs = pipeline.predict_proba(df)[0]
        prediction_label = label_encoder.inverse_transform([prediction_idx])[0]

        class_names = label_encoder.classes_
        prob_dict = {class_name: float(prob) for class_name, prob in zip(class_names, probs)}

        return {
            "prediction": prediction_label,
            "probability": prob_dict
        }

    except Exception as e:
        print(f"Prediction Error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))