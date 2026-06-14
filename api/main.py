"""
FastAPI Backend — Cardiovascular Disease Risk Prediction.

Endpoints:
- GET  /health   -> {"status": "ok", "model_loaded": true}
- POST /predict  -> takes patient data, returns risk level + probability

Run locally:
    uvicorn api.main:app --reload --port 8000

Then open http://localhost:8000/docs for interactive Swagger UI.
"""

import sys
from pathlib import Path

# Add project root to path so we can import src/feature_engineering.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.feature_engineering import engineer_features

PREPROCESSOR_PATH = PROJECT_ROOT / "models" / "preprocessor.joblib"
MODEL_PATH = PROJECT_ROOT / "models" / "best_model.pkl"

app = FastAPI(
    title="Cardiovascular Disease Risk Prediction API",
    description="Predicts CVD risk from patient examination data.",
    version="1.0.0",
)

preprocessor = joblib.load(PREPROCESSOR_PATH)
model = joblib.load(MODEL_PATH)
print(f"Loaded preprocessor from {PREPROCESSOR_PATH}")
print(f"Loaded model ({type(model).__name__}) from {MODEL_PATH}")


# ---- Request schema -------------------------------------------------------
class Patient(BaseModel):
    """Raw patient fields. age is given in YEARS (the API converts to days
    internally, matching the original dataset's units for feature engineering)."""
    age_years: float = Field(..., ge=18, le=120, example=55)
    gender: int = Field(..., ge=1, le=2, example=1,
                        description="1 = female, 2 = male")
    height: int = Field(..., ge=100, le=220, example=168)
    weight: float = Field(..., gt=30, le=250, example=70)
    ap_hi: int = Field(..., ge=70, le=250, example=120,
                       description="Systolic blood pressure")
    ap_lo: int = Field(..., ge=40, le=200, example=80,
                       description="Diastolic blood pressure")
    cholesterol: int = Field(..., ge=1, le=3, example=1,
                             description="1=normal, 2=above, 3=well above")
    gluc: int = Field(..., ge=1, le=3, example=1,
                      description="1=normal, 2=above, 3=well above")
    smoke: int = Field(..., ge=0, le=1, example=0)
    alco: int = Field(..., ge=0, le=1, example=0)
    active: int = Field(..., ge=0, le=1, example=1)


class PredictionResponse(BaseModel):
    cvd_risk: str
    probability: float
    model_used: str


def risk_label(prob: float) -> str:
    if prob >= 0.65:
        return "High"
    if prob >= 0.35:
        return "Medium"
    return "Low"


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "model_type": type(model).__name__,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(patient: Patient) -> PredictionResponse:
    try:
        # Build a one-row DataFrame in the same shape as data/cleaned.csv:
        # the feature engineer expects an 'age' column in DAYS.
        raw = patient.model_dump()
        row = {
            "age": int(raw["age_years"] * 365.25),
            "gender": raw["gender"],
            "height": raw["height"],
            "weight": raw["weight"],
            "ap_hi": raw["ap_hi"],
            "ap_lo": raw["ap_lo"],
            "cholesterol": raw["cholesterol"],
            "gluc": raw["gluc"],
            "smoke": raw["smoke"],
            "alco": raw["alco"],
            "active": raw["active"],
        }
        df = pd.DataFrame([row])

        df = engineer_features(df)
        X = preprocessor.transform(df)

        prob = float(model.predict_proba(X)[0, 1])

        return PredictionResponse(
            cvd_risk=risk_label(prob),
            probability=round(prob, 4),
            model_used=type(model).__name__,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {e}")
