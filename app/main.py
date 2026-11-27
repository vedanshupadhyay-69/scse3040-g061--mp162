import json
import logging
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


# JSON logging configuration
class JSONFormatter(logging.Formatter):
    def format(self, record):
        return json.dumps({
            "level": record.levelname,
            "message": record.getMessage()
        })


logger = logging.getLogger("opd_api")
logger.setLevel(logging.INFO)

handler = logging.StreamHandler()
handler.setFormatter(JSONFormatter())
logger.addHandler(handler)


# Load trained model
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "model.joblib"

model = joblib.load(MODEL_PATH)

app = FastAPI(title="OPD Waiting Time Prediction API")


# Input validation
class PredictionInput(BaseModel):
    patients_ahead: int = Field(ge=0)
    doctors_on_duty: int = Field(ge=1)
    arrival_hour: int = Field(ge=0, le=23)
    avg_consult_min: float = Field(gt=0)
    lab_tests_pending: int = Field(ge=0)
    distance_km: float = Field(ge=0)
    department: str
    wing: str


@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": model is not None}


@app.post("/predict")
def predict(data: PredictionInput):
    try:
        input_data = pd.DataFrame([data.model_dump()])

        prediction = model.predict(input_data)[0]

        logger.info("Prediction generated successfully")

        return {
            "predicted_wait_min": round(float(prediction), 2)
        }

    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Prediction failed"
        )