from contextlib import asynccontextmanager
from pathlib import Path

import joblib
from fastapi import FastAPI, Request

from app.schemas import FEATURE_NAMES, HeartFeatures, HeartPrediction

MODEL_PATH = Path(__file__).resolve().parents[1] / "model" / "heart_model.joblib"


@asynccontextmanager
async def lifespan(app: FastAPI):
    artifact = joblib.load(MODEL_PATH)
    if artifact["features"] != FEATURE_NAMES:
        raise RuntimeError("The model's feature order does not match the API schema.")
    app.state.classifier = artifact["classifier"]
    yield


app = FastAPI(
    title="Heart Disease Prediction API",
    description="An educational classifier trained on the Kaggle Heart Disease dataset.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/info")
def info(request: Request) -> dict:
    return {
        "model_type": type(request.app.state.classifier).__name__,
        "features": FEATURE_NAMES,
    }


@app.post("/predict")
def predict(features: HeartFeatures, request: Request) -> HeartPrediction:
    values = features.model_dump()
    row = [values[name] for name in FEATURE_NAMES]
    prediction = request.app.state.classifier.predict([row])[0]
    return HeartPrediction(heart_disease=bool(prediction))
