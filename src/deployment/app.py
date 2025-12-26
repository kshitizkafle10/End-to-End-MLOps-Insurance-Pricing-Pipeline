from __future__ import annotations

from typing import Any
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.deployment.model_loader import load_champion_model


app = FastAPI(title="Insurance Pricing API", version="1.0")

model = load_champion_model()


class PredictRequest(BaseModel):
    # flexible: accepts any feature keys/values
    features: dict[str, Any] = Field(..., description="Feature values for one policy record")


class PredictResponse(BaseModel):
    prediction: float


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    X = pd.DataFrame([req.features])
    pred = float(model.predict(X)[0])
    return PredictResponse(prediction=pred)

@app.get("/")
def root():
    return {"message": "Go to /docs for Swagger UI"}

