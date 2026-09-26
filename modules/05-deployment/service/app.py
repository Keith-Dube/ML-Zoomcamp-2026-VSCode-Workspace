from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="ML Zoomcamp Prediction Service")


class PredictionRequest(BaseModel):
    features: dict[str, float | int | str]


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict")
def predict(request: PredictionRequest) -> dict[str, object]:
    return {"prediction": None, "received_features": request.features}

