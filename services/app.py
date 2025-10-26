import os, json
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field
import mlflow
import mlflow.sklearn
from steps.features import window_feature_frame

FS = int(os.getenv("FS","100"))
MODEL_URI = os.getenv("MODEL_URI", "models:/uav_health_model/1")
THR = float(os.getenv("THRESHOLD","0.5"))

class WindowPayload(BaseModel):
    t: list[float] = Field(..., description="timestamps in seconds")
    ax: list[float]; ay: list[float]; az: list[float]
    p: list[float]; q: list[float]; r: list[float]
    altitude: list[float]; speed: list[float]; temperature: list[float]
    pitch: list[float]; roll: list[float]; heading: list[float]
    baro_alt: list[float]; gnss_alt: list[float]; gnss_speed: list[float]

app = FastAPI(title="UAV Health Predictor")

model = None
feature_order = None

@app.on_event("startup")
def load_model():
    global model, feature_order
    model = mlflow.sklearn.load_model(MODEL_URI)
    fl = "models/feature_list.json"
    if os.path.exists(fl):
        with open(fl,"r") as f:
            feature_order = json.load(f)

@app.get("/")
def health():
    return {"status":"ok", "model_uri": MODEL_URI}

@app.post("/predict")
def predict(payload: WindowPayload):
    df = pd.DataFrame(payload.dict())
    df["failure_type"] = "none"
    df["label"] = 0
    feats = window_feature_frame(df, fs=FS)
    feats.pop("y", None); feats.pop("failure_type", None)
    X = pd.DataFrame([feats])
    if feature_order:
        for c in feature_order:
            if c not in X.columns: X[c] = 0.0
        X = X[feature_order]
    proba = float(model.predict_proba(X.values)[:,1][0])
    return {"probability": proba, "alert": proba >= THR}