import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np

def batch_predict(feats_path="data/processed/features.parquet", model_uri="models:/uav_health_model/1"):
    df = pd.read_parquet(feats_path).replace([np.inf,-np.inf], np.nan).dropna()
    feature_cols = [c for c in df.columns if c not in ["y","flight_id","t_start","failure_type"]]
    model = mlflow.sklearn.load_model(model_uri)
    proba = model.predict_proba(df[feature_cols].values)[:,1]
    df["proba"] = proba
    return df[["flight_id","t_start","y","failure_type","proba"]]