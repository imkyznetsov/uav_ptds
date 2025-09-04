import os, json
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from lightgbm import LGBMClassifier

TARGET = "y"
DROP = ["y","flight_id","t_start","failure_type"]

class Trainer:
    def __init__(self, feats_path="data/processed/features.parquet", experiment="uav_failure_pred"):
        self.feats_path = feats_path
        self.experiment = experiment
        self.feature_cols = None
        self.model = None

    def load(self):
        df = pd.read_parquet(self.feats_path)
        df = df.replace([np.inf, -np.inf], np.nan).dropna()
        self.feature_cols = [c for c in df.columns if c not in DROP]
        return df

    def train_cv(self, df: pd.DataFrame, n_splits=5):
        X = df[self.feature_cols].values
        y = df[TARGET].values
        groups = df["flight_id"].values
        gkf = GroupKFold(n_splits=n_splits)

        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI","file:./mlruns"))
        mlflow.set_experiment(self.experiment)

        params = dict(
            n_estimators=600,
            learning_rate=0.03,
            num_leaves=63,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            min_child_samples=20,
            objective="binary",
            class_weight="balanced"
        )

        metrics_sum = dict(roc_auc=0.0, ap=0.0, f1=0.0)
        with mlflow.start_run() as run:
            mlflow.log_params(params)
            fold_models = []
            for i, (tr, va) in enumerate(gkf.split(X, y, groups)):
                clf = LGBMClassifier(**params)
                clf.fit(X[tr], y[tr], eval_set=[(X[va], y[va])], eval_metric="auc", verbose=False)
                proba = clf.predict_proba(X[va])[:,1]
                pred = (proba >= 0.5).astype(int)
                roc = roc_auc_score(y[va], proba)
                ap = average_precision_score(y[va], proba)
                f1 = f1_score(y[va], pred)
                mlflow.log_metric(f"fold_{i}_roc", float(roc))
                mlflow.log_metric(f"fold_{i}_ap", float(ap))
                mlflow.log_metric(f"fold_{i}_f1", float(f1))
                metrics_sum["roc_auc"] += roc; metrics_sum["ap"] += ap; metrics_sum["f1"] += f1
                fold_models.append(clf)

            for k in metrics_sum:
                metrics_sum[k] /= n_splits
                mlflow.log_metric(k, float(metrics_sum[k]))

            # обучим финальную модель на всем датасете
            self.model = LGBMClassifier(**params).fit(X, y)
            importances = dict(zip(self.feature_cols, self.model.feature_importances_.tolist()))
            mlflow.log_text(json.dumps(importances, indent=2), "feature_importances.json")
            # сохраняем список фич
            os.makedirs("models", exist_ok=True)
            with open("models/feature_list.json","w") as f:
                json.dump(self.feature_cols, f, indent=2)
            mlflow.log_artifact("models/feature_list.json")
            mlflow.sklearn.log_model(self.model, artifact_path="model", registered_model_name="uav_health_model")
            print("CV metrics:", metrics_sum)