"""Train v1 regression models, track with MLflow, register best model as @champion."""

import hashlib
import os
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.mp162.cleaning import clean_dataframe, load_params


def compute_sha256(file_path: str) -> str:
    with open(file_path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def build_pipeline(model, numeric_cols: list, categorical_cols: list) -> Pipeline:
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )
    return Pipeline(steps=[("preprocessor", preprocessor), ("regressor", model)])


def get_candidate_models(random_state: int) -> dict:
    return {
        "ridge": Ridge(alpha=1.0),
        "random_forest": RandomForestRegressor(
            n_estimators=200, max_depth=12, random_state=random_state, n_jobs=-1
        ),
        "gradient_boosting": GradientBoostingRegressor(
            n_estimators=200, learning_rate=0.05, max_depth=4, random_state=random_state
        ),
        "hist_gradient_boosting": HistGradientBoostingRegressor(
            max_iter=200, learning_rate=0.05, max_depth=6, random_state=random_state
        ),
    }


def train_v1(params_path: str = "params.yaml") -> dict:
    params = load_params(params_path)
    raw_train_path = params["data"]["raw_train"]
    data_sha256 = compute_sha256(raw_train_path)

    raw_df = pd.read_csv(raw_train_path)
    df = clean_dataframe(raw_df, is_train=True)

    num_cols = params["features"]["numeric"]
    cat_cols = params["features"]["categorical"]
    feature_cols = num_cols + cat_cols
    target_col = params["features"]["target"]

    X = df[feature_cols]
    y = df[target_col]

    rs = params["model"]["random_state"]
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=params["model"]["val_size"], random_state=rs
    )

    mlflow.set_tracking_uri(params["mlflow"]["tracking_uri"])
    experiment_name = params["mlflow"]["experiment_name"]
    mlflow.set_experiment(experiment_name)
    reg_model_name = params["mlflow"]["registered_model_name"]

    candidates = get_candidate_models(rs)
    best_mae = float("inf")
    best_run_id = None
    best_name = None
    best_pipeline = None

    for name, estimator in candidates.items():
        pipe = build_pipeline(estimator, num_cols, cat_cols)
        with mlflow.start_run(run_name=f"v1_{name}") as run:
            pipe.fit(X_train, y_train)
            val_preds = pipe.predict(X_val)
            train_preds = pipe.predict(X_train)
            val_mae = float(mean_absolute_error(y_val, val_preds))
            train_mae = float(mean_absolute_error(y_train, train_preds))

            mlflow.log_param("data_sha256", data_sha256)
            mlflow.log_param("model_type", name)
            mlflow.log_param("stage", "v1")
            mlflow.log_metric("val_mae", round(val_mae, 4))
            mlflow.log_metric("train_mae", round(train_mae, 4))
            mlflow.sklearn.log_model(
                pipe,
                artifact_path="model",
                serialization_format="cloudpickle",
            )

            print(f"[v1] {name:24s} | val_mae={val_mae:.4f} | run_id={run.info.run_id}")
            if val_mae < best_mae:
                best_mae = val_mae
                best_run_id = run.info.run_id
                best_name = name

    # Refit the winning architecture on 100% of cleaned train.csv for deployment/v1
    best_estimator = get_candidate_models(rs)[best_name]
    best_pipeline = build_pipeline(best_estimator, num_cols, cat_cols)
    best_pipeline.fit(X, y)

    # Register best v1 run as version 1 and assign @champion alias
    client = MlflowClient(tracking_uri=params["mlflow"]["tracking_uri"])
    model_uri = f"runs:/{best_run_id}/model"
    mv = mlflow.register_model(model_uri=model_uri, name=reg_model_name)
    client.set_registered_model_alias(reg_model_name, "champion", mv.version)

    os.makedirs(os.path.dirname(params["model"]["output_path"]), exist_ok=True)
    joblib.dump(best_pipeline, params["model"]["output_path"])
    # Also save v1 explicitly for Stage 4 comparison
    joblib.dump(best_pipeline, "models/model_v1.joblib")

    print(
        f"\nRegistered {reg_model_name} v{mv.version} (@champion) from {best_name} "
        f"(run_id={best_run_id}, val_mae={best_mae:.4f})"
    )
    return {
        "experiment_name": experiment_name,
        "best_run_id": best_run_id,
        "registered_model_name": reg_model_name,
        "champion_version": int(mv.version),
        "best_model_name": best_name,
    }


if __name__ == "__main__":
    train_v1()
