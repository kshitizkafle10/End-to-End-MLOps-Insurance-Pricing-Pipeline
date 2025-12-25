from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
TRAIN_PATH = PROCESSED_DIR / "train.csv"
TEST_PATH = PROCESSED_DIR / "test.csv"
TARGET_COL = "personalised_price"


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not TRAIN_PATH.exists() or not TEST_PATH.exists():
        raise FileNotFoundError(
            "Processed data not found. Run:\n"
            "  python src/preprocessing/make_dataset.py"
        )
    return pd.read_csv(TRAIN_PATH), pd.read_csv(TEST_PATH)


def make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric = X.select_dtypes(include=["number"]).columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]

    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
            ("num", "passthrough", numeric),
        ],
        remainder="drop",
    )


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(root_mean_squared_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def run_model(run_name: str, model, X_train, y_train, X_test, y_test) -> None:
    preprocessor = make_preprocessor(X_train)
    pipeline = Pipeline([("preprocess", preprocessor), ("model", model)])

    pipeline.fit(X_train, y_train)

    train_pred = pipeline.predict(X_train)
    test_pred = pipeline.predict(X_test)

    train_metrics = evaluate(y_train, train_pred)
    test_metrics = evaluate(y_test, test_pred)

    # Input example + signature (cast ints to float to avoid MLflow schema warning)
    input_example = X_train.head(5).copy()
    int_cols = input_example.select_dtypes(include=["int"]).columns
    input_example[int_cols] = input_example[int_cols].astype("float64")
    signature = infer_signature(input_example, pipeline.predict(input_example))

    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("model", model.__class__.__name__)
        for k, v in model.get_params().items():
            if isinstance(v, (str, int, float, bool)) or v is None:
                mlflow.log_param(k, v)

        for k, v in train_metrics.items():
            mlflow.log_metric(f"train_{k}", v)
        for k, v in test_metrics.items():
            mlflow.log_metric(f"test_{k}", v)

        mlflow.sklearn.log_model(
            pipeline,
            name="model",
            input_example=input_example,
            signature=signature,
        )

    print(f"\n✅ {run_name} complete")
    print("Train metrics:", train_metrics)
    print("Test metrics:", test_metrics)


def main() -> None:
    train_df, test_df = load_data()

    y_train = train_df[TARGET_COL].to_numpy()
    X_train = train_df.drop(columns=[TARGET_COL])
    y_test = test_df[TARGET_COL].to_numpy()
    X_test = test_df.drop(columns=[TARGET_COL])

    # SQLite backend
    mlflow.set_tracking_uri(f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}")
    mlflow.set_experiment("insurance_pricing_models")

    lgbm = LGBMRegressor(
        n_estimators=800,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
    )

    xgb = XGBRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
    )

    run_model("lightgbm_baseline", lgbm, X_train, y_train, X_test, y_test)
    run_model("xgboost_baseline", xgb, X_train, y_train, X_test, y_test)

    print("\nMLflow DB:", PROJECT_ROOT / "mlflow.db")


if __name__ == "__main__":
    main()
