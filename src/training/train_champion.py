from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

from lightgbm import LGBMRegressor

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


def main() -> None:
    train_df, test_df = load_data()

    y_train = train_df[TARGET_COL].to_numpy()
    X_train = train_df.drop(columns=[TARGET_COL])
    y_test = test_df[TARGET_COL].to_numpy()
    X_test = test_df.drop(columns=[TARGET_COL])

    # MLflow tracking (SQLite)
    mlflow.set_tracking_uri(f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}")
    mlflow.set_experiment("insurance_pricing_champion")

    # Best params from tuning
    best_params = {
        "n_estimators": 600,
        "learning_rate": 0.03,
        "num_leaves": 31,
        "max_depth": -1,
        "min_child_samples": 10,
        "subsample": 0.8,
        "colsample_bytree": 0.9,
        "reg_lambda": 1.0,
        "random_state": 42,
    }

    model = LGBMRegressor(**best_params)
    preprocessor = make_preprocessor(X_train)
    pipeline = Pipeline([("preprocess", preprocessor), ("model", model)])

    # Input example + signature (avoid integer-missing schema warning)
    input_example = X_train.head(5).copy()
    int_cols = input_example.select_dtypes(include=["int"]).columns
    input_example[int_cols] = input_example[int_cols].astype("float64")

    with mlflow.start_run(run_name="lgbm_champion_v1") as run:
        pipeline.fit(X_train, y_train)

        train_pred = pipeline.predict(X_train)
        test_pred = pipeline.predict(X_test)

        train_metrics = evaluate(y_train, train_pred)
        test_metrics = evaluate(y_test, test_pred)

        # log params
        mlflow.log_param("model", "LGBMRegressor")
        for k, v in best_params.items():
            mlflow.log_param(k, v)

        # log metrics
        for k, v in train_metrics.items():
            mlflow.log_metric(f"train_{k}", v)
        for k, v in test_metrics.items():
            mlflow.log_metric(f"test_{k}", v)

        signature = infer_signature(input_example, pipeline.predict(input_example))

        # log model
        mlflow.sklearn.log_model(
            pipeline,
            name="model",
            input_example=input_example,
            signature=signature,
        )

        mlflow.set_tag("stage", "champion")
        mlflow.set_tag("tuning_best_run_id", "4af38bc2d48646288f33ad8b0af2046a")
        mlflow.set_tag("notes", "Champion model trained from best LGBM tuning params (grid search).")

        print("\n✅ Champion training complete")
        print("Train metrics:", train_metrics)
        print("Test metrics:", test_metrics)
        print("MLflow run_id:", run.info.run_id)
        print("MLflow DB:", PROJECT_ROOT / "mlflow.db")


if __name__ == "__main__":
    main()
