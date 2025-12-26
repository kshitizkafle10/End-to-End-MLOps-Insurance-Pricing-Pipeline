from __future__ import annotations

from pathlib import Path
from itertools import product
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

    mlflow.set_tracking_uri(f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}")
    mlflow.set_experiment("insurance_pricing_lgbm_tuning")

    preprocessor = make_preprocessor(X_train)

    # Cast ints to float in input example to avoid MLflow schema warning
    input_example = X_train.head(5).copy()
    int_cols = input_example.select_dtypes(include=["int"]).columns
    input_example[int_cols] = input_example[int_cols].astype("float64")

    # Small grid (fast + meaningful)
    grid = {
        "n_estimators": [600, 1000],
        "learning_rate": [0.03, 0.05],
        "num_leaves": [31, 63],
        "max_depth": [-1, 8],
        "min_child_samples": [10, 25],
        "subsample": [0.8, 0.9],
        "colsample_bytree": [0.8, 0.9],
        "reg_lambda": [0.0, 1.0],
    }

    keys = list(grid.keys())
    combos = list(product(*[grid[k] for k in keys]))

    best = {"test_mae": float("inf"), "params": None, "run_id": None}

    for i, values in enumerate(combos, start=1):
        params = dict(zip(keys, values))

        model = LGBMRegressor(
            random_state=42,
            **params
        )

        pipeline = Pipeline([("preprocess", preprocessor), ("model", model)])
        pipeline.fit(X_train, y_train)

        train_pred = pipeline.predict(X_train)
        test_pred = pipeline.predict(X_test)

        train_metrics = evaluate(y_train, train_pred)
        test_metrics = evaluate(y_test, test_pred)

        signature = infer_signature(input_example, pipeline.predict(input_example))

        with mlflow.start_run(run_name=f"lgbm_grid_{i:03d}"):
            mlflow.log_param("model", "LGBMRegressor")
            for k, v in params.items():
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

            # Track best
            if test_metrics["mae"] < best["test_mae"]:
                best["test_mae"] = test_metrics["mae"]
                best["params"] = params
                best["run_id"] = mlflow.active_run().info.run_id

        print(
            f"[{i}/{len(combos)}] test_mae={test_metrics['mae']:.4f} "
            f"test_rmse={test_metrics['rmse']:.4f} test_r2={test_metrics['r2']:.4f}"
        )

    print("\n✅ Tuning complete")
    print("Best test_mae:", best["test_mae"])
    print("Best params:", best["params"])
    print("Best run_id:", best["run_id"])
    print("MLflow DB:", PROJECT_ROOT / "mlflow.db")


if __name__ == "__main__":
    main()