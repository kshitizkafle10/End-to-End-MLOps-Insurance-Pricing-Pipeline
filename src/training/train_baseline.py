from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from mlflow.models.signature import infer_signature


import mlflow
import mlflow.sklearn


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

TRAIN_PATH = PROCESSED_DIR / "train.csv"
TEST_PATH = PROCESSED_DIR / "test.csv"

TARGET_COL = "personalised_price"


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not TRAIN_PATH.exists() or not TEST_PATH.exists():
        raise FileNotFoundError(
            "Processed data not found.\n"
            f"Expected: {TRAIN_PATH} and {TEST_PATH}\n"
            "Run preprocessing first:\n"
            "  python src/preprocessing/make_dataset.py"
        )
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    return train_df, test_df


def build_pipeline(X: pd.DataFrame) -> Pipeline:
    # Identify columns
    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = [c for c in X.columns if c not in numeric_features]

    # Preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("num", "passthrough", numeric_features),
        ],
        remainder="drop",
    )

    # Baseline model
    model = Ridge(alpha=1.0, random_state=42)

    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )
    return pipeline


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {"mae": float(mae), "rmse": float(rmse), "r2": float(r2)}


def main() -> None:
    train_df, test_df = load_data()

    # Split features/target
    y_train = train_df[TARGET_COL].to_numpy()
    X_train = train_df.drop(columns=[TARGET_COL])

    y_test = test_df[TARGET_COL].to_numpy()
    X_test = test_df.drop(columns=[TARGET_COL])

    pipeline = build_pipeline(X_train)
    
    mlflow.set_tracking_uri(f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}")

    # MLflow setup (defaults to local ./mlruns)
    mlflow.set_experiment("insurance_pricing_baseline")

    with mlflow.start_run(run_name="ridge_baseline"):
        # Log basic params
        mlflow.log_param("model", "Ridge")
        mlflow.log_param("alpha", 1.0)
        mlflow.log_param("target", TARGET_COL)
        mlflow.log_param("n_train", len(X_train))
        mlflow.log_param("n_test", len(X_test))

        # Train
        pipeline.fit(X_train, y_train)

        # Predict + evaluate
        train_pred = pipeline.predict(X_train)
        test_pred = pipeline.predict(X_test)

        train_metrics = evaluate(y_train, train_pred)
        test_metrics = evaluate(y_test, test_pred)

        # Log metrics (prefix so you can compare later)
        for k, v in train_metrics.items():
            mlflow.log_metric(f"train_{k}", v)
        for k, v in test_metrics.items():
            mlflow.log_metric(f"test_{k}", v)
        
        
        # Log model artifact
        input_example = X_train.head(5)
        signature = infer_signature(input_example, pipeline.predict(input_example))

        mlflow.sklearn.log_model(
             pipeline,
             name="model",
             input_example=input_example,
             signature=signature,
)

        

        print("✅ Baseline training complete")
        print("Train metrics:", train_metrics)
        print("Test metrics:", test_metrics)
        print("MLflow DB:", PROJECT_ROOT / "mlflow.db")


if __name__ == "__main__":
    main()
