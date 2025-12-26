from __future__ import annotations

from pathlib import Path
import mlflow
import mlflow.sklearn


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MLFLOW_DB = PROJECT_ROOT / "mlflow.db"

# Your champion run id
CHAMPION_RUN_ID = "f77e04f9d14546e2bb7e35cae7c5830f"


def load_champion_model():
    """
    Loads the sklearn Pipeline logged in MLflow under the champion run.
    """
    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB}")
    model_uri = f"runs:/{CHAMPION_RUN_ID}/model"
    model = mlflow.sklearn.load_model(model_uri)
    return model