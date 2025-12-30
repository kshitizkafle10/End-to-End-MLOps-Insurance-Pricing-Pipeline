from __future__ import annotations

import os
from pathlib import Path
import mlflow


def load_champion_model():
    project_root = Path(__file__).resolve().parents[2]
    db_path = project_root / "mlflow.db"
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{db_path}"))

    model_uri = os.getenv("MODEL_URI", "models:/insurance_pricing_model@production")
    return mlflow.pyfunc.load_model(model_uri)
