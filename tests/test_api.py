import pandas as pd
from fastapi.testclient import TestClient

from src.deployment.app import app

client = TestClient(app)

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"

def test_predict_returns_number():
    df = pd.read_csv("data/processed/test.csv")
    X = df.drop(columns=["personalised_price"])
    payload = {"features": X.iloc[0].to_dict()}

    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert "prediction" in data
    assert isinstance(data["prediction"], (int, float))
