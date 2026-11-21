import os

import pytest

from src.etl_fetch import synthetic_btc
from src.train_forecast import train_and_register


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    d = tmp_path_factory.mktemp("models")
    os.environ["MODEL_DIR"] = str(d)
    train_and_register(synthetic_btc(400), str(d), force=True)
    import app as app_module

    return app_module.app.test_client()


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.get_json()
    assert body["status"] == "ok"
    assert body["model_version"].startswith("v")


def test_predict_default_7_days(client):
    r = client.post("/predict", json={})
    assert r.status_code == 200
    preds = r.get_json()["predictions"]
    assert len(preds) == 7
    assert "date" in preds[0] and "predicted_close" in preds[0]


def test_predict_custom_days(client):
    r = client.post("/predict", json={"forecast_days": 3})
    assert len(r.get_json()["predictions"]) == 3


def test_predict_bad_days(client):
    assert client.post("/predict", json={"forecast_days": "abc"}).status_code == 400


def test_predict_bad_json(client):
    r = client.post("/predict", data="not json", content_type="application/json")
    assert r.status_code == 400


def test_predict_short_history(client):
    assert client.post("/predict", json={"history": [1, 2, 3]}).status_code == 400


def test_predict_custom_history(client):
    hist = [30000 + i * 10 for i in range(60)]
    r = client.post("/predict", json={"history": hist})
    assert r.status_code == 200


def test_metrics(client):
    r = client.get("/metrics")
    assert r.status_code == 200
    assert "mae" in r.get_json()


def test_unknown_route(client):
    assert client.get("/nope").status_code == 404


def test_reload_needs_key(client, monkeypatch):
    monkeypatch.setenv("RELOAD_API_KEY", "secret")
    assert client.post("/reload").status_code == 401
    assert client.post("/reload", headers={"X-API-Key": "secret"}).status_code == 200
