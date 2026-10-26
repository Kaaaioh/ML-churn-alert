"""Testes da API FastAPI (via TestClient, sem subir servidor)."""

import pytest
from fastapi.testclient import TestClient

from churn_alert.api import create_app
from churn_alert.predict import DEFAULT_MODEL_PATH


@pytest.fixture
def client(tiny_model_path):
    # `with` dispara o lifespan (carrega o modelo), como no boot real
    with TestClient(create_app(model_path=tiny_model_path)) as c:
        yield c


def test_health_reports_model_loaded(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["model_name"] == "logistic_regression"


def test_health_without_model_still_answers(tmp_path):
    with TestClient(create_app(model_path=tmp_path / "missing.joblib")) as c:
        resp = c.get("/health")
        assert resp.status_code == 200
        assert resp.json()["model_loaded"] is False
        assert c.post("/predict", json={}).status_code in (422, 503)
        assert c.get("/model").status_code == 503


def test_model_info_exposes_threshold_and_features(client):
    resp = client.get("/model")
    assert resp.status_code == 200
    body = resp.json()
    assert body["threshold"] == 0.35
    assert "tenure" in body["features"]


def test_predict_returns_probability_label_and_tier(client, sample_customer):
    resp = client.post("/predict", json=sample_customer)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert body["churn_label"] in (0, 1)
    assert body["risk_tier"] in ("baixo", "medio", "alto")
    assert body["threshold"] == 0.35


def test_predict_rejects_invalid_category(client, sample_customer):
    resp = client.post("/predict", json={**sample_customer, "Contract": "Mensal"})
    assert resp.status_code == 422
    assert "Contract" in resp.text


def test_predict_rejects_missing_field_and_negative_number(client, sample_customer):
    payload = dict(sample_customer)
    del payload["tenure"]
    assert client.post("/predict", json=payload).status_code == 422
    assert (
        client.post("/predict", json={**sample_customer, "MonthlyCharges": -5}).status_code == 422
    )


def test_predict_batch_returns_one_prediction_per_customer(client, sample_customer, loyal_customer):
    resp = client.post("/predict/batch", json={"customers": [sample_customer, loyal_customer]})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["count"] == 2
    assert len(body["predictions"]) == 2
    assert body["predictions"][0]["churn_probability"] > body["predictions"][1]["churn_probability"]


def test_predict_batch_rejects_empty_list(client):
    assert client.post("/predict/batch", json={"customers": []}).status_code == 422


@pytest.mark.skipif(not DEFAULT_MODEL_PATH.exists(), reason="campeão real não gerado")
def test_real_champion_flags_risky_customer(sample_customer, loyal_customer):
    """Teste de fumaça com o artefato real: cliente de risco deve cair em 'medio' ou 'alto'."""
    with TestClient(create_app()) as c:
        risky = c.post("/predict", json=sample_customer).json()
        loyal = c.post("/predict", json=loyal_customer).json()
    assert risky["risk_tier"] in ("medio", "alto")
    assert loyal["risk_tier"] == "baixo"
    assert risky["churn_probability"] > loyal["churn_probability"]
