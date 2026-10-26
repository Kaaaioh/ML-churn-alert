"""Testes do módulo de predição (Predictor)."""

import pytest

from churn_alert.predict import HIGH_RISK_THRESHOLD, Predictor


def test_load_reads_pipeline_and_metadata(tiny_model_path):
    predictor = Predictor.load(tiny_model_path)
    assert predictor.threshold == 0.35
    assert predictor.model_name == "logistic_regression"
    assert len(predictor.features) == 19


def test_load_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        Predictor.load(tmp_path / "nao_existe.joblib")


def test_predict_one_returns_valid_prediction(tiny_model_path, sample_customer):
    predictor = Predictor.load(tiny_model_path)
    result = predictor.predict_one(sample_customer)
    assert 0.0 <= result.churn_probability <= 1.0
    assert result.churn_label in (0, 1)
    assert result.risk_tier in ("baixo", "medio", "alto")
    assert result.suggested_action
    assert result.churn_label == int(result.churn_probability >= predictor.threshold)


def test_risky_customer_scores_higher_than_loyal(tiny_model_path, sample_customer, loyal_customer):
    predictor = Predictor.load(tiny_model_path)
    risky = predictor.predict_one(sample_customer).churn_probability
    loyal = predictor.predict_one(loyal_customer).churn_probability
    assert risky > loyal


def test_risk_tiers_follow_thresholds(tiny_model_path):
    predictor = Predictor.load(tiny_model_path)
    assert predictor.risk_tier(predictor.threshold - 0.01) == "baixo"
    assert predictor.risk_tier(predictor.threshold) == "medio"
    assert predictor.risk_tier(HIGH_RISK_THRESHOLD) == "alto"


def test_predict_batch_preserves_order_and_handles_empty(
    tiny_model_path, sample_customer, loyal_customer
):
    predictor = Predictor.load(tiny_model_path)
    results = predictor.predict_batch([sample_customer, loyal_customer])
    assert len(results) == 2
    assert results[0].churn_probability > results[1].churn_probability
    assert predictor.predict_batch([]) == []
