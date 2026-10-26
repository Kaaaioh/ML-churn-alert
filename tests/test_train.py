"""Testes do fluxo de treino, comparação e seleção do campeão (com dados sintéticos pequenos)."""

import json

import numpy as np
import pandas as pd
import pytest

from churn_alert.config import CATEGORICAL_FEATURES, SEED
from churn_alert.train import (
    compare_models,
    fit_full,
    get_candidate_models,
    save_champion,
    select_champion,
)


@pytest.fixture
def toy_data():
    rng = np.random.default_rng(SEED)
    n = 120
    X = pd.DataFrame(
        {
            "tenure": rng.integers(0, 72, n),
            "MonthlyCharges": rng.uniform(20, 120, n),
            "TotalCharges": rng.uniform(0, 8000, n),
            "SeniorCitizen": rng.integers(0, 2, n),
        }
    )
    for col in CATEGORICAL_FEATURES:
        X[col] = rng.choice(["A", "B"], n)
    # alvo com sinal fraco em tenure, para o treino não ser trivial nem aleatório
    y = pd.Series(((X["tenure"] < 20) & (rng.uniform(size=n) < 0.8)).astype(int))
    return X, y


def test_candidate_models_cover_required_families():
    models = get_candidate_models()
    assert set(models) == {"logistic_regression", "random_forest", "mlp"}


def test_compare_models_returns_metrics_per_model(toy_data):
    X, y = toy_data
    models = {"logistic_regression": get_candidate_models()["logistic_regression"]}
    results = compare_models(models, X, y, n_splits=3)
    assert list(results.index) == ["logistic_regression"]
    for col in ["f1_mean", "f1_std", "roc_auc_mean", "fit_time_s"]:
        assert col in results.columns
    assert 0.0 <= results.loc["logistic_regression", "f1_mean"] <= 1.0


def test_select_champion_prefers_simpler_on_tie():
    results = pd.DataFrame(
        {"f1_mean": [0.600, 0.603, 0.580]},
        index=["logistic_regression", "random_forest", "mlp"],
    )
    assert select_champion(results, tolerance=0.005) == "logistic_regression"
    assert select_champion(results, tolerance=0.0) == "random_forest"


def test_fit_full_and_save_champion_writes_artifact_and_metadata(toy_data, tmp_path):
    X, y = toy_data
    pipe = fit_full(get_candidate_models()["logistic_regression"], X, y)
    meta_in = {"model": "lr", "threshold": 0.4}
    path, meta_path = save_champion(pipe, tmp_path / "champion.joblib", meta_in)
    assert path.exists() and meta_path.exists()
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["model"] == "lr"
    assert meta["threshold"] == 0.4
    assert "sklearn_version" in meta
