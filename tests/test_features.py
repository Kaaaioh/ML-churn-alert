"""Testes do pré-processamento e da montagem do pipeline."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from churn_alert.config import CATEGORICAL_FEATURES, NUMERIC_FEATURES, SEED
from churn_alert.features import build_pipeline, build_preprocessor


def _toy_frame(n: int = 20) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    data = {
        "tenure": rng.integers(0, 72, n),
        "MonthlyCharges": rng.uniform(20, 120, n),
        "TotalCharges": rng.uniform(0, 8000, n),
        "SeniorCitizen": rng.integers(0, 2, n),
    }
    for col in CATEGORICAL_FEATURES:
        data[col] = rng.choice(["A", "B"], n)
    return pd.DataFrame(data)


def test_preprocessor_outputs_numeric_matrix():
    X = _toy_frame()
    pre = build_preprocessor()
    Xt = pre.fit_transform(X)
    assert Xt.shape[0] == len(X)
    # 4 numéricas + 2 categorias para cada uma das 15 colunas categóricas
    assert Xt.shape[1] == len(NUMERIC_FEATURES) + 2 * len(CATEGORICAL_FEATURES)
    assert np.isfinite(Xt).all()


def test_preprocessor_ignores_unknown_category():
    X = _toy_frame()
    pre = build_preprocessor().fit(X)
    X_new = X.head(1).copy()
    X_new["Contract"] = "categoria-inexistente"
    Xt = pre.transform(X_new)  # não pode levantar exceção
    assert Xt.shape == (1, len(NUMERIC_FEATURES) + 2 * len(CATEGORICAL_FEATURES))


def test_pipeline_fits_and_predicts_probabilities():
    X = _toy_frame(40)
    y = np.array([0, 1] * 20)
    pipe = build_pipeline(LogisticRegression(random_state=SEED, max_iter=500))
    pipe.fit(X, y)
    proba = pipe.predict_proba(X)[:, 1]
    assert proba.shape == (40,)
    assert ((proba >= 0) & (proba <= 1)).all()
