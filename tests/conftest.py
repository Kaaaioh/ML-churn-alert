"""Fixtures compartilhadas pelos testes.

`tiny_model_path` treina um modelo pequeno em dados sintéticos e o salva numa pasta temporária.
Assim os testes da API rodam em qualquer máquina (ou CI) mesmo sem `models/champion.joblib`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from churn_alert.config import CATEGORICAL_FEATURES, SEED
from churn_alert.train import fit_full, get_candidate_models, save_champion

CATEGORY_DOMAINS: dict[str, list[str]] = {
    "gender": ["Female", "Male"],
    "Partner": ["Yes", "No"],
    "Dependents": ["Yes", "No"],
    "PhoneService": ["Yes", "No"],
    "MultipleLines": ["Yes", "No", "No phone service"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "OnlineSecurity": ["Yes", "No", "No internet service"],
    "OnlineBackup": ["Yes", "No", "No internet service"],
    "DeviceProtection": ["Yes", "No", "No internet service"],
    "TechSupport": ["Yes", "No", "No internet service"],
    "StreamingTV": ["Yes", "No", "No internet service"],
    "StreamingMovies": ["Yes", "No", "No internet service"],
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": ["Yes", "No"],
    "PaymentMethod": [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ],
}


def make_synthetic_customers(n: int = 200, seed: int = SEED) -> tuple[pd.DataFrame, pd.Series]:
    """Clientes sintéticos com os domínios reais do dataset e um sinal simples de churn."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        {
            "tenure": rng.integers(0, 72, n),
            "MonthlyCharges": rng.uniform(20, 120, n).round(2),
            "TotalCharges": rng.uniform(0, 8000, n).round(2),
            "SeniorCitizen": rng.integers(0, 2, n),
        }
    )
    for col in CATEGORICAL_FEATURES:
        X[col] = rng.choice(CATEGORY_DOMAINS[col], n)
    churn_signal = (X["tenure"] < 12) | (X["Contract"] == "Month-to-month")
    y = pd.Series((churn_signal & (rng.uniform(size=n) < 0.7)).astype(int))
    return X, y


@pytest.fixture(scope="session")
def tiny_model_path(tmp_path_factory):
    X, y = make_synthetic_customers()
    pipe = fit_full(get_candidate_models()["logistic_regression"], X, y)
    path = tmp_path_factory.mktemp("models") / "champion.joblib"
    save_champion(pipe, path, {"model": "logistic_regression", "threshold": 0.35})
    return path


@pytest.fixture
def sample_customer() -> dict:
    """Cliente de alto risco típico: novo, fibra, contrato mensal, cheque eletrônico."""
    return {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 2,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 95.7,
        "TotalCharges": 191.4,
    }


@pytest.fixture
def loyal_customer(sample_customer) -> dict:
    """Cliente de baixo risco: antigo, contrato de 2 anos, débito automático."""
    return {
        **sample_customer,
        "tenure": 70,
        "Contract": "Two year",
        "PaymentMethod": "Credit card (automatic)",
        "InternetService": "DSL",
        "OnlineSecurity": "Yes",
        "TechSupport": "Yes",
        "MonthlyCharges": 55.0,
        "TotalCharges": 3850.0,
    }
