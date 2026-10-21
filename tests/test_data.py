"""Testes das funções de carga e limpeza de dados."""

import pandas as pd
import pytest

from churn_alert.config import ALL_FEATURES, RAW_DATA_PATH, TARGET
from churn_alert.data import clean, load_raw, split_features_target, train_test_split_stratified


@pytest.fixture
def raw_sample() -> pd.DataFrame:
    """Amostra sintética que reproduz os problemas do CSV real (TotalCharges em branco)."""
    return pd.DataFrame(
        {
            "customerID": ["0001-A", "0002-B", "0003-C"],
            "gender": ["Female", "Male", "Male"],
            "SeniorCitizen": [0, 1, 0],
            "Partner": ["Yes", "No", "No"],
            "Dependents": ["No", "No", "Yes"],
            "tenure": [0, 34, 2],
            "PhoneService": ["No", "Yes", "Yes"],
            "MultipleLines": ["No phone service", "No", "No"],
            "InternetService": ["DSL", "DSL", "Fiber optic"],
            "OnlineSecurity": ["No", "Yes", "No"],
            "OnlineBackup": ["Yes", "No", "No"],
            "DeviceProtection": ["No", "Yes", "No"],
            "TechSupport": ["No", "No", "No"],
            "StreamingTV": ["No", "No", "Yes"],
            "StreamingMovies": ["No", "No", "Yes"],
            "Contract": ["Month-to-month", "One year", "Month-to-month"],
            "PaperlessBilling": ["Yes", "No", "Yes"],
            "PaymentMethod": ["Electronic check", "Mailed check", "Electronic check"],
            "MonthlyCharges": [29.85, 56.95, 70.70],
            "TotalCharges": [" ", "1889.5", "151.65"],
            "Churn": ["No", "No", "Yes "],
        }
    )


def test_clean_converts_total_charges_and_fills_blank(raw_sample):
    df = clean(raw_sample)
    assert pd.api.types.is_float_dtype(df["TotalCharges"])
    assert df["TotalCharges"].iloc[0] == 0.0
    assert df["TotalCharges"].iloc[1] == pytest.approx(1889.5)


def test_clean_drops_id_and_binarizes_target(raw_sample):
    df = clean(raw_sample)
    assert "customerID" not in df.columns
    assert df[TARGET].tolist() == [0, 0, 1]
    assert pd.api.types.is_integer_dtype(df[TARGET])


def test_clean_does_not_mutate_input(raw_sample):
    before = raw_sample.copy()
    clean(raw_sample)
    pd.testing.assert_frame_equal(raw_sample, before)


def test_split_features_target_returns_expected_columns(raw_sample):
    X, y = split_features_target(clean(raw_sample))
    assert list(X.columns) == ALL_FEATURES
    assert len(X) == len(y) == 3


def test_stratified_split_keeps_churn_ratio():
    pytest.importorskip("sklearn")
    if not RAW_DATA_PATH.exists():
        pytest.skip("dataset bruto não encontrado em data/raw")
    X, y = split_features_target(clean(load_raw()))
    X_tr, X_te, y_tr, y_te = train_test_split_stratified(X, y, test_size=0.2)
    assert abs(y_tr.mean() - y_te.mean()) < 0.01
    assert len(X_tr) + len(X_te) == len(X)
