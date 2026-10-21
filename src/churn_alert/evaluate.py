"""Métricas de avaliação para classificação binária de churn.

Por que não só acurácia? Com ~26% de churn, um modelo que chuta "ninguém sai"
acerta 74% e é inútil. Por isso olhamos métricas que enxergam a classe positiva:

- precision: dos clientes que o modelo marcou como churn, quantos realmente saíram?
- recall:    dos clientes que realmente saíram, quantos o modelo pegou?
- f1:        média harmônica de precision e recall (métrica técnica principal).
- roc_auc:   capacidade de ordenar clientes por risco, independente do limiar.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(y_true, y_pred, y_proba=None) -> dict[str, float]:
    """Calcula o conjunto padrão de métricas do projeto.

    `y_proba` é a probabilidade da classe positiva (churn = 1). Se não for
    informada, `roc_auc` fica como NaN.
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba) if y_proba is not None else np.nan,
    }
    return {k: float(v) for k, v in metrics.items()}
