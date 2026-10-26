"""Ajuste do limiar de decisão com base em custo/benefício de negócio.

Por padrão, `predict()` do scikit-learn usa limiar 0,5: probabilidade acima disso vira
"churn". Mas 0,5 não tem nada a ver com o negócio. Se abordar um cliente custa pouco e
perder um cliente custa muito, vale abordar mesmo com probabilidade bem menor que 0,5.

Analogia web: é como escolher o timeout de um retry. O valor "padrão" da biblioteca não
sabe quanto custa cada falha no seu sistema; quem sabe é você.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BusinessCosts:
    """Premissas econômicas de uma campanha de retenção (valores em R$ por cliente).

    - contact_cost: custo de abordar um cliente (ligação, desconto oferecido, tempo do time).
    - retained_value: quanto vale manter um cliente que ia sair (ex.: 6 mensalidades de margem).
    - retention_rate: fração dos clientes abordados que de fato ficam após a oferta.
    """

    contact_cost: float = 20.0
    retained_value: float = 390.0
    retention_rate: float = 0.30

    @property
    def gain_per_true_positive(self) -> float:
        """Ganho esperado ao abordar um cliente que realmente ia sair."""
        return self.retention_rate * self.retained_value - self.contact_cost

    @property
    def loss_per_false_positive(self) -> float:
        """Custo de abordar um cliente que ficaria de qualquer jeito."""
        return self.contact_cost


def expected_profit(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float,
    costs: BusinessCosts,
) -> float:
    """Lucro esperado da campanha se abordarmos todos com probabilidade >= threshold."""
    y_true = np.asarray(y_true)
    contacted = np.asarray(y_proba) >= threshold
    tp = int(np.sum(contacted & (y_true == 1)))
    fp = int(np.sum(contacted & (y_true == 0)))
    return tp * costs.gain_per_true_positive - fp * costs.loss_per_false_positive


def profit_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    costs: BusinessCosts,
    thresholds: np.ndarray | None = None,
) -> pd.DataFrame:
    """Lucro esperado para uma grade de limiares. Base para o gráfico e para `best_threshold`."""
    thresholds = thresholds if thresholds is not None else np.linspace(0.05, 0.95, 91)
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    rows = []
    for t in thresholds:
        contacted = y_proba >= t
        tp = int(np.sum(contacted & (y_true == 1)))
        fp = int(np.sum(contacted & (y_true == 0)))
        fn = int(np.sum(~contacted & (y_true == 1)))
        rows.append(
            {
                "threshold": float(t),
                "contacted": int(contacted.sum()),
                "true_positives": tp,
                "false_positives": fp,
                "false_negatives": fn,
                "recall": tp / (tp + fn) if (tp + fn) else 0.0,
                "precision": tp / (tp + fp) if (tp + fp) else 0.0,
                "profit": tp * costs.gain_per_true_positive - fp * costs.loss_per_false_positive,
            }
        )
    return pd.DataFrame(rows)


def best_threshold(y_true: np.ndarray, y_proba: np.ndarray, costs: BusinessCosts) -> float:
    """Limiar que maximiza o lucro esperado na grade padrão."""
    curve = profit_curve(y_true, y_proba, costs)
    return float(curve.loc[curve["profit"].idxmax(), "threshold"])
