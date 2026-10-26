"""Testes do ajuste de limiar por custo de negócio."""

import numpy as np

from churn_alert.threshold import BusinessCosts, best_threshold, expected_profit, profit_curve


def test_business_costs_derived_values():
    costs = BusinessCosts(contact_cost=10, retained_value=100, retention_rate=0.5)
    assert costs.gain_per_true_positive == 40  # 0.5 * 100 - 10
    assert costs.loss_per_false_positive == 10


def test_expected_profit_counts_tp_and_fp():
    costs = BusinessCosts(contact_cost=10, retained_value=100, retention_rate=0.5)
    y_true = np.array([1, 1, 0, 0])
    y_proba = np.array([0.9, 0.2, 0.8, 0.1])
    # limiar 0.5 → aborda índices 0 (TP) e 2 (FP): 1*40 - 1*10 = 30
    assert expected_profit(y_true, y_proba, 0.5, costs) == 30


def test_profit_curve_and_best_threshold_are_consistent():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, 500)
    y_proba = np.clip(y_true * 0.5 + rng.uniform(0, 0.6, 500), 0, 1)
    costs = BusinessCosts()
    curve = profit_curve(y_true, y_proba, costs)
    t = best_threshold(y_true, y_proba, costs)
    assert 0.05 <= t <= 0.95
    assert curve.loc[curve["threshold"] == t, "profit"].iloc[0] == curve["profit"].max()
