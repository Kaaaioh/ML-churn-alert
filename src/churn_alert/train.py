"""Treino, validação cruzada e comparação dos modelos candidatos.

Analogia web: `get_candidate_models()` é um registry de "handlers" intercambiáveis.
Todos recebem o mesmo pré-processamento (via `build_pipeline`) e são avaliados pela
mesma validação cruzada, então a comparação é justa: só o algoritmo muda.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline

from churn_alert.config import SEED
from churn_alert.features import build_pipeline

SCORING = ["f1", "roc_auc", "recall", "precision"]


def get_candidate_models(seed: int = SEED) -> dict[str, BaseEstimator]:
    """Os três candidatos exigidos pelo enunciado: linear, árvore (ensemble) e rede neural.

    - LogisticRegression: baseline da Etapa 1. `class_weight="balanced"` compensa o
      desbalanceamento.
    - RandomForestClassifier: comitê de 300 árvores, cada uma vendo amostras e features diferentes.
      `min_samples_leaf=5` evita folhas com 1 cliente (overfitting). Também aceita `class_weight`.
    - MLPClassifier: rede neural pequena (2 camadas ocultas). Não aceita `class_weight`; o
      desbalanceamento é tratado depois, pelo ajuste de limiar. `early_stopping` interrompe o
      treino quando a validação interna para de melhorar.
    """
    return {
        "logistic_regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=seed
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=seed,
            n_jobs=1,
        ),
        "mlp": MLPClassifier(
            hidden_layer_sizes=(64, 32),
            alpha=1e-3,
            max_iter=500,
            early_stopping=True,
            random_state=seed,
        ),
    }


def cross_validate_pipeline(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    seed: int = SEED,
) -> dict[str, float]:
    """Validação cruzada estratificada. Devolve média e desvio de cada métrica + tempo de treino."""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores = cross_validate(pipeline, X, y, cv=cv, scoring=SCORING, n_jobs=1)
    out: dict[str, float] = {}
    for metric in SCORING:
        vals = scores[f"test_{metric}"]
        out[f"{metric}_mean"] = float(np.mean(vals))
        out[f"{metric}_std"] = float(np.std(vals))
    out["fit_time_s"] = float(np.mean(scores["fit_time"]))
    return out


def compare_models(
    models: dict[str, BaseEstimator],
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    seed: int = SEED,
) -> pd.DataFrame:
    """Roda a validação cruzada para cada candidato e devolve uma tabela (1 linha por modelo)."""
    rows = []
    for name, estimator in models.items():
        pipeline = build_pipeline(estimator)
        result = cross_validate_pipeline(pipeline, X, y, n_splits=n_splits, seed=seed)
        rows.append({"model": name, **result})
    return pd.DataFrame(rows).set_index("model")


def select_champion(
    results: pd.DataFrame, metric: str = "f1_mean", tolerance: float = 0.005
) -> str:
    """Escolhe o campeão pela métrica principal.

    Regra de desempate: se dois modelos estão a menos de `tolerance` um do outro, vence o que
    aparece primeiro na tabela (o mais simples), porque modelo mais simples é mais barato de
    manter e mais fácil de explicar.
    """
    best_score = results[metric].max()
    candidates = results[results[metric] >= best_score - tolerance]
    return str(candidates.index[0])


def fit_full(estimator: BaseEstimator, X: pd.DataFrame, y: pd.Series) -> Pipeline:
    """Treina o pipeline completo (pré-processamento + modelo) em todo o conjunto informado."""
    pipeline = build_pipeline(estimator)
    pipeline.fit(X, y)
    return pipeline


def save_champion(
    pipeline: Pipeline,
    path: Path,
    metadata: dict,
) -> tuple[Path, Path]:
    """Salva o artefato `.joblib` e um `.json` ao lado com metadados (modelo, métricas, limiar).

    O JSON é o "package.json" do modelo: a API lê dele o limiar de decisão e a versão,
    sem precisar desserializar o pipeline para saber o que ele é.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)
    meta = {
        "trained_on": date.today().isoformat(),
        "sklearn_version": sklearn.__version__,
        "artifact": path.name,
        **metadata,
    }
    meta_path = path.with_suffix(".json")
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return path, meta_path
