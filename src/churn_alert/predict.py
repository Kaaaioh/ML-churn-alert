"""Carga do modelo campeão e predição com faixas de risco e ação sugerida.

Analogia web: `Predictor` é o "service" que a camada HTTP (api.py) chama. Ele carrega o
artefato uma vez (como abrir uma conexão com o banco no boot) e expõe métodos puros de
predição. A API não sabe nada de scikit-learn; só conversa com esta classe.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from churn_alert.config import ALL_FEATURES, MODELS_DIR

DEFAULT_MODEL_PATH = MODELS_DIR / "champion.joblib"

RiskTier = Literal["baixo", "medio", "alto"]

# Limite acima do qual o risco passa de "médio" para "alto". O limite entre "baixo" e
# "médio" é o limiar de negócio salvo nos metadados do modelo.
HIGH_RISK_THRESHOLD = 0.60

SUGGESTED_ACTIONS: dict[str, str] = {
    "baixo": "Nenhuma ação; manter monitoramento mensal.",
    "medio": "Incluir na campanha de retenção (oferta padrão).",
    "alto": "Contato proativo do time de retenção com oferta personalizada.",
}


@dataclass(frozen=True)
class Prediction:
    """Resultado de uma predição individual."""

    churn_probability: float
    churn_label: int
    risk_tier: RiskTier
    suggested_action: str
    threshold: float


class Predictor:
    """Envolve o pipeline treinado e seus metadados para servir predições."""

    def __init__(self, pipeline: Pipeline, metadata: dict) -> None:
        self.pipeline = pipeline
        self.metadata = metadata
        self.threshold = float(metadata.get("threshold", 0.5))
        self.features: list[str] = list(metadata.get("features", ALL_FEATURES))

    @classmethod
    def load(cls, model_path: Path | str = DEFAULT_MODEL_PATH) -> Predictor:
        """Lê o `.joblib` e o `.json` de metadados que fica ao lado dele."""
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Modelo não encontrado em {model_path}. "
                "Execute o notebook 02_modelos.ipynb para gerar o campeão."
            )
        pipeline = joblib.load(model_path)
        meta_path = model_path.with_suffix(".json")
        metadata = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        return cls(pipeline, metadata)

    @property
    def model_name(self) -> str:
        return str(
            self.metadata.get("model", self.pipeline.named_steps["model"].__class__.__name__)
        )

    def risk_tier(self, probability: float) -> RiskTier:
        """Converte probabilidade em faixa de risco acionável pelo time de retenção."""
        if probability >= HIGH_RISK_THRESHOLD:
            return "alto"
        if probability >= self.threshold:
            return "medio"
        return "baixo"

    def predict_proba(self, records: pd.DataFrame) -> np.ndarray:
        """Probabilidade de churn para cada linha, na ordem das features esperadas."""
        X = records[self.features]
        return self.pipeline.predict_proba(X)[:, 1]

    def predict_batch(self, records: list[dict]) -> list[Prediction]:
        """Predição para uma lista de clientes (cada um como dict feature → valor)."""
        if not records:
            return []
        df = pd.DataFrame.from_records(records)
        probas = self.predict_proba(df)
        return [self._to_prediction(float(p)) for p in probas]

    def predict_one(self, record: dict) -> Prediction:
        return self.predict_batch([record])[0]

    def _to_prediction(self, probability: float) -> Prediction:
        tier = self.risk_tier(probability)
        return Prediction(
            churn_probability=round(probability, 4),
            churn_label=int(probability >= self.threshold),
            risk_tier=tier,
            suggested_action=SUGGESTED_ACTIONS[tier],
            threshold=self.threshold,
        )
