"""API de inferência (FastAPI).

Endpoints:
- GET  /health         → status da API e se o modelo está carregado
- GET  /model          → metadados do campeão (nome, limiar, métricas)
- POST /predict        → predição para um cliente
- POST /predict/batch  → predição para até 1000 clientes

Analogia web: `create_app()` é uma app factory (como em Express: `const app = createApp(deps)`),
o que permite injetar um modelo diferente nos testes. O `lifespan` faz o papel do "on boot":
carrega o artefato uma vez e o guarda em `app.state`.

Subir localmente:
    uvicorn churn_alert.api:app --reload
Documentação interativa: http://127.0.0.1:8000/docs
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request

from churn_alert import __version__
from churn_alert.predict import DEFAULT_MODEL_PATH, Predictor
from churn_alert.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    CustomerFeatures,
    HealthResponse,
    ModelInfoResponse,
    PredictionResponse,
)

MODEL_PATH_ENV = "CHURN_ALERT_MODEL_PATH"


def create_app(model_path: Path | str | None = None) -> FastAPI:
    """Monta a aplicação. `model_path` permite apontar para outro artefato (ex.: nos testes)."""
    resolved_path = Path(model_path or os.getenv(MODEL_PATH_ENV, DEFAULT_MODEL_PATH))

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Carrega o modelo no boot. Se falhar, a API sobe mesmo assim e /health avisa,
        # o que facilita diagnosticar em produção em vez de um crash silencioso.
        try:
            app.state.predictor = Predictor.load(resolved_path)
        except FileNotFoundError:
            app.state.predictor = None
        yield
        app.state.predictor = None

    app = FastAPI(
        title="ML Churn Alert",
        version=__version__,
        description=(
            "Serviço de predição de churn para telecom. Recebe os dados cadastrais e "
            "contratuais de um cliente e devolve a probabilidade de cancelamento, a faixa "
            "de risco e a ação sugerida para o time de retenção."
        ),
        lifespan=lifespan,
    )

    def get_predictor(request: Request) -> Predictor:
        predictor = getattr(request.app.state, "predictor", None)
        if predictor is None:
            raise HTTPException(
                status_code=503,
                detail=(
                    "Modelo não carregado. Gere models/champion.joblib (notebook 02) e reinicie."
                ),
            )
        return predictor

    @app.get("/health", response_model=HealthResponse, tags=["operacional"])
    def health(request: Request) -> HealthResponse:
        predictor = getattr(request.app.state, "predictor", None)
        return HealthResponse(
            status="ok",
            model_loaded=predictor is not None,
            model_name=predictor.model_name if predictor else None,
        )

    @app.get("/model", response_model=ModelInfoResponse, tags=["operacional"])
    def model_info(request: Request) -> ModelInfoResponse:
        predictor = get_predictor(request)
        meta = predictor.metadata
        return ModelInfoResponse(
            model_name=predictor.model_name,
            threshold=predictor.threshold,
            trained_on=meta.get("trained_on"),
            sklearn_version=meta.get("sklearn_version"),
            cv_f1_mean=meta.get("cv_f1_mean"),
            cv_roc_auc_mean=meta.get("cv_roc_auc_mean"),
            features=predictor.features,
        )

    @app.post("/predict", response_model=PredictionResponse, tags=["predição"])
    def predict(customer: CustomerFeatures, request: Request) -> PredictionResponse:
        predictor = get_predictor(request)
        result = predictor.predict_one(customer.model_dump())
        return PredictionResponse(**result.__dict__)

    @app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["predição"])
    def predict_batch(payload: BatchPredictionRequest, request: Request) -> BatchPredictionResponse:
        predictor = get_predictor(request)
        results = predictor.predict_batch([c.model_dump() for c in payload.customers])
        return BatchPredictionResponse(
            predictions=[PredictionResponse(**r.__dict__) for r in results],
            count=len(results),
        )

    return app


# Instância padrão usada pelo uvicorn: `uvicorn churn_alert.api:app`
app = create_app()
