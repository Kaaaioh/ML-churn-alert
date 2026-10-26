"""Schemas Pydantic da API: contrato de entrada e saída.

Analogia web: é o Zod do projeto. O FastAPI valida o JSON contra estes modelos antes
de chamar o handler e gera a documentação do Swagger a partir deles. Valor fora do
domínio (ex.: `Contract = "Mensal"`) devolve 422 com a explicação, sem tocar no modelo.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

YesNo = Literal["Yes", "No"]
YesNoInternet = Literal["Yes", "No", "No internet service"]


class CustomerFeatures(BaseModel):
    """Dados de um cliente, com os mesmos nomes e domínios do dataset de treino."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
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
        }
    )

    gender: Literal["Female", "Male"]
    SeniorCitizen: Literal[0, 1] = Field(description="1 se o cliente tem 65 anos ou mais")
    Partner: YesNo
    Dependents: YesNo
    tenure: int = Field(ge=0, le=120, description="Meses como cliente")
    PhoneService: YesNo
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: YesNoInternet
    OnlineBackup: YesNoInternet
    DeviceProtection: YesNoInternet
    TechSupport: YesNoInternet
    StreamingTV: YesNoInternet
    StreamingMovies: YesNoInternet
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: YesNo
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]
    MonthlyCharges: float = Field(ge=0, description="Mensalidade atual")
    TotalCharges: float = Field(ge=0, description="Total já pago pelo cliente")


class PredictionResponse(BaseModel):
    churn_probability: float = Field(ge=0, le=1, description="Probabilidade de cancelamento")
    churn_label: int = Field(
        description="1 = risco de churn (probabilidade >= limiar), 0 = sem risco"
    )
    risk_tier: Literal["baixo", "medio", "alto"]
    suggested_action: str
    threshold: float = Field(description="Limiar de negócio usado para definir churn_label")


class BatchPredictionRequest(BaseModel):
    customers: list[CustomerFeatures] = Field(min_length=1, max_length=1000)


class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]
    count: int


class HealthResponse(BaseModel):
    status: Literal["ok"]
    model_loaded: bool
    model_name: str | None = None


class ModelInfoResponse(BaseModel):
    model_name: str
    threshold: float
    trained_on: str | None = None
    sklearn_version: str | None = None
    cv_f1_mean: float | None = None
    cv_roc_auc_mean: float | None = None
    features: list[str]
