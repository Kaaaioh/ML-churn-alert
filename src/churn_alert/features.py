"""Pré-processamento de features para os modelos do Scikit-Learn.

Analogia web: o `ColumnTransformer` é um "roteador" que manda cada coluna para o
middleware certo (escalonamento para números, one-hot para categorias) e junta
tudo de volta numa matriz numérica. Como ele vive DENTRO do `Pipeline` do modelo,
o artefato salvo já sabe se pré-processar sozinho: a API só chama `predict_proba`.
"""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from churn_alert.config import CATEGORICAL_FEATURES, NUMERIC_FEATURES


def build_preprocessor(
    numeric_features: list[str] | None = None,
    categorical_features: list[str] | None = None,
) -> ColumnTransformer:
    """Monta o transformador de colunas usado por todos os modelos.

    - Numéricas → `StandardScaler` (média 0, desvio 1). Regressão logística e MLP
      precisam disso para convergir bem; árvores não se importam, mas não atrapalha.
    - Categóricas → `OneHotEncoder` com `handle_unknown="ignore"`: se a API receber
      uma categoria nunca vista no treino, ela vira um vetor de zeros em vez de
      derrubar a requisição.
    """
    numeric_features = numeric_features or NUMERIC_FEATURES
    categorical_features = categorical_features or CATEGORICAL_FEATURES

    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_features,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_pipeline(estimator, preprocessor: ColumnTransformer | None = None) -> Pipeline:
    """Empacota pré-processamento + estimador em um único objeto treinável/salvável."""
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor or build_preprocessor()),
            ("model", estimator),
        ]
    )
