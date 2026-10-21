"""Carga e limpeza do dataset Telco Customer Churn.

Analogia web: este módulo é a camada de "repository + sanitização de input".
Ele lê a fonte bruta, corrige tipos e devolve um DataFrame limpo e previsível
para o resto do pipeline. Nenhuma função aqui guarda estado global.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from churn_alert.config import ALL_FEATURES, ID_COLUMN, RAW_DATA_PATH, SEED, TARGET


def load_raw(path: Path | str = RAW_DATA_PATH) -> pd.DataFrame:
    """Lê o CSV bruto exatamente como veio da fonte (sem nenhuma transformação)."""
    return pd.read_csv(path)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica a limpeza mínima necessária para o dataset ficar modelável.

    Regras (descobertas na EDA):
    1. `TotalCharges` vem como texto porque 11 clientes novos (tenure = 0) têm o
       campo em branco. Convertemos para número e preenchemos os brancos com 0,
       que é o valor economicamente correto (ainda não pagaram nada).
    2. `customerID` é só um identificador: não carrega informação preditiva e,
       se deixado, o modelo tentaria "decorar" clientes. Removemos.
    3. Alvo `Churn` vira inteiro: 1 = cancelou, 0 = ficou.
    4. Espaços extras em colunas de texto são removidos por segurança.
    """
    out = df.copy()

    out["TotalCharges"] = pd.to_numeric(out["TotalCharges"], errors="coerce").fillna(0.0)

    if ID_COLUMN in out.columns:
        out = out.drop(columns=[ID_COLUMN])

    # pandas 2 usa dtype `object` para texto; pandas 3 usa `str`. As funções abaixo cobrem ambos.
    if not pd.api.types.is_numeric_dtype(out[TARGET]):
        out[TARGET] = out[TARGET].str.strip().map({"Yes": 1, "No": 0}).astype(int)

    text_cols = [c for c in out.columns if pd.api.types.is_string_dtype(out[c])]
    for col in text_cols:
        out[col] = out[col].str.strip()

    return out


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separa X (features) de y (alvo), garantindo a ordem de colunas esperada."""
    X = df[ALL_FEATURES].copy()
    y = df[TARGET].copy()
    return X, y


def train_test_split_stratified(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    seed: int = SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split treino/teste estratificado: mantém a mesma proporção de churn nos dois lados.

    Por que estratificar? Só ~26% dos clientes dão churn. Sem estratificação, um
    split aleatório poderia deixar o conjunto de teste com proporção diferente e
    as métricas ficariam enviesadas.
    """
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)


def load_clean(path: Path | str = RAW_DATA_PATH) -> pd.DataFrame:
    """Atalho: lê o bruto e já devolve limpo."""
    return clean(load_raw(path))
