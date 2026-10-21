"""Configurações centrais do projeto: seed, caminhos e definição das colunas.

Analogia web: é o nosso `config.js` / `.env` tipado. Tudo que o resto do código
precisa saber sobre "onde estão as coisas" e "quais colunas existem" mora aqui.
"""

from pathlib import Path

# Seed global para reprodutibilidade. Use SEMPRE que algo envolver aleatoriedade
# (split, modelos, validação cruzada).
SEED = 42

# Caminhos (resolvidos a partir da raiz do projeto, independente de onde o script roda)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "telco_customer_churn.csv"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
DOCS_DIR = PROJECT_ROOT / "docs"

# Coluna alvo e identificador
TARGET = "Churn"
ID_COLUMN = "customerID"

# Colunas numéricas (SeniorCitizen já vem como 0/1 no CSV)
NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]

# Colunas categóricas (texto) que serão transformadas com one-hot encoding
CATEGORICAL_FEATURES = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
