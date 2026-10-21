# ML Churn Alert — Predição de Churn em Telecom

Tech Challenge Fase 1 · Pós-Tech FIAP Machine Learning Engineering (12MLET)

Uma operadora de telecomunicações está perdendo clientes. Este projeto constrói, do zero, um modelo que
classifica clientes com risco de cancelamento (churn) e o serve via API REST, seguindo boas práticas de
engenharia de software para ML.

> Estado atual: **Etapa 1 concluída** (EDA + baseline). Veja o checklist em [docs/progresso.md](docs/progresso.md).

## Estrutura

```
├── data/
│   ├── raw/          # dataset original (Telco Customer Churn, IBM)
│   └── processed/    # dados intermediários (gerados, fora do git)
├── docs/             # ML Canvas, registro de experimentos, figuras, Model Card
├── models/           # artefatos .joblib (fora do git)
├── notebooks/        # experimentação (EDA, baseline, comparação de modelos)
├── src/churn_alert/        # código produtivo: config, dados, features, avaliação, (api)
├── tests/            # testes pytest
└── pyproject.toml    # dependências e configuração
```

## Setup

Requer Python 3.11+.

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows  (Linux/macOS: source .venv/bin/activate)
pip install -e ".[dev]"
```

Dataset: baixe o CSV público da IBM e salve em `data/raw/telco_customer_churn.csv`.

```bash
curl -L -o data/raw/telco_customer_churn.csv https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv
```

## Como rodar

```bash
pytest                                     # testes
jupyter lab                                # abre os notebooks
jupyter nbconvert --to notebook --execute notebooks/01_eda_baseline.ipynb --inplace   # reexecuta a EDA
```

## Etapa 1 — resumo

- Dataset: 7.043 clientes, 20 features, alvo `Churn` com ~26,5% de positivos (desbalanceado).
- Limpeza: `TotalCharges` convertido para número (11 clientes novos com campo em branco → 0).
- Métricas: **F1 da classe churn** (principal) e **AUC-ROC** (complementar). Justificativa no [ML Canvas](docs/ml_canvas.md).
- Baseline: Regressão Logística com `class_weight="balanced"` dentro de um `Pipeline` do scikit-learn.
  Resultados no notebook e em [docs/experimentos.md](docs/experimentos.md).
- Seed fixa: `42` em splits, modelos e validação cruzada.

## Convenções

- Notebooks exploram; funções reutilizáveis vivem em `src/churn_alert/` e são importadas pelos notebooks.
- O pré-processamento fica **dentro** do `Pipeline` salvo, então a API só precisa carregar um artefato.
- Comentários em português, identificadores em inglês.
