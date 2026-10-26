# ML Churn Alert — Predição de Churn em Telecom

Tech Challenge Fase 1 · Pós-Tech FIAP Machine Learning Engineering (12MLET)

Uma operadora de telecomunicações está perdendo clientes. Este projeto constrói, do zero, um modelo que
classifica clientes com risco de cancelamento (churn) e o serve via API REST, seguindo boas práticas de
engenharia de software para ML.

> Estado atual: **Etapa 3 concluída** (API de inferência funcional e testada). Veja o checklist em [docs/progresso.md](docs/progresso.md).

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
jupyter nbconvert --to notebook --execute notebooks/02_modelos.ipynb --inplace        # reexecuta a comparação
uvicorn churn_alert.api:app --reload       # sobe a API em http://127.0.0.1:8000 (docs em /docs)
```

## API de inferência

| Método | Rota | O que faz |
|---|---|---|
| GET | `/health` | Status da API e se o modelo está carregado |
| GET | `/model` | Metadados do campeão: nome, limiar, métricas, features |
| POST | `/predict` | Predição para um cliente |
| POST | `/predict/batch` | Predição para até 1.000 clientes |

Exemplo:

```bash
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d '{
  "gender": "Female", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No", "tenure": 2,
  "PhoneService": "Yes", "MultipleLines": "No", "InternetService": "Fiber optic",
  "OnlineSecurity": "No", "OnlineBackup": "No", "DeviceProtection": "No", "TechSupport": "No",
  "StreamingTV": "Yes", "StreamingMovies": "Yes", "Contract": "Month-to-month",
  "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check",
  "MonthlyCharges": 95.7, "TotalCharges": 191.4
}'
```

Resposta:

```json
{
  "churn_probability": 0.893,
  "churn_label": 1,
  "risk_tier": "alto",
  "suggested_action": "Contato proativo do time de retenção com oferta personalizada.",
  "threshold": 0.33
}
```

- A resposta vai além do sim/não: `risk_tier` (baixo, medio, alto) e `suggested_action` traduzem a probabilidade em decisão para o time de retenção. O corte baixo/medio é o limiar de negócio salvo em `models/champion.json`; o corte medio/alto é 0,60.
- Entrada validada com Pydantic: categoria fora do domínio ou número negativo retorna `422` com a explicação, sem tocar no modelo.
- O caminho do artefato pode ser trocado pela variável de ambiente `CHURN_ALERT_MODEL_PATH`.


## Etapa 1 — resumo

- Dataset: 7.043 clientes, 20 features, alvo `Churn` com ~26,5% de positivos (desbalanceado).
- Limpeza: `TotalCharges` convertido para número (11 clientes novos com campo em branco → 0).
- Métricas: **F1 da classe churn** (principal) e **AUC-ROC** (complementar). Justificativa no [ML Canvas](docs/ml_canvas.md).
- Baseline: Regressão Logística com `class_weight="balanced"` dentro de um `Pipeline` do scikit-learn.
  Resultados no notebook e em [docs/experimentos.md](docs/experimentos.md).
- Seed fixa: `42` em splits, modelos e validação cruzada.

## Etapa 2 — resumo

Três candidatos treinados com o mesmo pré-processamento e a mesma validação cruzada estratificada (5 folds):

| Modelo | F1 (CV) | AUC-ROC (CV) | F1 (teste) |
|---|---|---|---|
| Regressão Logística ★ | 0,628 ± 0,023 | 0,846 ± 0,012 | 0,614 |
| Random Forest | 0,632 ± 0,014 | 0,845 ± 0,009 | 0,634 |
| MLP | 0,579 ± 0,026 | 0,839 ± 0,012 | 0,553 |

- Regra de escolha definida **antes** de ver os números: maior F1 em CV; empate técnico (< 0,005) vai para o modelo mais simples. Random Forest e Regressão Logística empataram, e a Regressão Logística foi escolhida por ser mais barata, mais rápida e interpretável.
- **Limiar de decisão por custo de negócio**: em vez do 0,5 padrão, o corte foi ajustado para maximizar o lucro esperado de uma campanha de retenção (premissas em `models/champion.json`). Resultado: limiar 0,33, recall de 0,92 e +8% de lucro esperado no teste.
- Artefatos: `models/champion.joblib` (pipeline completo) e `models/champion.json` (métricas, limiar, premissas). Figuras em `docs/figures/07` a `10`.

## Convenções

- Notebooks exploram; funções reutilizáveis vivem em `src/churn_alert/` e são importadas pelos notebooks.
- O pré-processamento fica **dentro** do `Pipeline` salvo, então a API só precisa carregar um artefato.
- Comentários em português, identificadores em inglês.
