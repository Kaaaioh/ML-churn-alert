# Progresso do Tech Challenge Fase 1

Prazo de entrega: **3 de novembro de 2026**.

## Etapa 1 — Entendimento e Preparação
- [x] Estrutura do repositório (`src/`, `data/`, `models/`, `tests/`, `notebooks/`, `docs/`)
- [x] `pyproject.toml` + `.gitignore` para Python/ML
- [x] Dataset Telco Customer Churn em `data/raw/`
- [x] ML Canvas (`docs/ml_canvas.md`)
- [x] EDA: volume, qualidade, distribuição das features (`notebooks/01_eda_baseline.ipynb`)
- [x] Métricas definidas: F1 (técnica principal), AUC-ROC (complementar), métricas de negócio no canvas
- [x] Baseline Regressão Logística treinado, avaliado e salvo (`models/baseline_logreg.joblib`)
- [x] Registro de experimentos iniciado (`docs/experimentos.md`)

## Etapa 2 — Modelagem e Avaliação
- [x] Random Forest (`RandomForestClassifier`, 300 árvores, `class_weight=balanced`)
- [x] MLPClassifier (2 camadas ocultas, `early_stopping`)
- [x] Validação cruzada 5-fold estratificada para os três modelos (`src/churn_alert/train.py`)
- [x] Tabela comparativa Linear vs Árvore vs MLP (`notebooks/02_modelos.ipynb`, `docs/experimentos.md`)
- [x] Modelo campeão escolhido por regra pré-definida e salvo em `models/champion.joblib` + `champion.json`
- [x] Limiar de decisão ajustado por custo de negócio (`src/churn_alert/threshold.py`)

## Etapa 3 — Engenharia de Software e API
- [x] Módulo de predição em `src/churn_alert/predict.py` (carrega campeão + metadados, faixas de risco, ação sugerida)
- [x] Schemas Pydantic com domínios validados (`src/churn_alert/schemas.py`)
- [x] API FastAPI com `/health`, `/model`, `/predict` e `/predict/batch` (`src/churn_alert/api.py`)
- [x] Testes pytest da API e do preditor (30 testes no total, com modelo sintético em pasta temporária)

## Etapa 4 — Documentação e Entrega
- [x] Model Card com métricas, faixas, análise de vieses por grupo e calibração (`docs/model_card.md`)
- [x] README final (problema, resultados, arquitetura, setup, API, limitações)
- [x] `requirements.txt` com versões fixas + CI no GitHub Actions (lint + testes)
- [x] Roteiro do vídeo STAR com falas, telas e tempos (`docs/roteiro_video_star.md`)
- [ ] Gravar o vídeo STAR (5 min)
- [ ] Publicar o repositório no GitHub
- [ ] (Opcional) Deploy em nuvem
