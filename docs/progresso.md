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
- [ ] Random Forest (ou árvore de decisão)
- [ ] MLPClassifier
- [ ] Validação cruzada 5-fold para todos os modelos
- [ ] Tabela comparativa Linear vs Árvore vs MLP
- [ ] Modelo campeão escolhido e salvo em `models/`

## Etapa 3 — Engenharia de Software e API
- [ ] Módulo de predição em `src/` carregando o campeão
- [ ] API FastAPI com `/health` e `/predict`
- [ ] Testes pytest da API (status e predição)

## Etapa 4 — Documentação e Entrega
- [ ] Model Card (`docs/model_card.md`)
- [ ] README final (instalação, execução, API)
- [ ] Vídeo STAR (5 min)
- [ ] (Opcional) Deploy em nuvem
