# Registro de experimentos

| data | etapa | modelo | cv_f1 | cv_roc_auc | teste_f1 | teste_roc_auc | teste_recall | teste_precision | artefato |
|---|---|---|---|---|---|---|---|---|---|
| 2026-10-21 | 1 | LogisticRegression(class_weight=balanced, C=1.0) | 0.628 ± 0.023 | 0.846 ± 0.012 | 0.614 | 0.842 | 0.783 | 0.504 | baseline_logreg.joblib |
| 2026-10-26 | 2 | LogisticRegression ★ campeão | 0.628 ± 0.023 | 0.846 ± 0.012 | 0.614 | 0.842 | 0.783 | 0.504 | champion.joblib |
| 2026-10-26 | 2 | RandomForestClassifier | 0.632 ± 0.014 | 0.845 ± 0.009 | 0.634 | 0.841 | 0.778 | 0.535 | - |
| 2026-10-26 | 2 | MLPClassifier | 0.579 ± 0.026 | 0.839 ± 0.012 | 0.553 | 0.840 | 0.476 | 0.659 | - |
