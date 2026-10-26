# Model Card — ML Churn Alert v0.1.0

> Documento de transparência do modelo: o que ele é, para que serve, como foi avaliado, onde falha e que
> cuidados exige. Formato inspirado em Mitchell et al. (2019), *Model Cards for Model Reporting*.

## 1. Detalhes do modelo

| Item | Valor |
|---|---|
| Nome | ML Churn Alert — campeão da Fase 1 |
| Versão | 0.1.0 (treinado em 2026-10-26) |
| Tipo | Classificação binária (churn = 1 / permanece = 0) |
| Algoritmo | Regressão Logística (`LogisticRegression`, `class_weight="balanced"`, C = 1,0) |
| Pré-processamento | `ColumnTransformer`: `StandardScaler` nas numéricas, `OneHotEncoder(handle_unknown="ignore")` nas categóricas, embutido no `Pipeline` |
| Framework | scikit-learn 1.9.1, Python 3.12 |
| Artefato | `models/champion.joblib` + `models/champion.json` (métricas, limiar, premissas) |
| Entrada | 19 features cadastrais e contratuais de um cliente (ver `src/churn_alert/schemas.py`) |
| Saída | probabilidade de churn, rótulo (limiar de negócio), faixa de risco (baixo / medio / alto), ação sugerida |
| Candidatos comparados | Regressão Logística, Random Forest (300 árvores), MLP (64, 32) — ver `docs/experimentos.md` |

**Por que a Regressão Logística venceu:** empate técnico com o Random Forest em F1 (0,628 vs 0,632, diferença
menor que o desvio padrão da validação cruzada). A regra de desempate, definida antes de ver os resultados,
favorece o modelo mais simples: treina em 0,08 s, é interpretável (pesos legíveis) e mais fácil de manter.

## 2. Uso pretendido

- **Uso primário:** priorizar clientes de uma operadora de telecom para ações de retenção, via API
  (`POST /predict`, `POST /predict/batch`), pontuando a base mensalmente ou sob consulta do CRM.
- **Usuários:** time de retenção / CRM e analistas de negócio.
- **Fora de escopo:** decisões automáticas sem revisão humana (ex.: cancelar benefícios, negar serviço);
  uso em outra operadora, país ou período sem re-treino e revalidação; qualquer uso que trate a
  probabilidade como "certeza" sobre um indivíduo.

## 3. Dados de treino

- **Fonte:** Telco Customer Churn (IBM, dataset público). 7.043 clientes, 1 linha por cliente, foto estática.
- **Alvo:** `Churn` (26,5% positivos — classes desbalanceadas).
- **Split:** 80% treino (5.634) / 20% teste (1.409), estratificado, `random_state=42`. O teste nunca foi usado
  para escolher modelo ou hiperparâmetros.
- **Limpeza:** `TotalCharges` convertido de texto para número (11 clientes com `tenure=0` recebem 0);
  `customerID` removido.
- **O que o dataset não tem:** histórico de uso ao longo do tempo, reclamações/atendimento, região,
  concorrência, motivo do cancelamento. O modelo só enxerga o cadastro e o contrato.

## 4. Métricas e resultados

Métrica principal: **F1 da classe churn**. Complementar: **AUC-ROC**. Acurácia é reportada só para contexto,
porque um modelo que nunca prevê churn já teria 73,5%.

### Validação cruzada (5 folds, treino)

| Modelo | F1 | AUC-ROC | Recall | Precision |
|---|---|---|---|---|
| **Regressão Logística (campeão)** | 0,628 ± 0,023 | 0,846 ± 0,012 | 0,801 ± 0,038 | 0,517 ± 0,019 |
| Random Forest | 0,632 ± 0,014 | 0,845 ± 0,009 | 0,766 ± 0,028 | 0,539 ± 0,012 |
| MLP | 0,579 ± 0,026 | 0,839 ± 0,012 | 0,513 ± 0,038 | 0,667 ± 0,028 |

### Conjunto de teste (1.409 clientes nunca vistos)

| Limiar | Acurácia | Precision | Recall | F1 | AUC-ROC |
|---|---|---|---|---|---|
| 0,50 (padrão) | 0,738 | 0,504 | 0,783 | 0,614 | 0,842 |
| **0,33 (negócio, em produção)** | 0,669 | 0,440 | 0,917 | 0,595 | 0,842 |

**Leitura:** com o limiar de negócio o modelo encontra 92 de cada 100 clientes que vão cancelar, ao custo de
abordar também clientes que ficariam (precision 44%). Essa troca é intencional: pelas premissas de custo
(abordagem R$ 20, cliente retido R$ 390, 30% de sucesso), deixar um cliente sair custa mais que uma oferta
desnecessária. O lucro esperado da campanha no teste sobe 8% em relação ao limiar 0,5.

### Faixas de risco no teste

| Faixa | Critério | % dos clientes | Churn real na faixa |
|---|---|---|---|
| baixo | p < 0,33 | 44,7% | 4,9% |
| medio | 0,33 ≤ p < 0,60 | 20,6% | 27,2% |
| alto | p ≥ 0,60 | 34,7% | 54,0% |

As faixas separam bem os grupos: um cliente "alto" tem 11 vezes mais chance de cancelar que um "baixo".

## 5. Análise de vieses e desempenho por grupo (teste)

| Grupo | n | Churn real | Taxa de alerta | Recall | Precision | AUC |
|---|---|---|---|---|---|---|
| Mulheres | 687 | 28,1% | 55,3% | 0,907 | 0,461 | 0,838 |
| Homens | 722 | 25,1% | 55,3% | 0,928 | 0,421 | 0,847 |
| Não idosos | 1.187 | 23,3% | 50,5% | 0,895 | 0,412 | 0,842 |
| Idosos (65+) | 222 | 44,1% | **80,6%** | 0,980 | 0,536 | 0,781 |
| Contrato mensal | 773 | 42,6% | 89,3% | 0,979 | 0,467 | 0,742 |
| Contrato 1 ano | 300 | 12,0% | 29,0% | 0,583 | 0,241 | 0,743 |
| Contrato 2 anos | 336 | 2,7% | 0,6% | **0,000** | 0,000 | 0,759 |

- **Gênero:** sem evidência de viés. Taxa de alerta idêntica e recall equivalente nos dois grupos.
- **Idade:** idosos são alertados muito mais (81% vs 51%). Isso reflete a taxa de churn real maior nesse
  grupo (44% vs 23%), não um erro do modelo, mas tem consequência prática: quase todo idoso entra na campanha.
  Recomenda-se revisar se a abordagem padrão é adequada a esse público e monitorar reclamações.
- **Contratos longos:** o modelo praticamente nunca alerta clientes de 2 anos e **não detecta nenhum** dos 9
  churns desse grupo no teste. Para esse segmento, o modelo não serve como alerta; é uma limitação conhecida.

## 6. Calibração das probabilidades

As probabilidades são **superestimadas**: `class_weight="balanced"` desloca a saída para cima. No teste,
clientes com probabilidade prevista em torno de 0,46 cancelaram em 25% dos casos; em torno de 0,71, em 40%.
A **ordem** está correta (quem tem probabilidade maior cancela mais), então ranking e faixas funcionam.
Mas `churn_probability` **não deve ser lida como frequência real** ("46% de chance"). Se o negócio precisar
de probabilidades calibradas (ex.: para calcular valor esperado por cliente), aplicar
`CalibratedClassifierCV` é o próximo passo.

## 7. Limitações

1. Dataset pequeno, público e estático: desempenho em dados reais da operadora pode ser diferente.
2. Sem informação temporal: o modelo não sabe se o cliente está "piorando"; só vê a foto atual.
3. Premissas de custo (R$ 20 / R$ 390 / 30%) são hipotéticas. Alterá-las muda o limiar e o lucro esperado.
4. Sem ajuste fino de hiperparâmetros (decisão de escopo). `GridSearchCV` pode render ganhos marginais.
5. Multicolinearidade entre `tenure`, `MonthlyCharges` e `TotalCharges` torna coeficientes individuais pouco
   interpretáveis, embora não prejudique a predição.
6. Não detecta churn em contratos de 2 anos (ver seção 5).

## 8. Recomendações de uso e monitoramento

- Re-treinar a cada ciclo relevante (trimestral) ou quando a taxa de churn observada divergir da prevista
  por faixa (ex.: faixa "baixo" passando de 5% para 10%).
- Monitorar drift nas distribuições de entrada (proporção de fibra, contratos mensais, preços).
- Manter revisão humana nas ações de retenção; o modelo prioriza, não decide.
- Registrar predições e desfechos para permitir recalibração e auditoria de vieses.

## 9. Reprodutibilidade

`pip install -e ".[dev]"`, dataset em `data/raw/telco_customer_churn.csv`, executar
`notebooks/01_eda_baseline.ipynb` e `notebooks/02_modelos.ipynb` (seed 42 em todos os passos).
Os números deste documento são regenerados pelos notebooks e pelos metadados em `models/champion.json`.
