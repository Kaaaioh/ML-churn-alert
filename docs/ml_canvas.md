# ML Canvas — Predição de Churn em Telecom

> Preenchido na Etapa 1 do Tech Challenge Fase 1 (FIAP 12MLET).
> O ML Canvas organiza, em uma página, **por que** o modelo existe, **para quem** e **como** saberemos se funcionou.

## 1. Proposta de valor

Reduzir a perda de receita recorrente identificando, **antes do cancelamento**, os clientes com maior
risco de churn, para que o time de retenção atue de forma direcionada (oferta, desconto, contato proativo)
em vez de tratar toda a base igual.

## 2. Stakeholders

| Quem | Interesse |
|---|---|
| Diretoria | Reduzir churn mensal e proteger receita recorrente (MRR) |
| Time de Retenção / CRM | Receber uma lista priorizada de clientes para abordar |
| Marketing | Entender quais perfis e serviços estão associados ao cancelamento |
| Engenharia / Dados | Manter um serviço de predição confiável, reprodutível e testado |

## 3. Decisão apoiada pelo modelo

Para cada cliente ativo: **abordar ou não com ação de retenção neste mês?**
Saída do modelo: probabilidade de churn (0 a 1) + classe prevista (risco / sem risco).

## 4. Dados

- **Fonte:** Telco Customer Churn (IBM, público). 7.043 clientes, 21 colunas, 1 linha por cliente.
- **Alvo:** `Churn` (Yes/No). Cerca de 26,5% de churn → classes desbalanceadas.
- **Features:** perfil demográfico (gênero, idoso, parceiro, dependentes), serviços contratados
  (telefone, internet, segurança, backup, streaming...), contrato e cobrança (tipo de contrato,
  fatura digital, forma de pagamento, mensalidade, total pago, tempo de casa).
- **Limitações conhecidas:** foto estática (sem histórico temporal), dataset público e pequeno,
  sem dados de atendimento/reclamações, sem informação de concorrência ou região.

## 5. Métricas

**Técnica (principal): F1-score da classe churn.** Equilibra precision (não gastar oferta com quem
não ia sair) e recall (não deixar escapar quem vai sair). Complementar: **AUC-ROC**, que mede a
capacidade de ordenar clientes por risco independentemente do limiar de corte.

**Negócio:** taxa de churn mensal da carteira abordada vs. não abordada; receita retida por campanha;
custo da campanha / clientes retidos.

Premissas de custo usadas para ajustar o limiar de decisão (Etapa 2, `BusinessCosts`):

| Premissa | Valor | Racional |
|---|---|---|
| Custo de abordar um cliente | R$ 20 | contato + desconto médio oferecido |
| Valor de reter um cliente que ia sair | R$ 390 | ≈ 6 mensalidades de margem (mensalidade média ≈ R$ 65) |
| Taxa de sucesso da abordagem | 30% | fração dos abordados em risco que de fato ficam |

Ganho esperado por acerto = 0,30 × 390 − 20 = R$ 97; custo por falso alarme = R$ 20. Por isso vale abordar
clientes com probabilidade bem abaixo de 50%, e o modelo compensa mesmo com precision moderada. Os valores são
hipotéticos e devem ser substituídos pelos reais da operadora; o limiar é recalculado automaticamente.

**Baseline a superar:** Regressão Logística com pré-processamento padrão (Etapa 1). Os modelos da
Etapa 2 (Random Forest, MLP) precisam vencer o baseline em F1 com validação cruzada para virar campeão.

## 6. Como o modelo será usado (inferência)

- API REST (FastAPI) com `POST /predict` recebendo os dados cadastrais/contratuais de um cliente e
  retornando `churn_probability` e `churn_label`.
- Uso em lote mensal pelo time de retenção (score de toda a base) e consulta pontual pelo CRM.

## 7. Riscos e vieses a monitorar

- Desbalanceamento: acurácia enganosa; sempre reportar F1/recall da classe churn.
- Possível viés por gênero ou idade (`SeniorCitizen`): documentar no Model Card se o erro for
  desigual entre grupos.
- Drift: mudança de planos/preços da operadora invalida o modelo; prever re-treino periódico.
