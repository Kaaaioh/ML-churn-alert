# Roteiro do vídeo STAR — ML Churn Alert (5 minutos)

Formato exigido: **S**ituation → **T**ask → **A**ction → **R**esult. Limite: 5 minutos (mire em 4:30 para ter folga).
Grave tela + voz; rosto é opcional. Cada bloco traz **o que mostrar na tela**, **o que dizer** e o **tempo-alvo**.

Dica de gravação: abra antes as abas/arquivos na ordem abaixo, deixe a API rodando (`uvicorn churn_alert.api:app`)
e o Swagger aberto em `http://127.0.0.1:8000/docs`. Fale como quem apresenta um produto, não como quem lê código.

---

## 0. Abertura (0:00 – 0:20)

**Tela:** README do repositório no GitHub, com o gráfico de capa visível.

**Fala:**
> "Esse é o ML Churn Alert, um serviço que prevê quais clientes de uma operadora de telecom estão prestes a cancelar
> e diz ao time de retenção o que fazer com cada um. Em 5 minutos vou mostrar o problema, o que construímos, as
> decisões técnicas e o resultado funcionando."

---

## S — Situation (0:20 – 1:00)

**Tela:** `docs/ml_canvas.md` (seção de proposta de valor e stakeholders) e depois a figura
`docs/figures/01_target_distribution.png`.

**Fala:**
> "Uma operadora está perdendo clientes em ritmo acelerado. A diretoria precisa saber **quem** vai sair **antes** que
> saia, para agir de forma direcionada em vez de dar desconto para todo mundo."
>
> "Usamos o dataset público Telco Customer Churn da IBM: 7 mil clientes, 19 variáveis de cadastro e contrato, e um
> rótulo dizendo se o cliente cancelou. 26% cancelaram. Isso importa: um modelo que diz 'ninguém vai sair' já acerta
> 74%, então acurácia não serve para avaliar."

---

## T — Task (1:00 – 1:30)

**Tela:** tabela das 4 etapas no README (ou `docs/progresso.md` com os checks).

**Fala:**
> "Nossa tarefa era construir o ciclo completo: análise exploratória, baseline, comparação de três famílias de
> modelo no scikit-learn, escolher um campeão com validação cruzada e servir esse modelo numa API FastAPI com
> testes, tudo com reprodutibilidade garantida por seeds fixas."
>
> "Definimos como métrica técnica o **F1 da classe churn**, que equilibra encontrar quem vai sair com não
> desperdiçar oferta, e AUC-ROC como apoio. Como métrica de negócio, o lucro esperado de uma campanha de retenção."

---

## A — Action (1:30 – 3:30)

### A1. Dados e EDA (1:30 – 2:00)

**Tela:** `docs/figures/03_churn_rate_by_category.png` (grade de barras).

**Fala:**
> "Na exploração, o único dado sujo era o total pago vindo como texto: 11 clientes novos com o campo em branco,
> que tratamos como zero. E os sinais de risco ficaram claros: contrato mensal, fibra óptica, pagamento por cheque
> eletrônico e pouco tempo de casa. Gênero não influencia nada, o que é bom sinal para vieses."

### A2. Modelos e campeão (2:00 – 2:45)

**Tela:** `docs/figures/07_cv_comparison.png` e depois a tabela em `docs/experimentos.md`.

**Fala:**
> "Treinamos Regressão Logística, Random Forest e uma rede neural MLP, todos com o mesmo pré-processamento dentro
> de um Pipeline e a mesma validação cruzada de 5 folds. Random Forest e Regressão Logística empataram tecnicamente
> em F1, 0,63. A regra que definimos **antes** de ver os números diz: empate vai para o modelo mais simples.
> Ficamos com a Regressão Logística: treina em menos de um décimo de segundo e dá para ler os pesos."

### A3. Limiar de negócio (2:45 – 3:10)

**Tela:** `docs/figures/10_profit_curve.png`.

**Fala:**
> "Aqui está a decisão que mais gostei. O corte padrão de 50% não sabe nada do negócio. Calculamos o lucro esperado
> da campanha para cada limiar, com premissas explícitas de custo de abordagem e valor do cliente retido. O ótimo
> ficou em 33%: o modelo passa a encontrar 92% dos clientes que vão sair, e o lucro esperado sobe 8%."

### A4. Engenharia e API (3:10 – 3:30)

**Tela:** estrutura de pastas no VS Code (`src/churn_alert/`), depois `tests/` com o `pytest` rodando.

**Fala:**
> "Todo o código reutilizável saiu dos notebooks para módulos em `src/`: dados, features, treino, limiar, predição
> e API. 30 testes com pytest cobrem limpeza, pipeline, treino e os endpoints, e rodam em CI sem precisar do
> modelo real, porque treinam um modelo pequeno em pasta temporária."

---

## R — Result (3:30 – 4:30)

**Tela:** Swagger em `/docs`. Executar `POST /predict` com o exemplo pré-preenchido (cliente de risco) e mostrar a
resposta. Depois trocar `Contract` para `Two year` e `tenure` para 70 e executar de novo.

**Fala:**
> "A API no ar. `/health` confirma que o modelo está carregado. Envio um cliente novo, de fibra, contrato mensal:
> 89% de probabilidade, faixa **alto**, ação sugerida: contato proativo com oferta personalizada."
>
> "Agora um cliente antigo com contrato de dois anos: 1,5%, faixa **baixo**, nenhuma ação. A resposta não é só um
> número: é uma decisão pronta para o time de retenção."
>
> "No conjunto de teste, as faixas separam bem: 5% de churn real na faixa baixa, 54% na alta."

**Tela:** `docs/model_card.md`, rolar até a seção de vieses.

**Fala:**
> "Documentamos tudo num Model Card, inclusive as limitações: o modelo alerta muito mais idosos porque eles cancelam
> mais, quase não detecta churn em contratos de dois anos, e as probabilidades são superestimadas pelo balanceamento,
> então servem para ranking, não como frequência real."

---

## Fechamento (4:30 – 4:50)

**Tela:** README, seção de setup.

**Fala:**
> "Repositório com README de instalação, Model Card, registro de experimentos e CI. Próximos passos naturais:
> calibrar as probabilidades, ajustar hiperparâmetros e colocar a API na nuvem. Obrigado."

---

## Checklist antes de gravar

- [ ] `uvicorn churn_alert.api:app` rodando e `/docs` aberto
- [ ] Figuras 01, 03, 07 e 10 abertas em abas
- [ ] `pytest -q` já executado no terminal (30 passed visível)
- [ ] README no GitHub aberto no navegador
- [ ] Cronômetro: ensaie uma vez; se passar de 4:50, corte a seção A1 pela metade
