# SPEC — Projeção própria de IPCA, Selic e spread bancário com VAR/VECM usando dados BCB

## 1. Visão executiva

**Nome do projeto:** `macro-credit-forecast-bcb`

**Objetivo:** construir uma aplicação quantitativa, reprodutível e visualmente forte para projetar **IPCA, Selic, spread bancário, concessões de crédito e inadimplência** por 12 meses à frente, usando séries públicas do Banco Central do Brasil, modelos VAR/VECM e comparação contra Focus quando houver benchmark público compatível.

**Mensagem de entrevista técnica:**

> “Eu consigo pegar dados oficiais, transformar séries macrofinanceiras corretamente, testar estacionariedade e cointegração, escolher defasagens de forma defensável, estimar um VAR/VECM pequeno, gerar projeções com incerteza, comparar com Focus e entregar isso em produto analítico via Streamlit.”

O projeto deve ser tratado como um **mini-sistema de forecasting macroeconômico**, não como notebook exploratório. A entrega final deve ter pipeline modular, relatório metodológico, dashboard e backtest.

---

## 2. Escopo funcional

### 2.1 Indicadores projetados

O modelo principal trabalha com frequência **mensal** e projeta:

| Bloco | Indicador | Uso no modelo | Fonte/série sugerida |
|---|---:|---|---|
| Inflação | IPCA mensal | variável endógena; projeção direta e IPCA 12m acumulado derivado | SGS 433, Broad National Consumer Price Index/IPCA, variação mensal |
| Política monetária | Selic meta | variável endógena; usar nível mensal de fim de mês | SGS 432, Meta Selic definida pelo Copom |
| Crédito | Spread médio das operações de crédito | variável endógena; nível em p.p. ou diferença, conforme teste | SGS 20783, spread médio total |
| Crédito | Concessões de crédito | variável endógena; preferencialmente concessões reais transformadas | SGS 20631, concessões totais em milhões de reais |
| Risco de crédito | Inadimplência da carteira de crédito | variável endógena; nível percentual ou diferença, conforme teste | SGS 21082, inadimplência total acima de 90 dias |

**Decisão de escopo:** o modelo base usa cinco variáveis. Isso é deliberado. A amostra de crédito começa em 2011, então um VAR grande consumiria graus de liberdade rapidamente.

### 2.2 Comparação com Focus

A comparação com Focus deve ser feita para **IPCA e Selic**, porque o conjunto público de Expectativas de Mercado do BCB cobre indicadores como índices de preços, PIB, produção industrial, câmbio, Selic, variáveis fiscais e setor externo. Ele não lista explicitamente spread bancário, concessões ou inadimplência como indicadores-padrão do Focus.

Para crédito, a comparação deve ser com benchmarks estatísticos:

- random walk;
- AR univariado;
- média móvel ou média histórica sazonal;
- backtest rolling-origin.

O Focus deve ser acessado pela API Olinda/OData de Expectativas de Mercado. O BCB documenta endpoint OData e recursos como:

```text
ExpectativaMercadoMensais
ExpectativasMercadoSelic
ExpectativasMercadoAnuais
ExpectativasMercadoInflacao12Meses
```

---

## 3. Fontes de dados e ingestão

### 3.1 SGS — séries históricas

O SGS é o Sistema Gerenciador de Séries Temporais do BCB, criado para consolidar e disponibilizar informações econômico-financeiras.

Endpoint padrão:

```text
https://api.bcb.gov.br/dados/serie/bcdata.sgs.{codigo_serie}/dados?formato=json&dataInicial={dd/MM/aaaa}&dataFinal={dd/MM/aaaa}
```

Parâmetros principais:

| Parâmetro | Descrição |
|---|---|
| `codigo_serie` | código SGS da série |
| `dataInicial` | início da consulta no formato `dd/MM/aaaa` |
| `dataFinal` | fim da consulta no formato `dd/MM/aaaa` |
| `formato` | usar `json` para ingestão em Python |

**Observação operacional:** para séries diárias, implementar download em janelas, porque o BCB informa limitação de consulta por período de até 10 anos para séries históricas diárias em JSON/CSV desde março de 2025.

### 3.2 Focus — expectativas de mercado

Endpoint base:

```text
https://olinda.bcb.gov.br/olinda/servico/Expectativas/versao/v1/odata/
```

Recursos principais:

```text
ExpectativaMercadoMensais
ExpectativasMercadoSelic
ExpectativasMercadoAnuais
ExpectativasMercadoInflacao12Meses
```

Exemplo de filtro para IPCA:

```text
$filter=Indicador eq 'IPCA'
$orderby=Data desc
$top=100
$format=json
```

Para Selic, usar o recurso específico `ExpectativasMercadoSelic` e, em paralelo, `ExpectativasMercadoAnuais` para comparação de Selic meta de fechamento de ano quando útil.

---

## 4. Definição da base modelável

### 4.1 Frequência e janela

**Frequência:** mensal.

**Janela inicial sugerida:** março de 2011 em diante, porque as séries de crédito selecionadas têm início em março de 2011 nas páginas de metadados do BCB.

**Data de corte do modelo:** última data comum entre as cinco séries após transformação e limpeza.

### 4.2 Tratamento por variável

| Variável | Tratamento recomendado |
|---|---|
| IPCA | manter como variação mensal em %. Derivar IPCA 12m por acumulação composta. |
| Selic meta | série diária SGS 432 convertida para mensal por valor de fim de mês. Justificativa: representa postura de política monetária no fechamento do período. |
| Spread | manter em p.p. se estacionário; caso contrário, usar primeira diferença. |
| Concessões | deflacionar por índice acumulado de IPCA; depois usar `log(concessões reais)` para VECM ou `Δlog(concessões reais)` para VAR estacionário. |
| Inadimplência | manter em nível percentual se estacionária; caso contrário, usar primeira diferença. |

### 4.3 Variáveis finais do modelo base

Definir:

```text
y_t = [π_t, i_t, spread_t, Δlog(concessoes_reais_t), inad_t]'
```

Onde:

- `π_t` = IPCA mensal;
- `i_t` = Selic meta de fim de mês;
- `spread_t` = spread médio das operações de crédito;
- `Δlog(concessoes_reais_t)` = crescimento mensal real das concessões;
- `inad_t` = inadimplência da carteira de crédito.

Para VECM, a matriz alternativa pode usar níveis I(1):

```text
z_t = [i_t, spread_t, log(concessoes_reais_t), inad_t]'
```

IPCA mensal normalmente entra melhor como taxa estacionária; portanto, não forçar IPCA no VECM se os testes indicarem I(0). O projeto deve mostrar essa decisão explicitamente.

---

## 5. Metodologia econométrica

### 5.1 Testes de estacionariedade

Implementar testes por variável:

- ADF: hipótese nula de raiz unitária;
- KPSS: hipótese nula de estacionariedade;
- Phillips-Perron, se biblioteca disponível.

Regra de decisão:

1. Se ADF rejeita raiz unitária e KPSS não rejeita estacionariedade: tratar como I(0).
2. Se ADF não rejeita e KPSS rejeita: testar primeira diferença.
3. Se os testes divergem: documentar como caso ambíguo e usar critério econômico + inspeção gráfica + estabilidade do modelo.

Output obrigatório no dashboard:

| Série | Transformação | ADF p-value | KPSS p-value | Ordem escolhida | Justificativa |
|---|---:|---:|---:|---:|---|
| IPCA | nível mensal |  |  |  |  |
| Selic | nível/diferença |  |  |  |  |
| Spread | nível/diferença |  |  |  |  |
| Concessões reais | log/diferença log |  |  |  |  |
| Inadimplência | nível/diferença |  |  |  |  |

### 5.2 Seleção entre VAR e VECM

A biblioteca `statsmodels` oferece modelos VAR e VECM no módulo `tsa.vector_ar`, além de teste de Johansen e seleção de ordem em `statsmodels.tsa.vector_ar.vecm`.

**Árvore de decisão:**

1. Rodar testes de integração.
2. Se todas ou quase todas as variáveis relevantes forem I(1), rodar Johansen.
3. Se Johansen indicar posto de cointegração `r > 0`, estimar VECM.
4. Se não houver cointegração ou se houver mistura forte de I(0)/I(1), estimar VAR em transformações estacionárias.
5. Nunca estimar VECM por estética. A decisão precisa aparecer no relatório.

### 5.3 Seleção de defasagens

Como a amostra é mensal e relativamente curta, usar:

- `maxlags = 6` no modelo principal;
- `maxlags = 12` como robustez;
- BIC como critério primário;
- AIC como critério secundário;
- validação por diagnóstico de resíduos.

Justificativa: BIC penaliza mais a complexidade, o que é apropriado para VAR com cinco variáveis e amostra desde 2011.

### 5.4 Diagnósticos obrigatórios

Para cada modelo estimado:

- estabilidade do VAR: raízes dentro do círculo unitário;
- autocorrelação residual: Portmanteau/Ljung-Box multivariado quando disponível;
- normalidade residual: Jarque-Bera multivariado ou por equação;
- heterocedasticidade/volatilidade residual: diagnóstico visual e, se possível, ARCH;
- matriz de correlação residual;
- sensibilidade a defasagens alternativas.

### 5.5 Projeção

Gerar horizonte de **12 meses à frente**:

- previsão pontual;
- intervalo de confiança de 68% e 95%;
- previsão acumulada para IPCA 12m;
- previsão em nível para Selic, spread e inadimplência;
- previsão de crescimento real para concessões e, opcionalmente, reconversão para nível real/nominal.

Para intervalos:

- MVP: intervalo analítico do `statsmodels`;
- versão forte: bootstrap residual com 1.000 simulações para fan chart.

---

## 6. Comparação com Focus e benchmarks

### 6.1 IPCA

Comparações:

1. **IPCA mensal projetado pelo modelo vs mediana Focus mensal**, quando disponível.
2. **IPCA acumulado em 12 meses projetado vs Focus inflação 12 meses**, usando `ExpectativasMercadoInflacao12Meses`.
3. **IPCA anual projetado por ano-calendário vs Focus anual**, usando `ExpectativasMercadoAnuais`.

Métrica visual:

```text
Gap_h = Forecast_modelo_{t+h} - Forecast_Focus_{t+h}
```

Mostrar gap em bps ou p.p.

### 6.2 Selic

Comparações:

1. Selic meta projetada mensalmente vs Focus Selic por reunião Copom, mapeada para mês.
2. Selic final de ano projetada vs Focus anual.

A Selic do modelo deve ser tratada como trajetória mensal de nível. O Focus pode estar em formato por reunião ou ano-calendário, então o dashboard precisa explicitar o mapeamento.

### 6.3 Spread, concessões e inadimplência

Como não há benchmark Focus público padrão para esses indicadores, comparar contra:

- random walk;
- AR(1);
- média móvel 12 meses;
- modelo sazonal ingênuo: valor do mesmo mês do ano anterior;
- backtest rolling-origin.

Métricas:

| Métrica | Uso |
|---|---|
| MAE | erro absoluto médio, fácil de explicar |
| RMSE | penaliza erros grandes |
| MAPE/sMAPE | apenas se série for sempre positiva e escala permitir |
| Directional accuracy | útil para Selic, spread e inadimplência |
| Coverage | checar se intervalos de confiança cobrem realizações |

---

## 7. Streamlit — especificação do produto

### 7.1 Página 1 — Executive Dashboard

Objetivo: impressionar em 30 segundos.

Componentes:

- cards com último valor observado de IPCA, Selic, spread, concessões e inadimplência;
- cards com projeção 12 meses à frente;
- gap modelo vs Focus para IPCA e Selic;
- alerta metodológico: “Modelo selecionado: VAR(p) em transformações estacionárias” ou “VECM com posto r”.

Gráficos:

- linha histórica + forecast + fan chart para IPCA;
- linha histórica + forecast + Focus para Selic;
- linha histórica + forecast para spread;
- painel compacto de crédito: concessões reais e inadimplência.

### 7.2 Página 2 — Forecast Explorer

Filtros:

- horizonte: 1, 3, 6, 12 meses;
- modelo: VAR base, VECM candidato, benchmark AR/random walk;
- defasagens: automática/BIC, AIC, manual;
- intervalo: 68%, 95%.

Visualizações:

- forecast por variável;
- tabela de projeções;
- download CSV/Parquet;
- diferença contra Focus.

### 7.3 Página 3 — Econometric Diagnostics

Mostrar:

- testes ADF/KPSS;
- ordem de integração escolhida;
- resultado Johansen;
- critério de informação por defasagem;
- estabilidade do VAR;
- autocorrelação residual;
- matriz de correlação dos resíduos.

Essa página é crucial para entrevista técnica. Ela demonstra que o projeto não é só gráfico.

### 7.4 Página 4 — Backtest

Configuração:

- janela inicial mínima: 72 meses;
- expanding window por padrão;
- rolling window opcional;
- horizontes: 1, 3, 6, 12 meses.

Mostrar:

- tabela de erros por variável/modelo;
- gráfico de erro acumulado;
- comparação VAR/VECM vs benchmarks;
- para IPCA/Selic, comparação adicional com Focus quando a coleta histórica permitir.

### 7.5 Página 5 — Crédito e transmissão monetária

Visualizações técnicas:

- impulso-resposta reduzido: choque em Selic → spread, concessões e inadimplência;
- impulso-resposta: choque em inadimplência → spread;
- decomposição de variância, se estável;
- correlação defasada entre Selic, spread e concessões.

Nota obrigatória no app: IRF reduzido não é identificação causal estrutural. Para causalidade estrutural, seria necessário SVAR com restrições de identificação.

---

## 8. Arquitetura técnica

### 8.1 Estrutura do repositório

```text
macro-credit-forecast-bcb/
  README.md
  pyproject.toml
  Makefile
  .streamlit/
    config.toml
  config/
    series_sgs.yaml
    model.yaml
  data/
    raw/
    processed/
    forecasts/
  notebooks/
    01_eda_bcb_series.ipynb
    02_model_selection.ipynb
  src/
    data/
      sgs_client.py
      focus_client.py
      build_dataset.py
    features/
      transformations.py
      stationarity.py
      seasonal_adjustment.py
    models/
      var_model.py
      vecm_model.py
      model_selection.py
      forecast.py
      backtest.py
    viz/
      charts.py
      tables.py
    utils/
      dates.py
      logging.py
  app/
    streamlit_app.py
    pages/
      1_Executive_Dashboard.py
      2_Forecast_Explorer.py
      3_Econometric_Diagnostics.py
      4_Backtest.py
      5_Credit_Transmission.py
  tests/
    test_sgs_client.py
    test_transformations.py
    test_model_selection.py
```

### 8.2 Configuração das séries

`config/series_sgs.yaml`:

```yaml
series:
  ipca:
    code: 433
    name: "IPCA mensal"
    frequency: "M"
    unit: "% m/m"
    transform: "level_rate"

  selic_meta:
    code: 432
    name: "Meta Selic definida pelo Copom"
    frequency: "D"
    unit: "% a.a."
    monthly_conversion: "end_of_month"
    transform: "level_rate"

  spread_credito_total:
    code: 20783
    name: "Spread médio das operações de crédito - Total"
    frequency: "M"
    unit: "p.p."
    transform: "test_level_or_diff"

  concessoes_credito_total:
    code: 20631
    name: "Concessões de crédito - Total"
    frequency: "M"
    unit: "R$ milhões"
    transform: "deflate_log_diff"

  inadimplencia_total:
    code: 21082
    name: "Inadimplência da carteira de crédito - Total"
    frequency: "M"
    unit: "%"
    transform: "test_level_or_diff"
```

### 8.3 Funções mínimas

```python
get_sgs_series(code: int, start: str, end: str) -> pd.Series
get_focus_expectations(resource: str, filters: dict) -> pd.DataFrame
build_monthly_dataset(config) -> pd.DataFrame
run_stationarity_tests(df) -> pd.DataFrame
select_var_lag(df, maxlags=6, criterion="bic") -> dict
select_vecm_rank(df, det_order=0, k_ar_diff=selected_lag-1) -> dict
fit_var(df, lag) -> VARResults
fit_vecm(df, rank, k_ar_diff) -> VECMResults
forecast_12m(model, steps=12) -> pd.DataFrame
rolling_backtest(df, model_spec, horizons=[1,3,6,12]) -> pd.DataFrame
```

---

## 9. Critérios de aceite

A entrega é considerada completa se cumprir todos os critérios abaixo:

1. **Ingestão BCB funcional:** baixa automaticamente as cinco séries SGS e as expectativas Focus relevantes.
2. **Base mensal limpa:** dataset final tem índice mensal, última data comum e transformações documentadas.
3. **Testes econométricos documentados:** ADF/KPSS, escolha de defasagens e decisão VAR vs VECM aparecem no app e no README.
4. **Projeção 12 meses:** gera previsão pontual e intervalos para todos os indicadores.
5. **Comparação Focus:** IPCA e Selic têm gráfico e tabela de gap contra Focus.
6. **Benchmarks para crédito:** spread, concessões e inadimplência são comparados contra modelos ingênuos/univariados.
7. **Backtest:** pelo menos 36 meses de avaliação rolling/expanding quando a amostra permitir.
8. **Streamlit apresentável:** dashboard roda localmente com `streamlit run app/streamlit_app.py`.
9. **Reprodutibilidade:** `make refresh`, `make forecast` e `make app` funcionam.
10. **Clareza metodológica:** README explica transformações, defasagens, estacionariedade, cointegração, limitações e próximos passos.

---

## 10. Plano de execução sugerido

### Etapa 1 — Dados e EDA

- Criar clientes SGS e Focus.
- Baixar séries.
- Alinhar frequência mensal.
- Criar gráficos históricos.
- Validar datas, missing values e unidades.

Entrega: notebook EDA + dataset `processed/monthly_macro_credit.parquet`.

### Etapa 2 — Transformações e testes

- Deflacionar concessões.
- Testar estacionariedade.
- Criar tabela metodológica.
- Definir transformações finais.

Entrega: `stationarity_report.csv` + documentação no README.

### Etapa 3 — Modelagem

- Estimar VAR base.
- Rodar seleção de defasagens.
- Testar Johansen para candidato VECM.
- Escolher modelo campeão por estabilidade, diagnóstico e backtest.

Entrega: `model_summary.json`, `diagnostics.parquet`.

### Etapa 4 — Forecast e backtest

- Gerar projeções 12 meses.
- Implementar intervalos.
- Comparar IPCA/Selic com Focus.
- Comparar crédito com benchmarks.

Entrega: `forecast_12m.parquet`, `backtest_metrics.parquet`.

### Etapa 5 — Streamlit

- Montar dashboard executivo.
- Criar páginas de forecast, diagnóstico e backtest.
- Adicionar download de resultados.
- Polir visualmente.

Entrega: app completo e publicável.

---

## 11. Limitações e riscos

| Risco | Mitigação |
|---|---|
| Amostra curta para VAR de 5 variáveis | limitar defasagens, usar BIC, testar robustez com VAR menor |
| Mistura de I(0) e I(1) | não forçar VECM; usar VAR em transformações estacionárias como base |
| Selic como variável de decisão discreta | tratar forecast como trajetória quantitativa aproximada; comparar com Focus |
| Concessões com sazonalidade forte | usar deflação + transformação log + opção de dummies sazonais/STL |
| Focus sem spread/concessões/inadimplência | comparar crédito com benchmarks estatísticos e backtest |
| IRF interpretado como causalidade | rotular como impulso-resposta reduzido, não SVAR causal |

---

## 12. Diferencial técnico para entrevista

A entrega fica forte porque combina quatro competências raras no mesmo projeto:

1. **Macroeconomia aplicada:** inflação, política monetária, crédito e inadimplência no mesmo sistema.
2. **Econometria séria:** estacionariedade, cointegração, lag selection, diagnóstico residual e estabilidade.
3. **Data engineering leve:** APIs públicas, tratamento de frequência, cache, reprodutibilidade.
4. **Produto analítico:** Streamlit com forecast, Focus gap, backtest e documentação metodológica.

Pitch final:

> “Construí uma ferramenta própria de projeção macrofinanceira com dados oficiais do BCB. O sistema baixa SGS e Focus, estima VAR/VECM pequeno, documenta estacionariedade, cointegração e defasagens, gera forecasts de 12 meses para IPCA, Selic e crédito, compara IPCA/Selic com Focus e valida crédito por backtest contra benchmarks. A interface em Streamlit transforma o modelo em produto de decisão.”

---

## 13. Fontes úteis

- SGS/BCB — Sistema Gerenciador de Séries Temporais: `https://www4.bcb.gov.br/pec/series/port/aviso.asp?frame=1`
- API SGS BCB: `https://api.bcb.gov.br/dados/serie/bcdata.sgs.{codigo_serie}/dados?formato=json`
- Dados Abertos BCB — Expectativas de Mercado: `https://dadosabertos.bcb.gov.br/dataset/expectativas-mercado`
- API Olinda Expectativas: `https://olinda.bcb.gov.br/olinda/servico/Expectativas/versao/v1/odata/`
- Statsmodels VAR/VECM: `https://www.statsmodels.org/stable/vector_ar.html`
