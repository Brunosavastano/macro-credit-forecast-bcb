# Macro Credit Forecast BCB

Aplicação analítica para projetar IPCA, Selic, spread bancário, concessões de crédito e inadimplência usando séries públicas do Banco Central do Brasil, modelos VAR/VECM, benchmarks estatísticos e dashboard Streamlit.

O projeto foi estruturado como um mini-sistema de forecasting macroeconômico, não como notebook exploratório. A implementação baixa dados oficiais, trata frequências, estima modelos, gera forecasts, salva outputs e exibe resultados em uma interface executável.

## Fontes de dados

Séries históricas via SGS/BCB:

| Variável | Código SGS | Tratamento |
|---|---:|---|
| IPCA mensal | 433 | Mantido como variação mensal em %. |
| Selic meta | 432 | Série diária convertida para valor de fim de mês. |
| Spread médio de crédito | 20783 | Nível em p.p. no modelo base. |
| Concessões de crédito | 20631 | Deflacionadas por IPCA, depois `diff log`. |
| Inadimplência total | 21082 | Nível percentual no modelo base. |

Expectativas Focus via Olinda/OData são baixadas em modo best-effort para IPCA e Selic. Se a API não estiver disponível, o pipeline registra a falha e segue sem criar dados fictícios.

## Metodologia

1. As séries são alinhadas em frequência mensal e restritas à última janela comum completa.
2. A Selic diária é convertida para fim de mês.
3. Concessões nominais são deflacionadas por índice acumulado de IPCA.
4. A base VAR usa:
   `ipca`, `selic`, `spread`, `dlog_concessoes_reais`, `inadimplencia`.
5. Checks de qualidade validam escala, missing values, duplicatas e saltos anormais antes da modelagem.
6. ADF e KPSS são calculados para documentar estacionariedade.
7. A defasagem do VAR é escolhida por BIC, com `maxlags=6` por padrão.
8. Johansen e VECM são tratados como candidatos, não como obrigação. O modelo base permanece VAR em transformações estacionárias quando há mistura I(0)/I(1).
9. Forecasts de 12 meses são gerados com intervalos analíticos de 68% e 95%.
10. Backtest expanding-window compara VAR contra random walk, AR(1), média móvel de 12 meses e sazonal ingênuo.

As funções impulso-resposta exibidas no dashboard são reduzidas. Elas não constituem identificação causal estrutural.

## Instalação

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e .[dev]
```

Em PowerShell, se o extra `[dev]` for interpretado pelo shell, use:

```bash
python -m pip install -e ".[dev]"
```

## Execução

Com `make` disponível:

```bash
make refresh
make forecast
make backtest
make audit
make app
```

Sem `make`:

```bash
python -m macro_credit_forecast_bcb.pipeline.refresh
python -m macro_credit_forecast_bcb.pipeline.forecast
python -m macro_credit_forecast_bcb.pipeline.backtest
python -m macro_credit_forecast_bcb.pipeline.audit
streamlit run app/streamlit_app.py
```

Parâmetros úteis:

```bash
python -m macro_credit_forecast_bcb.pipeline.refresh --start 2011-03-01 --skip-focus
python -m macro_credit_forecast_bcb.pipeline.refresh --allow-quality-warnings
python -m macro_credit_forecast_bcb.pipeline.forecast --horizon 12
python -m macro_credit_forecast_bcb.pipeline.backtest --initial-window 72
```

## Outputs

| Arquivo | Descrição |
|---|---|
| `data/processed/monthly_macro_credit_raw.parquet` | Base mensal alinhada antes das transformações finais. |
| `data/processed/monthly_macro_credit.parquet` | Base modelável com variáveis transformadas. |
| `data/processed/series_metadata.csv` | Metadados das séries baixadas. |
| `outputs/data_quality_report.csv` | Checks de escala, missing values, duplicatas e saltos anormais. |
| `outputs/stationarity_report.csv` | ADF, KPSS e decisão documentada por variável. |
| `outputs/var_lag_selection.parquet` | Critérios de informação por defasagem. |
| `outputs/model_summary.json` | Resumo do modelo, diagnósticos e candidato VECM. |
| `outputs/diagnostics.parquet` | Diagnósticos serializados do VAR. |
| `data/forecasts/forecast_12m.parquet` | Forecast pontual e intervalos. |
| `outputs/backtest_records.parquet` | Previsões fora da amostra por origem. |
| `outputs/backtest_metrics.parquet` | MAE, RMSE, sMAPE e directional accuracy. |
| `outputs/econometric_audit.json` | Scorecard executivo da auditoria econométrica. |
| `outputs/econometric_audit_metrics.parquet` | MAE, RMSE, sMAPE, MASE, Theil U e ranking por horizonte. |
| `outputs/interval_coverage.parquet` | Cobertura e largura média dos intervalos do VAR no backtest. |
| `outputs/model_comparison_tests.parquet` | Testes Diebold-Mariano entre VAR e benchmarks. |

## Dashboard

O Streamlit contém:

- Executive Dashboard;
- Forecast Explorer;
- Econometric Diagnostics;
- Econometric Audit;
- Backtest;
- Crédito e Transmissão Monetária.

Se os artefatos ainda não existirem, o app mostra os comandos necessários para gerá-los.

## Testes

```bash
pytest
```

Os testes cobrem parsing SGS, transformações, seleção de defasagens VAR e benchmarks.

## Limitações conhecidas

- A amostra de crédito é curta para VARs muito grandes; por isso a defasagem padrão é conservadora.
- O Focus público não cobre spread, concessões ou inadimplência como indicadores padrão comparáveis; essas séries usam benchmarks estatísticos.
- Intervalos usam a rotina analítica do `statsmodels`; bootstrap residual com muitas simulações é um aprimoramento natural.
- IRFs são reduzidas e não identificam choques estruturais de política monetária.
- APIs públicas podem falhar ou alterar esquemas; o pipeline explicita essas falhas em vez de substituir por dados fictícios.
- Se as séries SGS vierem em escala incompatível com as regras de plausibilidade, o refresh falha antes de salvar um modelo contaminado.

## Próximos passos naturais

- Adicionar bootstrap residual para fan charts.
- Persistir comparações Focus já mapeadas para horizontes mensais e ano-calendário.
- Incluir ajuste sazonal opcional para concessões.
- Publicar o dashboard em ambiente gerenciado.
- Adicionar SVAR com restrições explícitas se o objetivo passar a ser inferência causal.
