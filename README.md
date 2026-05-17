# Macro Credit Forecast BCB

Aplicacao analitica para projetar IPCA, Selic, spread bancario, concessoes de credito e inadimplencia usando series publicas do Banco Central do Brasil, modelos VAR/VECM, benchmarks estatisticos e dashboard Streamlit.

O projeto foi estruturado como um mini-sistema de forecasting macroeconomico, nao como notebook exploratorio. A implementacao baixa dados oficiais, trata frequencias, estima modelos, gera forecasts, salva outputs e exibe resultados em uma interface executavel.

## Fontes de dados

Series historicas via SGS/BCB:

| Variavel | Codigo SGS | Tratamento |
|---|---:|---|
| IPCA mensal | 433 | Mantido como variacao mensal em %. |
| Selic meta | 432 | Serie diaria convertida para valor de fim de mes. |
| Spread medio de credito | 20783 | Nivel em p.p. no modelo base. |
| Concessoes de credito | 20631 | Deflacionadas por IPCA, depois `diff log`. |
| Inadimplencia total | 21082 | Nivel percentual no modelo base. |

Expectativas Focus via Olinda/OData sao baixadas em modo best-effort para IPCA e Selic. Se a API nao estiver disponivel, o pipeline registra a falha e segue sem criar dados ficticios.

## Metodologia

1. As series sao alinhadas em frequencia mensal e restritas a ultima janela comum completa.
2. A Selic diaria e convertida para fim de mes.
3. Concessoes nominais sao deflacionadas por indice acumulado de IPCA.
4. A base VAR usa:
   `ipca`, `selic`, `spread`, `dlog_concessoes_reais`, `inadimplencia`.
5. ADF e KPSS sao calculados para documentar estacionariedade.
6. A defasagem do VAR e escolhida por BIC, com `maxlags=6` por padrao.
7. Johansen e VECM sao tratados como candidatos, nao como obrigacao. O modelo base permanece VAR em transformacoes estacionarias quando ha mistura I(0)/I(1).
8. Forecasts de 12 meses sao gerados com intervalos analiticos de 68% e 95%.
9. Backtest expanding-window compara VAR contra random walk, AR(1), media movel de 12 meses e sazonal ingenuo.

As funcoes impulso-resposta exibidas no dashboard sao reduzidas. Elas nao constituem identificacao causal estrutural.

## Instalacao

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

## Execucao

Com `make` disponivel:

```bash
make refresh
make forecast
make backtest
make app
```

Sem `make`:

```bash
python -m macro_credit_forecast_bcb.pipeline.refresh
python -m macro_credit_forecast_bcb.pipeline.forecast
python -m macro_credit_forecast_bcb.pipeline.backtest
streamlit run app/streamlit_app.py
```

Parametros uteis:

```bash
python -m macro_credit_forecast_bcb.pipeline.refresh --start 2011-03-01 --skip-focus
python -m macro_credit_forecast_bcb.pipeline.forecast --horizon 12
python -m macro_credit_forecast_bcb.pipeline.backtest --initial-window 72
```

## Outputs

| Arquivo | Descricao |
|---|---|
| `data/processed/monthly_macro_credit_raw.parquet` | Base mensal alinhada antes das transformacoes finais. |
| `data/processed/monthly_macro_credit.parquet` | Base modelavel com variaveis transformadas. |
| `data/processed/series_metadata.csv` | Metadados das series baixadas. |
| `outputs/stationarity_report.csv` | ADF, KPSS e decisao documentada por variavel. |
| `outputs/var_lag_selection.parquet` | Criterios de informacao por defasagem. |
| `outputs/model_summary.json` | Resumo do modelo, diagnosticos e candidato VECM. |
| `outputs/diagnostics.parquet` | Diagnosticos serializados do VAR. |
| `data/forecasts/forecast_12m.parquet` | Forecast pontual e intervalos. |
| `outputs/backtest_records.parquet` | Previsoes fora da amostra por origem. |
| `outputs/backtest_metrics.parquet` | MAE, RMSE, sMAPE e directional accuracy. |

## Dashboard

O Streamlit contem:

- Executive Dashboard;
- Forecast Explorer;
- Econometric Diagnostics;
- Backtest;
- Credito e Transmissao Monetaria.

Se os artefatos ainda nao existirem, o app mostra os comandos necessarios para gera-los.

## Testes

```bash
pytest
```

Os testes cobrem parsing SGS, transformacoes, selecao de defasagens VAR e benchmarks.

## Limitacoes conhecidas

- A amostra de credito e curta para VARs muito grandes; por isso a defasagem padrao e conservadora.
- O Focus publico nao cobre spread, concessoes ou inadimplencia como indicadores padrao comparaveis; essas series usam benchmarks estatisticos.
- Intervalos usam a rotina analitica do `statsmodels`; bootstrap residual com muitas simulacoes e um aprimoramento natural.
- IRFs sao reduzidas e nao identificam choques estruturais de politica monetaria.
- APIs publicas podem falhar ou alterar esquemas; o pipeline explicita essas falhas em vez de substituir por dados ficticios.

## Proximos passos naturais

- Adicionar bootstrap residual para fan charts.
- Persistir comparacoes Focus ja mapeadas para horizontes mensais e ano-calendario.
- Incluir ajuste sazonal opcional para concessoes.
- Publicar o dashboard em ambiente gerenciado.
- Adicionar SVAR com restricoes explicitas se o objetivo passar a ser inferencia causal.

