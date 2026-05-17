from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "app"))

import streamlit as st

from app_utils import (
    dataframe_download,
    load_forecast,
    load_history,
    load_model_summary,
    missing_artifacts_message,
)
from macro_credit_forecast_bcb.viz.charts import history_forecast_chart, label, unit
from macro_credit_forecast_bcb.viz.tables import latest_values_table


st.set_page_config(page_title="Macro Credit Forecast BCB", layout="wide")

st.title("Macro Credit Forecast BCB")
st.caption("Forecast mensal de IPCA, Selic, spread, concessoes reais e inadimplencia com dados BCB.")

history = load_history()
forecast = load_forecast()
summary = load_model_summary()

if history.empty or forecast.empty:
    missing_artifacts_message()
    st.stop()

model_name = summary.get("model_selected", "Modelo ainda nao identificado")
sample_start = summary.get("sample_start", "")
sample_end = summary.get("sample_end", "")

col_a, col_b, col_c = st.columns(3)
col_a.metric("Modelo selecionado", model_name)
col_b.metric("Amostra inicial", str(sample_start)[:10])
col_c.metric("Ultima observacao", str(sample_end)[:10])

st.subheader("Resumo executivo")
table = latest_values_table(history, forecast, horizon=12)
st.dataframe(table, use_container_width=True, hide_index=True)
dataframe_download(table, "executive_summary.csv")

st.subheader("Trajetorias principais")
tabs = st.tabs(["IPCA", "Selic", "Spread", "Credito", "Inadimplencia"])
with tabs[0]:
    st.plotly_chart(history_forecast_chart(history, forecast, "ipca"), use_container_width=True)
    if "ipca_12m" in forecast["variable"].unique():
        st.plotly_chart(history_forecast_chart(history, forecast, "ipca_12m"), use_container_width=True)
with tabs[1]:
    st.plotly_chart(history_forecast_chart(history, forecast, "selic"), use_container_width=True)
with tabs[2]:
    st.plotly_chart(history_forecast_chart(history, forecast, "spread"), use_container_width=True)
with tabs[3]:
    st.plotly_chart(
        history_forecast_chart(history, forecast, "dlog_concessoes_reais"),
        use_container_width=True,
    )
with tabs[4]:
    st.plotly_chart(history_forecast_chart(history, forecast, "inadimplencia"), use_container_width=True)

st.info(
    "As projecoes sao quantitativas e condicionais ao modelo. IRFs no app sao reduzidos, "
    "sem identificacao causal estrutural."
)

with st.expander("Unidades"):
    for variable in ["ipca", "ipca_12m", "selic", "spread", "dlog_concessoes_reais", "inadimplencia"]:
        st.write(f"- {label(variable)}: {unit(variable)}")

