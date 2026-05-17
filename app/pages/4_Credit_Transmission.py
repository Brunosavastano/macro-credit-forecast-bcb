from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "app"))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app_utils import load_history, missing_artifacts_message
from macro_credit_forecast_bcb.features.transformations import model_columns
from macro_credit_forecast_bcb.models.model_selection import select_var_lag
from macro_credit_forecast_bcb.models.var_model import fit_var
from macro_credit_forecast_bcb.viz.charts import label


st.set_page_config(page_title="Credito e Transmissao", layout="wide")
st.title("Credito e Transmissao Monetaria")
st.info(
    "As funcoes impulso-resposta abaixo sao reduzidas. Elas descrevem dinamica do VAR "
    "e nao devem ser interpretadas como causalidade estrutural."
)

history = load_history()
if history.empty:
    missing_artifacts_message()
    st.stop()

data = history[model_columns()].dropna()
shock = st.selectbox("Choque", ["selic", "inadimplencia", "spread"], format_func=label)
responses = st.multiselect(
    "Respostas",
    ["spread", "dlog_concessoes_reais", "inadimplencia", "selic"],
    default=["spread", "dlog_concessoes_reais", "inadimplencia"],
    format_func=label,
)
horizon = st.slider("Horizonte da IRF", 6, 24, 12)

try:
    selection = select_var_lag(data, maxlags=6, criterion="bic")
    result = fit_var(data, int(selection["selected_lag"]))
    irf = result.irf(horizon)
    shock_idx = result.names.index(shock)
    periods = list(range(horizon + 1))

    fig = go.Figure()
    for response in responses:
        response_idx = result.names.index(response)
        values = irf.irfs[:, response_idx, shock_idx]
        fig.add_trace(go.Scatter(x=periods, y=values, mode="lines+markers", name=label(response)))
    fig.update_layout(
        title=f"Resposta a choque em {label(shock)}",
        xaxis_title="Meses apos o choque",
        yaxis_title="Resposta",
        template="plotly_white",
        margin=dict(l=30, r=20, t=70, b=35),
    )
    st.plotly_chart(fig, use_container_width=True)
except Exception as exc:
    st.error(f"Nao foi possivel estimar IRF: {exc}")

st.subheader("Correlacao defasada simples")
max_lag = st.slider("Defasagem maxima", 1, 12, 6)
rows = []
for lag in range(0, max_lag + 1):
    rows.append(
        {
            "lag": lag,
            "corr_selic_spread": data["selic"].shift(lag).corr(data["spread"]),
            "corr_selic_credito": data["selic"].shift(lag).corr(data["dlog_concessoes_reais"]),
            "corr_spread_inadimplencia": data["spread"].shift(lag).corr(data["inadimplencia"]),
        }
    )
st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

