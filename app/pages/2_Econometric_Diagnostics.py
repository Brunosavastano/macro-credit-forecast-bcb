from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "app"))

import pandas as pd
import streamlit as st

from app_utils import load_diagnostics, load_model_summary, load_stationarity, missing_artifacts_message
from macro_credit_forecast_bcb.viz.charts import residual_correlation_chart


st.set_page_config(page_title="Econometric Diagnostics", layout="wide")
st.title("Econometric Diagnostics")

stationarity = load_stationarity()
summary = load_model_summary()
diagnostics = load_diagnostics()

if not summary:
    missing_artifacts_message()
    st.stop()

st.subheader("Modelo e amostra")
cols = st.columns(4)
cols[0].metric("Modelo", summary.get("model_selected", ""))
cols[1].metric("N obs.", summary.get("nobs", ""))
cols[2].metric("Inicio", str(summary.get("sample_start", ""))[:10])
cols[3].metric("Fim", str(summary.get("sample_end", ""))[:10])

st.subheader("Testes ADF/KPSS")
if stationarity.empty:
    st.info("Relatorio de estacionariedade nao encontrado.")
else:
    st.dataframe(stationarity, use_container_width=True, hide_index=True)

st.subheader("Selecao de defasagens")
ic = pd.DataFrame(summary.get("information_criteria", []))
if ic.empty:
    st.info("Tabela de criterios de informacao nao encontrada.")
else:
    st.dataframe(ic, use_container_width=True, hide_index=True)

st.subheader("Diagnosticos do VAR")
diag_cols = st.columns(4)
diag_cols[0].metric("Estavel", str(diagnostics.get("is_stable", "")))
diag_cols[1].metric("Lag", diagnostics.get("lag_order", ""))
diag_cols[2].metric("BIC", round(float(diagnostics.get("bic", 0)), 3) if diagnostics.get("bic") is not None else "")
diag_cols[3].metric("Menor raiz abs.", round(float(diagnostics.get("roots_abs_min", 0)), 3) if diagnostics.get("roots_abs_min") is not None else "")

if diagnostics.get("residual_correlation"):
    st.plotly_chart(residual_correlation_chart(diagnostics["residual_correlation"]), use_container_width=True)

st.subheader("VECM candidato")
st.json(summary.get("vecm_candidate", {}))

st.info(
    "A decisao VAR/VECM deve seguir os testes de integracao, Johansen e diagnosticos. "
    "O app nao forca VECM por estetica."
)

