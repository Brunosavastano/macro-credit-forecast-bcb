from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "app"))

import pandas as pd
import streamlit as st

from app_utils import (
    apply_app_style,
    load_data_quality,
    load_diagnostics,
    load_model_summary,
    load_stationarity,
    missing_artifacts_message,
    quality_status,
    render_quality_pill,
)
from macro_credit_forecast_bcb.viz.charts import residual_correlation_chart


st.set_page_config(page_title="Econometric Diagnostics", layout="wide")
apply_app_style()
st.title("Econometric Diagnostics")

stationarity = load_stationarity()
quality = load_data_quality()
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

st.subheader("Qualidade dos dados")
render_quality_pill(quality_status(quality))
if quality.empty:
    st.info("Relatório de qualidade não encontrado.")
else:
    quality_display = quality.copy()
    for column in ["first_date", "last_date"]:
        quality_display[column] = quality_display[column].dt.strftime("%Y-%m-%d")
    if "scale_flags" in quality_display.columns:
        quality_display["scale_flags"] = quality_display["scale_flags"].fillna("")
    st.dataframe(
        quality_display,
        use_container_width=True,
        hide_index=True,
        column_config={
            "variable": st.column_config.TextColumn("Variável"),
            "status": st.column_config.TextColumn("Status"),
            "missing_values": st.column_config.NumberColumn("Missing", format="%d"),
            "duplicate_dates": st.column_config.NumberColumn("Duplicatas", format="%d"),
            "min": st.column_config.NumberColumn("Min", format="%.3f"),
            "max": st.column_config.NumberColumn("Max", format="%.3f"),
            "latest_value": st.column_config.NumberColumn("Último", format="%.3f"),
        },
    )

st.subheader("Testes ADF/KPSS")
if stationarity.empty:
    st.info("Relatório de estacionariedade não encontrado.")
else:
    st.dataframe(
        stationarity,
        use_container_width=True,
        hide_index=True,
        column_config={
            "adf_pvalue": st.column_config.NumberColumn("ADF p-value", format="%.4f"),
            "kpss_pvalue": st.column_config.NumberColumn("KPSS p-value", format="%.4f"),
        },
    )

st.subheader("Seleção de defasagens")
ic = pd.DataFrame(summary.get("information_criteria", []))
if ic.empty:
    st.info("Tabela de critérios de informação não encontrada.")
else:
    st.dataframe(
        ic,
        use_container_width=True,
        hide_index=True,
        column_config={
            "aic": st.column_config.NumberColumn("AIC", format="%.3f"),
            "bic": st.column_config.NumberColumn("BIC", format="%.3f"),
            "hqic": st.column_config.NumberColumn("HQIC", format="%.3f"),
            "fpe": st.column_config.NumberColumn("FPE", format="%.2e"),
        },
    )

st.subheader("Diagnosticos do VAR")
diag_cols = st.columns(4)
diag_cols[0].metric("Estável", str(diagnostics.get("is_stable", "")))
diag_cols[1].metric("Lag", diagnostics.get("lag_order", ""))
diag_cols[2].metric("BIC", round(float(diagnostics.get("bic", 0)), 3) if diagnostics.get("bic") is not None else "")
diag_cols[3].metric("Menor raiz abs.", round(float(diagnostics.get("roots_abs_min", 0)), 3) if diagnostics.get("roots_abs_min") is not None else "")

if diagnostics.get("residual_correlation"):
    st.plotly_chart(residual_correlation_chart(diagnostics["residual_correlation"]), use_container_width=True)

st.subheader("VECM candidato")
st.json(summary.get("vecm_candidate", {}))

st.info(
    "A decisão VAR/VECM deve seguir os testes de integração, Johansen e diagnósticos. "
    "O app não força VECM por estética."
)
