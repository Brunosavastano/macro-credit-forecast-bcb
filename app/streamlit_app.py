from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "app"))

import streamlit as st

from app_utils import (
    apply_app_style,
    dataframe_download,
    load_data_quality,
    load_forecast,
    load_history,
    load_model_summary,
    missing_artifacts_message,
    quality_status,
    render_quality_pill,
)
from macro_credit_forecast_bcb.viz.charts import history_forecast_chart, label, unit
from macro_credit_forecast_bcb.viz.tables import latest_values_table


st.set_page_config(page_title="Macro Credit Forecast BCB", layout="wide")
apply_app_style()

st.title("Macro Credit Forecast BCB")
st.caption("Forecast mensal de IPCA, Selic, spread, concessoes reais e inadimplencia com dados BCB.")

history = load_history()
forecast = load_forecast()
summary = load_model_summary()
quality = load_data_quality()

if history.empty or forecast.empty:
    missing_artifacts_message()
    st.stop()

model_name = summary.get("model_selected", "Modelo ainda nao identificado")
sample_start = summary.get("sample_start", "")
sample_end = summary.get("sample_end", "")
quality_summary = summary.get("data_quality", {})
quality_checked_at = str(quality_summary.get("checked_at", ""))[:19].replace("T", " ")

top_cols = st.columns([1.1, 1, 1, 0.9])
top_cols[0].metric("Modelo", model_name)
top_cols[1].metric("Amostra", f"{str(sample_start)[:10]} -> {str(sample_end)[:10]}")
top_cols[2].metric("Qualidade", str(quality_summary.get("status", quality_status(quality))).upper())
top_cols[3].metric("Atualizado", quality_checked_at or "-")

status_col, note_col = st.columns([0.22, 0.78])
with status_col:
    render_quality_pill(quality_status(quality))
with note_col:
    st.caption("Os checks validam escala, missing values, duplicatas e saltos anormais antes da modelagem.")

st.subheader("Resumo executivo")
table = latest_values_table(history, forecast, horizon=12, formatted=True)
st.dataframe(
    table,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Indicador": st.column_config.TextColumn("Indicador", width="medium"),
        "Unidade": st.column_config.TextColumn("Unidade", width="medium"),
        "Ultimo observado": st.column_config.TextColumn("Ultimo observado", width="small"),
        "Forecast h=12": st.column_config.TextColumn("Forecast h=12", width="small"),
    },
)
dataframe_download(table, "executive_summary.csv")

with st.expander("Qualidade dos dados", expanded=False):
    if quality.empty:
        st.info("Relatorio de qualidade ainda nao gerado.")
    else:
        display_quality = quality.copy()
        for column in ["first_date", "last_date"]:
            display_quality[column] = display_quality[column].dt.strftime("%Y-%m-%d")
        if "scale_flags" in display_quality.columns:
            display_quality["scale_flags"] = display_quality["scale_flags"].fillna("")
        st.dataframe(
            display_quality,
            use_container_width=True,
            hide_index=True,
            column_config={
                "variable": st.column_config.TextColumn("Variavel"),
                "status": st.column_config.TextColumn("Status"),
                "min": st.column_config.NumberColumn("Min", format="%.3f"),
                "max": st.column_config.NumberColumn("Max", format="%.3f"),
                "latest_value": st.column_config.NumberColumn("Ultimo", format="%.3f"),
            },
        )

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
    if "concessoes_reais" in forecast["variable"].unique():
        st.plotly_chart(
            history_forecast_chart(history, forecast, "concessoes_reais"),
            use_container_width=True,
        )
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
