from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "app"))

import streamlit as st

from app_utils import apply_app_style, dataframe_download, load_forecast, load_history, missing_artifacts_message
from macro_credit_forecast_bcb.viz.charts import VARIABLE_LABELS, forecast_table, history_forecast_chart, label


st.set_page_config(page_title="Forecast Explorer", layout="wide")
apply_app_style()
st.title("Forecast Explorer")

history = load_history()
forecast = load_forecast()
if history.empty or forecast.empty:
    missing_artifacts_message()
    st.stop()

variables = [v for v in VARIABLE_LABELS if v in forecast["variable"].unique()]
variable = st.selectbox("Indicador", variables, format_func=label)
interval = st.radio("Intervalo", ["68", "95"], index=1, horizontal=True)
horizon = st.slider("Horizonte exibido na tabela", 1, int(forecast["horizon"].max()), 12)

st.plotly_chart(
    history_forecast_chart(history, forecast, variable, interval=interval),
    use_container_width=True,
)

subset = forecast[(forecast["variable"] == variable) & (forecast["horizon"] <= horizon)]
table = forecast_table(subset)
st.dataframe(
    table,
    use_container_width=True,
    hide_index=True,
    column_config={
        "date": st.column_config.TextColumn("Data"),
        "horizon": st.column_config.NumberColumn("Horizonte", format="%d"),
        "variable": st.column_config.TextColumn("Indicador"),
        "forecast": st.column_config.TextColumn("Forecast"),
        "lower_68": st.column_config.TextColumn("IC 68% inferior"),
        "upper_68": st.column_config.TextColumn("IC 68% superior"),
        "lower_95": st.column_config.TextColumn("IC 95% inferior"),
        "upper_95": st.column_config.TextColumn("IC 95% superior"),
    },
)
dataframe_download(table, f"forecast_{variable}.csv")
