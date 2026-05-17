from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "app"))

import streamlit as st

from app_utils import apply_app_style, dataframe_download, load_metrics, missing_artifacts_message
from macro_credit_forecast_bcb.viz.charts import VARIABLE_LABELS, label, metrics_heatmap
from macro_credit_forecast_bcb.viz.formatting import format_metric


st.set_page_config(page_title="Backtest", layout="wide")
apply_app_style()
st.title("Backtest")

metrics = load_metrics()
if metrics.empty:
    missing_artifacts_message()
    st.stop()

variable_options = [v for v in VARIABLE_LABELS if v in metrics["variable"].unique()]
variable = st.selectbox("Indicador", variable_options, format_func=label)
metric = st.selectbox("Metrica", ["rmse", "mae", "smape", "directional_accuracy"])

st.plotly_chart(metrics_heatmap(metrics, variable, metric=metric), use_container_width=True)

subset = metrics.loc[metrics["variable"] == variable].sort_values(["horizon", "model"]).copy()
display = subset.copy()
for column in ["mae", "rmse", "smape", "directional_accuracy"]:
    if column in display.columns:
        display[column] = display[column].map(lambda value: format_metric(value, 4))
st.dataframe(
    display,
    use_container_width=True,
    hide_index=True,
    column_config={
        "model": st.column_config.TextColumn("Modelo"),
        "variable": st.column_config.TextColumn("Indicador"),
        "horizon": st.column_config.NumberColumn("Horizonte", format="%d"),
        "mae": st.column_config.TextColumn("MAE"),
        "rmse": st.column_config.TextColumn("RMSE"),
        "smape": st.column_config.TextColumn("sMAPE"),
        "directional_accuracy": st.column_config.TextColumn("Directional acc."),
    },
)
dataframe_download(subset, f"backtest_{variable}.csv")
