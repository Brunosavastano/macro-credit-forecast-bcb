from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "app"))

import streamlit as st

from app_utils import dataframe_download, load_metrics, missing_artifacts_message
from macro_credit_forecast_bcb.viz.charts import VARIABLE_LABELS, label, metrics_heatmap


st.set_page_config(page_title="Backtest", layout="wide")
st.title("Backtest")

metrics = load_metrics()
if metrics.empty:
    missing_artifacts_message()
    st.stop()

variable_options = [v for v in VARIABLE_LABELS if v in metrics["variable"].unique()]
variable = st.selectbox("Indicador", variable_options, format_func=label)
metric = st.selectbox("Metrica", ["rmse", "mae", "smape", "directional_accuracy"])

st.plotly_chart(metrics_heatmap(metrics, variable, metric=metric), use_container_width=True)

subset = metrics.loc[metrics["variable"] == variable].sort_values(["horizon", "model"])
st.dataframe(subset, use_container_width=True, hide_index=True)
dataframe_download(subset, f"backtest_{variable}.csv")

