from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
FORECASTS = ROOT / "data" / "forecasts"
OUTPUTS = ROOT / "outputs"


@st.cache_data(show_spinner=False)
def load_history() -> pd.DataFrame:
    path = PROCESSED / "monthly_macro_credit.parquet"
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_parquet(path)
    frame.index = pd.to_datetime(frame.index)
    return frame


@st.cache_data(show_spinner=False)
def load_forecast() -> pd.DataFrame:
    path = FORECASTS / "forecast_12m.parquet"
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_parquet(path)
    frame["date"] = pd.to_datetime(frame["date"])
    return frame


@st.cache_data(show_spinner=False)
def load_stationarity() -> pd.DataFrame:
    path = OUTPUTS / "stationarity_report.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_metrics() -> pd.DataFrame:
    path = OUTPUTS / "backtest_metrics.parquet"
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_model_summary() -> dict:
    path = OUTPUTS / "model_summary.json"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


@st.cache_data(show_spinner=False)
def load_diagnostics() -> dict:
    path = OUTPUTS / "diagnostics.parquet"
    if not path.exists():
        return {}
    frame = pd.read_parquet(path)
    return {row["metric"]: json.loads(row["value"]) for _, row in frame.iterrows()}


def add_src_to_path() -> None:
    import sys

    src = str(ROOT / "src")
    app = str(ROOT / "app")
    if src not in sys.path:
        sys.path.insert(0, src)
    if app not in sys.path:
        sys.path.insert(0, app)


def missing_artifacts_message() -> None:
    st.warning("Os artefatos do pipeline ainda nao foram gerados neste workspace.")
    st.code(
        "python -m pip install -e .\n"
        "python -m macro_credit_forecast_bcb.pipeline.refresh\n"
        "python -m macro_credit_forecast_bcb.pipeline.forecast\n"
        "python -m macro_credit_forecast_bcb.pipeline.backtest",
        language="bash",
    )


def dataframe_download(frame: pd.DataFrame, filename: str, label: str = "Download CSV") -> None:
    st.download_button(
        label,
        frame.to_csv(index=False).encode("utf-8"),
        file_name=filename,
        mime="text/csv",
    )

