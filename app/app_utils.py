from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
FORECASTS = ROOT / "data" / "forecasts"
OUTPUTS = ROOT / "outputs"


def _file_signature(path: Path) -> tuple[int, int] | None:
    if not path.exists():
        return None
    stat = path.stat()
    return stat.st_mtime_ns, stat.st_size


@st.cache_data(show_spinner=False)
def _read_history(path: str, signature: tuple[int, int] | None) -> pd.DataFrame:
    del signature
    if not Path(path).exists():
        return pd.DataFrame()
    frame = pd.read_parquet(path)
    frame.index = pd.to_datetime(frame.index)
    inferred_freq = pd.infer_freq(frame.index)
    if inferred_freq:
        frame = frame.asfreq(inferred_freq)
    return frame


def load_history() -> pd.DataFrame:
    path = PROCESSED / "monthly_macro_credit.parquet"
    return _read_history(str(path), _file_signature(path))


@st.cache_data(show_spinner=False)
def _read_forecast(path: str, signature: tuple[int, int] | None) -> pd.DataFrame:
    del signature
    if not Path(path).exists():
        return pd.DataFrame()
    frame = pd.read_parquet(path)
    frame["date"] = pd.to_datetime(frame["date"])
    return frame


def load_forecast() -> pd.DataFrame:
    path = FORECASTS / "forecast_12m.parquet"
    return _read_forecast(str(path), _file_signature(path))


@st.cache_data(show_spinner=False)
def _read_csv(path: str, signature: tuple[int, int] | None) -> pd.DataFrame:
    del signature
    return pd.read_csv(path) if Path(path).exists() else pd.DataFrame()


def load_stationarity() -> pd.DataFrame:
    path = OUTPUTS / "stationarity_report.csv"
    return _read_csv(str(path), _file_signature(path))


@st.cache_data(show_spinner=False)
def _read_data_quality(path: str, signature: tuple[int, int] | None) -> pd.DataFrame:
    del signature
    frame = pd.read_csv(path) if Path(path).exists() else pd.DataFrame()
    for column in ["first_date", "last_date"]:
        if column in frame.columns:
            frame[column] = pd.to_datetime(frame[column], errors="coerce")
    return frame


def load_data_quality() -> pd.DataFrame:
    path = OUTPUTS / "data_quality_report.csv"
    return _read_data_quality(str(path), _file_signature(path))


@st.cache_data(show_spinner=False)
def _read_parquet(path: str, signature: tuple[int, int] | None) -> pd.DataFrame:
    del signature
    return pd.read_parquet(path) if Path(path).exists() else pd.DataFrame()


def load_metrics() -> pd.DataFrame:
    path = OUTPUTS / "backtest_metrics.parquet"
    return _read_parquet(str(path), _file_signature(path))


@st.cache_data(show_spinner=False)
def _read_json(path: str, signature: tuple[int, int] | None) -> dict:
    del signature
    if not Path(path).exists():
        return {}
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


def load_audit_summary() -> dict:
    path = OUTPUTS / "econometric_audit.json"
    return _read_json(str(path), _file_signature(path))


def load_audit_table(filename: str) -> pd.DataFrame:
    path = OUTPUTS / filename
    return _read_parquet(str(path), _file_signature(path))


def load_model_summary() -> dict:
    path = OUTPUTS / "model_summary.json"
    return _read_json(str(path), _file_signature(path))


def load_diagnostics() -> dict:
    path = OUTPUTS / "diagnostics.parquet"
    frame = _read_parquet(str(path), _file_signature(path))
    if frame.empty:
        return {}
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
    st.warning("Os artefatos do pipeline ainda não foram gerados neste workspace.")
    st.code(
        "python -m pip install -e .\n"
        "python -m macro_credit_forecast_bcb.pipeline.refresh\n"
        "python -m macro_credit_forecast_bcb.pipeline.forecast\n"
        "python -m macro_credit_forecast_bcb.pipeline.backtest\n"
        "python -m macro_credit_forecast_bcb.pipeline.audit",
        language="bash",
    )


def dataframe_download(frame: pd.DataFrame, filename: str, label: str = "Download CSV") -> None:
    st.download_button(
        label,
        frame.to_csv(index=False).encode("utf-8"),
        file_name=filename,
        mime="text/csv",
    )


def apply_app_style() -> None:
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 2.1rem;
            padding-bottom: 3rem;
            max-width: 1180px;
        }
        h1, h2, h3 {
            letter-spacing: 0;
        }
        [data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            padding: 14px 16px;
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
        }
        [data-testid="stMetricLabel"] {
            color: #64748b;
        }
        [data-testid="stMetricValue"] {
            color: #0f172a;
            font-size: 1.55rem;
        }
        .status-pill {
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
            border-radius: 999px;
            padding: 0.28rem 0.68rem;
            font-size: 0.82rem;
            font-weight: 650;
            border: 1px solid;
        }
        .status-ok {
            color: #166534;
            background: #dcfce7;
            border-color: #86efac;
        }
        .status-warning {
            color: #92400e;
            background: #fef3c7;
            border-color: #fcd34d;
        }
        .status-fail {
            color: #991b1b;
            background: #fee2e2;
            border-color: #fca5a5;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def quality_status(report: pd.DataFrame) -> str:
    if report.empty:
        return "missing"
    if (report["status"] == "fail").any():
        return "fail"
    if (report["status"] == "warning").any():
        return "warning"
    return "ok"


def render_quality_pill(status: str) -> None:
    status_class = "status-ok" if status == "ok" else "status-warning" if status == "warning" else "status-fail"
    label = {"ok": "Dados OK", "warning": "Dados com alertas", "fail": "Dados com falhas"}.get(
        status,
        "Dados não validados",
    )
    st.markdown(
        f'<span class="status-pill {status_class}">{label}</span>',
        unsafe_allow_html=True,
    )


def render_status_pill(status: str) -> None:
    normalized = {"pass": "ok", "ok": "ok", "warning": "warning", "fail": "fail"}.get(status, "warning")
    status_class = "status-ok" if normalized == "ok" else "status-warning" if normalized == "warning" else "status-fail"
    label = {"pass": "Pass", "ok": "OK", "warning": "Warning", "fail": "Fail"}.get(status, status.title())
    st.markdown(
        f'<span class="status-pill {status_class}">{label}</span>',
        unsafe_allow_html=True,
    )
