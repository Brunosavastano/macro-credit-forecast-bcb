from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss


def _safe_adf(series: pd.Series) -> tuple[float, str | None]:
    values = series.dropna().astype(float)
    if values.nunique() <= 1 or len(values) < 12:
        return np.nan, "Insufficient variation or observations"
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = adfuller(values, autolag="AIC")
        return float(result[1]), None
    except Exception as exc:
        return np.nan, str(exc)


def _safe_kpss(series: pd.Series) -> tuple[float, str | None]:
    values = series.dropna().astype(float)
    if values.nunique() <= 1 or len(values) < 12:
        return np.nan, "Insufficient variation or observations"
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = kpss(values, regression="c", nlags="auto")
        return float(result[1]), None
    except Exception as exc:
        return np.nan, str(exc)


def choose_order(adf_pvalue: float, kpss_pvalue: float, variable: str) -> tuple[str, str]:
    adf_rejects_unit_root = pd.notna(adf_pvalue) and adf_pvalue < 0.05
    kpss_rejects_stationary = pd.notna(kpss_pvalue) and kpss_pvalue < 0.05

    if adf_rejects_unit_root and not kpss_rejects_stationary:
        return "I(0)", "ADF rejects unit root and KPSS does not reject stationarity."
    if not adf_rejects_unit_root and kpss_rejects_stationary:
        return "I(1)", "ADF does not reject unit root and KPSS rejects stationarity."
    if variable in {"ipca", "dlog_concessoes_reais"}:
        return "I(0)", "Tests are mixed; default to stationary rate/growth transformation."
    return "ambiguous", "ADF/KPSS evidence is mixed; inspect economics, charts and diagnostics."


def run_stationarity_tests(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for column in frame.columns:
        series = frame[column].dropna()
        adf_pvalue, adf_error = _safe_adf(series)
        kpss_pvalue, kpss_error = _safe_kpss(series)
        order, justification = choose_order(adf_pvalue, kpss_pvalue, column)
        rows.append(
            {
                "series": column,
                "transformation": column,
                "adf_pvalue": adf_pvalue,
                "kpss_pvalue": kpss_pvalue,
                "chosen_order": order,
                "justification": justification,
                "adf_error": adf_error,
                "kpss_error": kpss_error,
                "nobs": int(series.shape[0]),
            }
        )
    return pd.DataFrame(rows)

