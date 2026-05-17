from __future__ import annotations

import numpy as np
import pandas as pd


REQUIRED_RAW_COLUMNS = {
    "ipca",
    "selic_meta",
    "spread_credito_total",
    "concessoes_credito_total",
    "inadimplencia_total",
}


def validate_time_series_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Sort, deduplicate, and validate a time-indexed frame."""
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise TypeError("DataFrame index must be a DatetimeIndex")
    cleaned = frame.copy().sort_index()
    cleaned = cleaned[~cleaned.index.duplicated(keep="last")]
    if cleaned.index.hasnans:
        raise ValueError("DatetimeIndex contains NaT values")
    inferred = pd.infer_freq(cleaned.index)
    if inferred:
        cleaned = cleaned.asfreq(inferred)
    return cleaned


def compound_ipca_index(ipca_monthly_pct: pd.Series, base: float = 100.0) -> pd.Series:
    """Build a price index from monthly IPCA percentage changes."""
    factors = 1.0 + ipca_monthly_pct.astype(float) / 100.0
    index = factors.cumprod() * base
    index.name = "ipca_price_index"
    return index


def deflate_nominal_series(
    nominal: pd.Series,
    price_index: pd.Series,
    *,
    reference: str = "last",
) -> pd.Series:
    """Deflate a nominal series using a price index.

    The default returns values in last-observation prices.
    """
    aligned = pd.concat([nominal.astype(float), price_index.astype(float)], axis=1).dropna()
    if aligned.empty:
        return pd.Series(dtype="float64", name=f"{nominal.name}_real")
    base_price = aligned.iloc[-1, 1] if reference == "last" else aligned.iloc[0, 1]
    real = aligned.iloc[:, 0] / aligned.iloc[:, 1] * base_price
    real.name = f"{nominal.name}_real"
    return real


def log_diff(series: pd.Series, scale: float = 100.0) -> pd.Series:
    if (series <= 0).any():
        raise ValueError(f"Cannot take log of non-positive values in {series.name}")
    transformed = np.log(series.astype(float)).diff() * scale
    transformed.name = f"dlog_{series.name}"
    return transformed


def rolling_compounded_rate(monthly_pct: pd.Series, window: int = 12) -> pd.Series:
    compounded = (1.0 + monthly_pct.astype(float) / 100.0).rolling(window).apply(np.prod, raw=True)
    output = (compounded - 1.0) * 100.0
    output.name = f"{monthly_pct.name}_{window}m"
    return output


def build_model_dataset(raw_monthly: pd.DataFrame) -> pd.DataFrame:
    """Create the base stationary VAR dataset from raw monthly BCB series."""
    missing = REQUIRED_RAW_COLUMNS.difference(raw_monthly.columns)
    if missing:
        raise ValueError(f"Missing required raw columns: {sorted(missing)}")

    frame = validate_time_series_frame(raw_monthly)
    ipca_index = compound_ipca_index(frame["ipca"])
    concessoes_reais = deflate_nominal_series(frame["concessoes_credito_total"], ipca_index)
    concessoes_reais.name = "concessoes_reais"
    dlog_concessoes = log_diff(concessoes_reais)
    dlog_concessoes.name = "dlog_concessoes_reais"

    model = pd.DataFrame(
        {
            "ipca": frame["ipca"],
            "selic": frame["selic_meta"],
            "spread": frame["spread_credito_total"],
            "dlog_concessoes_reais": dlog_concessoes,
            "inadimplencia": frame["inadimplencia_total"],
            "concessoes_reais": concessoes_reais,
            "ipca_12m": rolling_compounded_rate(frame["ipca"], 12),
        }
    )
    model.index.name = "date"
    return validate_time_series_frame(model).dropna(subset=[
        "ipca",
        "selic",
        "spread",
        "dlog_concessoes_reais",
        "inadimplencia",
    ])


def model_columns() -> list[str]:
    return ["ipca", "selic", "spread", "dlog_concessoes_reais", "inadimplencia"]
