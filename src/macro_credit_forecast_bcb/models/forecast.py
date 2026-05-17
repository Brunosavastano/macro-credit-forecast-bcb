from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.vector_ar.var_model import VARResults

from macro_credit_forecast_bcb.features.transformations import rolling_compounded_rate
from macro_credit_forecast_bcb.utils.dates import next_month_ends


def forecast_var(
    result: VARResults,
    history: pd.DataFrame,
    *,
    steps: int = 12,
) -> pd.DataFrame:
    """Return long-format VAR forecasts with 68% and 95% analytic intervals."""
    data = history[result.names].dropna().astype(float)
    last_values = data.values[-result.k_ar :]
    mean_95, lower_95, upper_95 = result.forecast_interval(last_values, steps=steps, alpha=0.05)
    mean_68, lower_68, upper_68 = result.forecast_interval(last_values, steps=steps, alpha=0.32)
    if not np.allclose(mean_95, mean_68):
        mean = mean_95
    else:
        mean = mean_95

    dates = next_month_ends(data.index.max(), steps)
    records: list[dict[str, object]] = []
    for step_idx, date in enumerate(dates, start=1):
        for var_idx, variable in enumerate(result.names):
            records.append(
                {
                    "date": date,
                    "horizon": step_idx,
                    "variable": variable,
                    "forecast": float(mean[step_idx - 1, var_idx]),
                    "lower_68": float(lower_68[step_idx - 1, var_idx]),
                    "upper_68": float(upper_68[step_idx - 1, var_idx]),
                    "lower_95": float(lower_95[step_idx - 1, var_idx]),
                    "upper_95": float(upper_95[step_idx - 1, var_idx]),
                    "model": f"VAR({result.k_ar})",
                }
            )
    output = pd.DataFrame(records)
    output = add_ipca_12m_forecast(output, history)
    return add_concessoes_reais_forecast(output, history)


def add_ipca_12m_forecast(forecast: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    if "ipca" not in forecast["variable"].unique() or "ipca" not in history.columns:
        return forecast
    ipca_forecast = forecast.loc[forecast["variable"] == "ipca", ["date", "forecast", "model"]]
    ipca_path = pd.concat(
        [
            history[["ipca"]].rename(columns={"ipca": "forecast"}).assign(model="observed"),
            ipca_forecast.set_index("date"),
        ],
        axis=0,
    )
    ipca_12m = rolling_compounded_rate(ipca_path["forecast"], 12)
    future = ipca_12m.loc[ipca_forecast["date"]]
    rows = [
        {
            "date": date,
            "horizon": int((date.to_period("M") - history.index.max().to_period("M")).n),
            "variable": "ipca_12m",
            "forecast": float(value),
            "lower_68": np.nan,
            "upper_68": np.nan,
            "lower_95": np.nan,
            "upper_95": np.nan,
            "model": str(ipca_forecast.loc[ipca_forecast["date"] == date, "model"].iloc[0]),
        }
        for date, value in future.items()
    ]
    return pd.concat([forecast, pd.DataFrame(rows)], ignore_index=True)


def _rebuild_level_path(last_level: float, growth_pct: pd.Series) -> pd.Series:
    levels = []
    current = float(last_level)
    for value in growth_pct.astype(float):
        current *= float(np.exp(value / 100.0))
        levels.append(current)
    return pd.Series(levels, index=growth_pct.index)


def add_concessoes_reais_forecast(forecast: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    if "concessoes_reais" not in history.columns:
        return forecast
    growth = forecast.loc[
        forecast["variable"] == "dlog_concessoes_reais",
        ["date", "horizon", "forecast", "lower_68", "upper_68", "lower_95", "upper_95", "model"],
    ].sort_values("date")
    if growth.empty:
        return forecast

    last_level = float(history["concessoes_reais"].dropna().iloc[-1])
    indexed = growth.set_index("date")
    rebuilt = pd.DataFrame(
        {
            "horizon": indexed["horizon"],
            "variable": "concessoes_reais",
            "forecast": _rebuild_level_path(last_level, indexed["forecast"]),
            "lower_68": _rebuild_level_path(last_level, indexed["lower_68"]),
            "upper_68": _rebuild_level_path(last_level, indexed["upper_68"]),
            "lower_95": _rebuild_level_path(last_level, indexed["lower_95"]),
            "upper_95": _rebuild_level_path(last_level, indexed["upper_95"]),
            "model": indexed["model"],
        }
    ).reset_index(names="date")
    return pd.concat([forecast, rebuilt], ignore_index=True)
