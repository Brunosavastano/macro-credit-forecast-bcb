from __future__ import annotations

import numpy as np
import pandas as pd


def random_walk_forecast(series: pd.Series, steps: int) -> np.ndarray:
    last = float(series.dropna().iloc[-1])
    return np.repeat(last, steps)


def moving_average_forecast(series: pd.Series, steps: int, window: int = 12) -> np.ndarray:
    values = series.dropna().astype(float)
    mean = float(values.tail(min(window, len(values))).mean())
    return np.repeat(mean, steps)


def seasonal_naive_forecast(series: pd.Series, steps: int, season: int = 12) -> np.ndarray:
    values = series.dropna().astype(float)
    forecasts = []
    for h in range(1, steps + 1):
        idx = -season + ((h - 1) % season)
        if len(values) >= abs(idx):
            forecasts.append(float(values.iloc[idx]))
        else:
            forecasts.append(float(values.iloc[-1]))
    return np.asarray(forecasts)


def ar1_forecast(series: pd.Series, steps: int) -> np.ndarray:
    values = series.dropna().astype(float)
    if len(values) < 8 or values.nunique() <= 1:
        return random_walk_forecast(values, steps)
    y = values.iloc[1:].to_numpy()
    x = np.column_stack([np.ones(len(values) - 1), values.iloc[:-1].to_numpy()])
    try:
        beta = np.linalg.lstsq(x, y, rcond=None)[0]
    except np.linalg.LinAlgError:
        return random_walk_forecast(values, steps)
    current = float(values.iloc[-1])
    forecasts = []
    for _ in range(steps):
        current = float(beta[0] + beta[1] * current)
        forecasts.append(current)
    return np.asarray(forecasts)


def forecast_benchmark(series: pd.Series, model: str, steps: int) -> np.ndarray:
    if model == "random_walk":
        return random_walk_forecast(series, steps)
    if model == "ar1":
        return ar1_forecast(series, steps)
    if model == "moving_average_12m":
        return moving_average_forecast(series, steps, window=12)
    if model == "seasonal_naive":
        return seasonal_naive_forecast(series, steps, season=12)
    raise ValueError(f"Unknown benchmark model: {model}")


def error_metrics(records: pd.DataFrame) -> pd.DataFrame:
    if records.empty:
        return pd.DataFrame()
    frame = records.copy()
    frame["error"] = frame["forecast"] - frame["actual"]
    frame["abs_error"] = frame["error"].abs()
    frame["sq_error"] = frame["error"] ** 2
    denominator = (frame["forecast"].abs() + frame["actual"].abs()) / 2
    frame["smape_component"] = np.where(denominator > 0, frame["abs_error"] / denominator, np.nan)

    def directional_accuracy(group: pd.DataFrame) -> float:
        ordered = group.sort_values("origin")
        actual_change = ordered["actual"].diff()
        forecast_change = ordered["forecast"].diff()
        valid = actual_change.notna() & forecast_change.notna()
        if valid.sum() == 0:
            return np.nan
        return float((np.sign(actual_change[valid]) == np.sign(forecast_change[valid])).mean())

    metrics = (
        frame.groupby(["model", "variable", "horizon"], dropna=False)
        .agg(
            mae=("abs_error", "mean"),
            rmse=("sq_error", lambda x: float(np.sqrt(np.mean(x)))),
            smape=("smape_component", "mean"),
            nobs=("error", "count"),
        )
        .reset_index()
    )
    direction_rows = []
    for keys, group in frame.groupby(["model", "variable", "horizon"], dropna=False):
        model, variable, horizon = keys
        direction_rows.append(
            {
                "model": model,
                "variable": variable,
                "horizon": horizon,
                "directional_accuracy": directional_accuracy(group),
            }
        )
    direction = pd.DataFrame(direction_rows)
    return metrics.merge(direction, on=["model", "variable", "horizon"], how="left")
