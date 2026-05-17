from __future__ import annotations

import logging

import pandas as pd

from macro_credit_forecast_bcb.models.benchmarks import error_metrics, forecast_benchmark
from macro_credit_forecast_bcb.models.forecast import forecast_var
from macro_credit_forecast_bcb.models.model_selection import select_var_lag
from macro_credit_forecast_bcb.models.var_model import fit_var

LOGGER = logging.getLogger(__name__)


BENCHMARK_MODELS = ["random_walk", "ar1", "moving_average_12m", "seasonal_naive"]


def rolling_backtest(
    frame: pd.DataFrame,
    *,
    horizons: list[int] | tuple[int, ...] = (1, 3, 6, 12),
    initial_window: int = 72,
    expanding: bool = True,
    maxlags: int = 6,
    criterion: str = "bic",
    variables: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = frame.dropna().astype(float)
    variables = variables or list(data.columns)
    max_horizon = max(horizons)
    if data.shape[0] <= initial_window + max_horizon:
        LOGGER.warning("Insufficient observations for requested backtest")
        return pd.DataFrame(), pd.DataFrame()

    records: list[dict[str, object]] = []
    for end in range(initial_window, data.shape[0] - max_horizon + 1):
        train = data.iloc[:end] if expanding else data.iloc[end - initial_window : end]
        origin = train.index.max()
        actual_window = data.iloc[end : end + max_horizon]

        try:
            selection = select_var_lag(train[variables], maxlags=maxlags, criterion=criterion)
            result = fit_var(train[variables], int(selection["selected_lag"]))
            var_forecast = forecast_var(result, train[variables], steps=max_horizon)
        except Exception as exc:
            LOGGER.warning("Skipping VAR backtest origin %s: %s", origin.date(), exc)
            var_forecast = pd.DataFrame()

        for horizon in horizons:
            actual_date = actual_window.index[horizon - 1]
            for variable in variables:
                actual = float(actual_window.loc[actual_date, variable])
                if not var_forecast.empty:
                    match = var_forecast[
                        (var_forecast["variable"] == variable) & (var_forecast["horizon"] == horizon)
                    ]
                    if not match.empty:
                        row = match.iloc[0]
                        records.append(
                            {
                                "origin": origin,
                                "target_date": actual_date,
                                "model": "VAR",
                                "variable": variable,
                                "horizon": horizon,
                                "actual": actual,
                                "forecast": float(row["forecast"]),
                                "lower_68": float(row["lower_68"]),
                                "upper_68": float(row["upper_68"]),
                                "lower_95": float(row["lower_95"]),
                                "upper_95": float(row["upper_95"]),
                            }
                        )

                for model in BENCHMARK_MODELS:
                    values = forecast_benchmark(train[variable], model, max_horizon)
                    records.append(
                        {
                            "origin": origin,
                            "target_date": actual_date,
                            "model": model,
                            "variable": variable,
                            "horizon": horizon,
                            "actual": actual,
                            "forecast": float(values[horizon - 1]),
                            "lower_68": pd.NA,
                            "upper_68": pd.NA,
                            "lower_95": pd.NA,
                            "upper_95": pd.NA,
                        }
                    )

    record_frame = pd.DataFrame(records)
    return record_frame, error_metrics(record_frame)
