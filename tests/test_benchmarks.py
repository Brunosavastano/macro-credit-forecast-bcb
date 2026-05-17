from __future__ import annotations

import pandas as pd

from macro_credit_forecast_bcb.models.benchmarks import (
    ar1_forecast,
    error_metrics,
    moving_average_forecast,
    random_walk_forecast,
    seasonal_naive_forecast,
)


def test_benchmark_forecasts_have_requested_length() -> None:
    series = pd.Series(range(1, 25), index=pd.date_range("2020-01-31", periods=24, freq="ME"))

    assert len(random_walk_forecast(series, 3)) == 3
    assert len(moving_average_forecast(series, 3)) == 3
    assert len(seasonal_naive_forecast(series, 3)) == 3
    assert len(ar1_forecast(series, 3)) == 3


def test_error_metrics_groups_by_model_variable_horizon() -> None:
    records = pd.DataFrame(
        {
            "origin": pd.date_range("2022-01-31", periods=3, freq="ME"),
            "model": ["m1", "m1", "m1"],
            "variable": ["ipca", "ipca", "ipca"],
            "horizon": [1, 1, 1],
            "actual": [1.0, 2.0, 3.0],
            "forecast": [1.1, 1.8, 3.2],
        }
    )

    metrics = error_metrics(records)

    assert metrics.loc[0, "mae"] > 0
    assert metrics.loc[0, "rmse"] > 0
    assert metrics.loc[0, "nobs"] == 3

