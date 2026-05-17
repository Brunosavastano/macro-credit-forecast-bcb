from __future__ import annotations

import numpy as np
import pandas as pd

from macro_credit_forecast_bcb.models.audit import (
    build_model_scorecard,
    diebold_mariano_test,
    interval_coverage,
)


def test_diebold_mariano_identifies_lower_var_loss() -> None:
    index = pd.MultiIndex.from_arrays(
        [pd.date_range("2020-01-31", periods=30, freq="ME"), pd.date_range("2020-02-29", periods=30, freq="ME")],
        names=["origin", "target_date"],
    )
    var_errors = pd.Series(np.repeat(0.2, 30), index=index)
    benchmark_errors = pd.Series(np.repeat(1.0, 30), index=index)

    result = diebold_mariano_test(var_errors, benchmark_errors, horizon=1)

    assert result["status"] == "ok"
    assert result["dm_stat"] < 0
    assert result["pvalue"] < 0.10


def test_diebold_mariano_handles_equal_errors() -> None:
    index = pd.MultiIndex.from_arrays(
        [pd.date_range("2020-01-31", periods=12, freq="ME"), pd.date_range("2020-02-29", periods=12, freq="ME")],
        names=["origin", "target_date"],
    )
    errors = pd.Series(np.repeat(0.5, 12), index=index)

    result = diebold_mariano_test(errors, errors, horizon=1)

    assert result["status"] == "zero_variance"


def test_interval_coverage_computes_expected_rates() -> None:
    records = pd.DataFrame(
        {
            "model": ["VAR", "VAR", "VAR", "random_walk"],
            "variable": ["ipca", "ipca", "ipca", "ipca"],
            "horizon": [1, 1, 1, 1],
            "actual": [1.0, 2.0, 5.0, 1.0],
            "forecast": [1.0, 2.0, 3.0, 1.0],
            "lower_68": [0.0, 1.5, 4.5, pd.NA],
            "upper_68": [1.5, 2.5, 5.5, pd.NA],
            "lower_95": [0.0, 1.0, 2.0, pd.NA],
            "upper_95": [2.0, 3.0, 4.0, pd.NA],
        }
    )

    coverage = interval_coverage(records)

    assert coverage.loc[0, "coverage_68"] == 1.0
    assert round(coverage.loc[0, "coverage_95"], 6) == round(2 / 3, 6)


def test_scorecard_flags_rejected_residual_whiteness() -> None:
    scorecard = build_model_scorecard(
        data_quality=pd.DataFrame({"status": ["ok"]}),
        stationarity=pd.DataFrame({"chosen_order": ["I(0)"]}),
        model_summary={
            "diagnostics": {
                "is_stable": True,
                "lag_order": 1,
                "whiteness": {"conclusion": "reject"},
                "normality": {"conclusion": "reject"},
            }
        },
        residual_diagnostics=pd.DataFrame({"status": ["fail"]}),
        audit_metrics=pd.DataFrame(
            {
                "model": ["VAR"],
                "variable": ["ipca"],
                "horizon": [1],
                "is_best_rmse": [True],
            }
        ),
        coverage=pd.DataFrame({"coverage_95": [0.9]}),
        dm_tests=pd.DataFrame(),
    )

    residual_status = scorecard.loc[scorecard["dimension"] == "Resíduos", "status"].iloc[0]

    assert residual_status == "fail"

