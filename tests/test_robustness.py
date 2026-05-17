from __future__ import annotations

import numpy as np
import pandas as pd

from macro_credit_forecast_bcb.models.robustness import (
    build_transformation_candidates,
    default_var_specifications,
    month_dummies,
    select_recommended_models,
)


def test_month_dummies_omit_january_and_keep_stable_columns() -> None:
    index = pd.date_range("2024-01-31", periods=12, freq="ME")

    dummies = month_dummies(index)

    assert list(dummies.columns) == [f"month_{month}" for month in range(2, 13)]
    assert dummies.iloc[0].sum() == 0.0
    assert dummies.iloc[1]["month_2"] == 1.0


def test_default_var_specifications_include_requested_robustness_models() -> None:
    specs = default_var_specifications(
        ["ipca", "selic", "spread", "dlog_concessoes_reais", "inadimplencia"],
        base_maxlags=6,
        robustness_maxlags=12,
        criterion="bic",
    )
    names = {spec.name for spec in specs}

    assert {"VAR_maxlags_12", "VAR_seasonal_dummies", "VAR_core_rates", "VAR_credit_block"} <= names
    assert next(spec for spec in specs if spec.name == "VAR_seasonal_dummies").seasonal_dummies


def test_select_recommended_models_prefers_lowest_rmse_and_keeps_dm_context() -> None:
    metrics = pd.DataFrame(
        {
            "model": ["VAR", "VAR_seasonal_dummies", "random_walk"],
            "variable": ["ipca", "ipca", "ipca"],
            "horizon": [1, 1, 1],
            "rmse": [2.0, 1.5, 1.0],
            "mae": [1.5, 1.2, 0.8],
        }
    )
    dm_tests = pd.DataFrame(
        {
            "variable": ["ipca"],
            "horizon": [1],
            "benchmark": ["random_walk"],
            "dm_stat": [2.2],
            "pvalue": [0.04],
        }
    )

    recommendations = select_recommended_models(metrics, dm_tests)

    assert recommendations.loc[0, "recommended_model"] == "random_walk"
    assert recommendations.loc[0, "recommendation_type"] == "benchmark"
    assert "Diebold-Mariano" in recommendations.loc[0, "reason"]


def test_select_recommended_models_preserves_base_var_for_small_gain() -> None:
    metrics = pd.DataFrame(
        {
            "model": ["VAR", "VAR_core_rates"],
            "variable": ["inadimplencia", "inadimplencia"],
            "horizon": [12, 12],
            "rmse": [1.00, 0.98],
            "mae": [0.80, 0.79],
        }
    )

    recommendations = select_recommended_models(metrics)

    assert recommendations.loc[0, "recommended_model"] == "VAR"
    assert recommendations.loc[0, "recommendation_type"] == "base_var"
    assert "preservado" in recommendations.loc[0, "reason"]


def test_build_transformation_candidates_marks_current_model_transformations() -> None:
    index = pd.date_range("2020-01-31", periods=36, freq="ME")
    rng = np.random.default_rng(123)
    raw = pd.DataFrame(
        {
            "ipca": rng.normal(0.4, 0.2, size=36),
            "selic_meta": np.linspace(2.0, 12.0, 36),
            "spread_credito_total": rng.normal(20.0, 1.0, size=36),
            "concessoes_credito_total": np.linspace(1000.0, 1300.0, 36),
            "inadimplencia_total": rng.normal(3.0, 0.2, size=36),
        },
        index=index,
    )
    model = pd.DataFrame(
        {
            "concessoes_reais": np.linspace(1000.0, 1200.0, 36),
            "dlog_concessoes_reais": rng.normal(0.2, 0.1, size=36),
        },
        index=index,
    )

    candidates = build_transformation_candidates(raw, model)
    current = set(candidates.loc[candidates["current_model"], "candidate"])

    assert {"level_rate", "real_log_difference"} <= current
    assert {"adf_pvalue", "kpss_pvalue", "chosen_order", "recommendation"} <= set(candidates.columns)
