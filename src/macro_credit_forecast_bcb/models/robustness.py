from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from macro_credit_forecast_bcb.features.stationarity import stationarity_check
from macro_credit_forecast_bcb.features.transformations import log_diff, rolling_compounded_rate
from macro_credit_forecast_bcb.models.model_selection import select_var_lag
from macro_credit_forecast_bcb.models.var_model import fit_var, var_diagnostics

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class VarSpecification:
    name: str
    variables: tuple[str, ...]
    maxlags: int
    criterion: str = "bic"
    seasonal_dummies: bool = False
    block: str = "full"
    description: str = ""


def month_dummies(index: pd.Index) -> pd.DataFrame:
    """Return 11 monthly dummies, leaving January as the omitted category."""
    dates = pd.DatetimeIndex(index)
    dummies = pd.get_dummies(dates.month, prefix="month", dtype=float)
    dummies.index = dates
    expected = [f"month_{month}" for month in range(2, 13)]
    return dummies.reindex(columns=expected, fill_value=0.0)


def default_var_specifications(
    variables: list[str] | tuple[str, ...],
    *,
    base_maxlags: int = 6,
    robustness_maxlags: int = 12,
    criterion: str = "bic",
) -> list[VarSpecification]:
    full = tuple(variables)
    specs = [
        VarSpecification(
            name="VAR_maxlags_12",
            variables=full,
            maxlags=robustness_maxlags,
            criterion=criterion,
            block="full",
            description="VAR completo com busca de defasagens até 12 meses.",
        ),
        VarSpecification(
            name="VAR_seasonal_dummies",
            variables=full,
            maxlags=base_maxlags,
            criterion=criterion,
            seasonal_dummies=True,
            block="full",
            description="VAR completo com dummies mensais determinísticas.",
        ),
    ]
    core = tuple(variable for variable in ("ipca", "selic", "spread", "inadimplencia") if variable in full)
    credit = tuple(
        variable
        for variable in ("selic", "spread", "dlog_concessoes_reais", "inadimplencia")
        if variable in full
    )
    if len(core) >= 2:
        specs.append(
            VarSpecification(
                name="VAR_core_rates",
                variables=core,
                maxlags=robustness_maxlags,
                criterion=criterion,
                block="core_rates",
                description="VAR menor para inflação, juros, spread e inadimplência.",
            )
        )
    if len(credit) >= 2:
        specs.append(
            VarSpecification(
                name="VAR_credit_block",
                variables=credit,
                maxlags=robustness_maxlags,
                criterion=criterion,
                block="credit",
                description="VAR menor focado no bloco de crédito.",
            )
        )
    return specs


def _fit_specification(train: pd.DataFrame, specification: VarSpecification) -> tuple[Any, dict[str, object]]:
    variables = list(specification.variables)
    exog = month_dummies(train.index) if specification.seasonal_dummies else None
    selection = select_var_lag(
        train[variables],
        maxlags=specification.maxlags,
        criterion=specification.criterion,
        exog=exog,
    )
    result = fit_var(train[variables], int(selection["selected_lag"]), exog=exog)
    return result, selection


def _forecast_result(
    result: Any,
    train: pd.DataFrame,
    *,
    steps: int,
    model_name: str,
    exog_future: pd.DataFrame | None = None,
) -> pd.DataFrame:
    data = train[result.names].dropna().astype(float)
    last_values = data.values[-result.k_ar :]
    future_values = exog_future.to_numpy(dtype=float) if exog_future is not None else None
    mean_95, lower_95, upper_95 = result.forecast_interval(
        last_values,
        steps=steps,
        alpha=0.05,
        exog_future=future_values,
    )
    _, lower_68, upper_68 = result.forecast_interval(
        last_values,
        steps=steps,
        alpha=0.32,
        exog_future=future_values,
    )
    records: list[dict[str, object]] = []
    for step_idx in range(steps):
        for var_idx, variable in enumerate(result.names):
            records.append(
                {
                    "step": step_idx + 1,
                    "variable": variable,
                    "forecast": float(mean_95[step_idx, var_idx]),
                    "lower_68": float(lower_68[step_idx, var_idx]),
                    "upper_68": float(upper_68[step_idx, var_idx]),
                    "lower_95": float(lower_95[step_idx, var_idx]),
                    "upper_95": float(upper_95[step_idx, var_idx]),
                    "model": model_name,
                }
            )
    return pd.DataFrame(records)


def rolling_specification_backtest(
    frame: pd.DataFrame,
    specifications: list[VarSpecification],
    *,
    horizons: list[int] | tuple[int, ...] = (1, 3, 6, 12),
    initial_window: int = 72,
    expanding: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = frame.dropna().astype(float)
    max_horizon = max(horizons)
    if data.shape[0] <= initial_window + max_horizon:
        LOGGER.warning("Insufficient observations for robustness backtest")
        return pd.DataFrame(), pd.DataFrame()

    records: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    for end in range(initial_window, data.shape[0] - max_horizon + 1):
        train_all = data.iloc[:end] if expanding else data.iloc[end - initial_window : end]
        actual_window = data.iloc[end : end + max_horizon]
        origin = train_all.index.max()
        for specification in specifications:
            variables = list(specification.variables)
            train = train_all[variables].dropna()
            if train.shape[0] < initial_window // 2:
                continue
            try:
                result, selection = _fit_specification(train, specification)
                exog_future = (
                    month_dummies(actual_window.index[:max_horizon])
                    if specification.seasonal_dummies
                    else None
                )
                forecast = _forecast_result(
                    result,
                    train,
                    steps=max_horizon,
                    model_name=specification.name,
                    exog_future=exog_future,
                )
            except Exception as exc:
                failures.append(
                    {
                        "origin": origin,
                        "model": specification.name,
                        "block": specification.block,
                        "error": str(exc),
                    }
                )
                continue

            for horizon in horizons:
                actual_date = actual_window.index[horizon - 1]
                for variable in variables:
                    match = forecast[
                        (forecast["variable"] == variable)
                        & (forecast["step"] == horizon)
                    ]
                    if match.empty:
                        continue
                    row = match.iloc[0]
                    records.append(
                        {
                            "origin": origin,
                            "target_date": actual_date,
                            "model": specification.name,
                            "variable": variable,
                            "horizon": horizon,
                            "actual": float(actual_window.loc[actual_date, variable]),
                            "forecast": float(row["forecast"]),
                            "lower_68": float(row["lower_68"]),
                            "upper_68": float(row["upper_68"]),
                            "lower_95": float(row["lower_95"]),
                            "upper_95": float(row["upper_95"]),
                            "selected_lag": int(selection["selected_lag"]),
                            "block": specification.block,
                            "seasonal_dummies": specification.seasonal_dummies,
                        }
                    )
    return pd.DataFrame(records), pd.DataFrame(failures)


def evaluate_var_specifications(frame: pd.DataFrame, specifications: list[VarSpecification]) -> pd.DataFrame:
    data = frame.dropna().astype(float)
    rows: list[dict[str, object]] = []
    for specification in specifications:
        try:
            result, selection = _fit_specification(data[list(specification.variables)], specification)
            diagnostics = var_diagnostics(result)
            whiteness = diagnostics.get("whiteness", {})
            normality = diagnostics.get("normality", {})
            status = "pass"
            if not diagnostics.get("is_stable", False):
                status = "fail"
            elif (
                isinstance(whiteness, dict)
                and whiteness.get("conclusion") == "reject"
            ) or (
                isinstance(normality, dict)
                and normality.get("conclusion") == "reject"
            ):
                status = "warning"
            rows.append(
                {
                    "model": specification.name,
                    "block": specification.block,
                    "variables": ", ".join(specification.variables),
                    "seasonal_dummies": specification.seasonal_dummies,
                    "selected_lag": int(selection["selected_lag"]),
                    "maxlags": specification.maxlags,
                    "criterion": specification.criterion,
                    "is_stable": bool(diagnostics.get("is_stable", False)),
                    "aic": diagnostics.get("aic"),
                    "bic": diagnostics.get("bic"),
                    "roots_abs_min": diagnostics.get("roots_abs_min"),
                    "whiteness": whiteness.get("conclusion") if isinstance(whiteness, dict) else None,
                    "normality": normality.get("conclusion") if isinstance(normality, dict) else None,
                    "status": status,
                    "description": specification.description,
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "model": specification.name,
                    "block": specification.block,
                    "variables": ", ".join(specification.variables),
                    "seasonal_dummies": specification.seasonal_dummies,
                    "selected_lag": np.nan,
                    "maxlags": specification.maxlags,
                    "criterion": specification.criterion,
                    "is_stable": False,
                    "aic": np.nan,
                    "bic": np.nan,
                    "roots_abs_min": np.nan,
                    "whiteness": None,
                    "normality": None,
                    "status": "fail",
                    "description": specification.description,
                    "error": str(exc),
                }
            )
    return pd.DataFrame(rows)


def _maybe_log(series: pd.Series) -> pd.Series:
    values = series.dropna().astype(float)
    if (values <= 0).any():
        return pd.Series(dtype=float)
    output = np.log(values)
    output.name = f"log_{series.name}"
    return output


def build_transformation_candidates(raw: pd.DataFrame, model: pd.DataFrame) -> pd.DataFrame:
    candidates: list[dict[str, object]] = []

    def add_candidate(
        *,
        raw_variable: str,
        model_variable: str,
        candidate: str,
        series: pd.Series,
        current_model: bool,
        note: str,
    ) -> None:
        clean = series.dropna().astype(float)
        if clean.empty:
            return
        check = stationarity_check(clean, model_variable)
        status = "pass" if check["chosen_order"] == "I(0)" else "warning" if check["chosen_order"] == "ambiguous" else "fail"
        recommendation = "Candidata estacionária para teste de robustez." if status == "pass" else "Usar apenas com justificativa econômica e diagnóstico adicional."
        if current_model and status != "pass":
            recommendation = "Transformação atual preservada por critério econômico; revisar diagnóstico e benchmark."
        candidates.append(
            {
                "raw_variable": raw_variable,
                "model_variable": model_variable,
                "candidate": candidate,
                "current_model": current_model,
                "nobs": int(clean.shape[0]),
                "adf_pvalue": check["adf_pvalue"],
                "kpss_pvalue": check["kpss_pvalue"],
                "chosen_order": check["chosen_order"],
                "status": status,
                "recommendation": recommendation,
                "note": note,
            }
        )

    rate_specs = [
        ("ipca", "ipca"),
        ("selic_meta", "selic"),
        ("spread_credito_total", "spread"),
        ("inadimplencia_total", "inadimplencia"),
    ]
    for raw_variable, model_variable in rate_specs:
        if raw_variable not in raw.columns:
            continue
        series = raw[raw_variable]
        add_candidate(
            raw_variable=raw_variable,
            model_variable=model_variable,
            candidate="level_rate",
            series=series,
            current_model=True,
            note="Nível percentual usado no VAR base.",
        )
        add_candidate(
            raw_variable=raw_variable,
            model_variable=model_variable,
            candidate="first_difference",
            series=series.diff(),
            current_model=False,
            note="Diferença mensal do nível percentual.",
        )

    if "ipca" in raw.columns:
        add_candidate(
            raw_variable="ipca",
            model_variable="ipca_12m",
            candidate="rolling_12m_compounded",
            series=rolling_compounded_rate(raw["ipca"], 12),
            current_model=False,
            note="IPCA acumulado em 12 meses; útil para comunicação, não para VAR base.",
        )

    if "concessoes_credito_total" in raw.columns:
        nominal = raw["concessoes_credito_total"]
        add_candidate(
            raw_variable="concessoes_credito_total",
            model_variable="concessoes_reais",
            candidate="nominal_level",
            series=nominal,
            current_model=False,
            note="Nível nominal sem deflação.",
        )
        log_nominal = _maybe_log(nominal)
        if not log_nominal.empty:
            add_candidate(
                raw_variable="concessoes_credito_total",
                model_variable="concessoes_reais",
                candidate="log_nominal_level",
                series=log_nominal,
                current_model=False,
                note="Log do nível nominal.",
            )
        if "concessoes_reais" in model.columns:
            add_candidate(
                raw_variable="concessoes_credito_total",
                model_variable="concessoes_reais",
                candidate="real_level",
                series=model["concessoes_reais"],
                current_model=False,
                note="Nível deflacionado por IPCA.",
            )
        if "dlog_concessoes_reais" in model.columns:
            add_candidate(
                raw_variable="concessoes_credito_total",
                model_variable="dlog_concessoes_reais",
                candidate="real_log_difference",
                series=model["dlog_concessoes_reais"],
                current_model=True,
                note="Crescimento real mensal usado no VAR base.",
            )
        try:
            add_candidate(
                raw_variable="concessoes_credito_total",
                model_variable="dlog_concessoes_nominais",
                candidate="nominal_log_difference",
                series=log_diff(nominal),
                current_model=False,
                note="Crescimento nominal mensal.",
            )
        except ValueError:
            pass

    return pd.DataFrame(candidates)


def select_recommended_models(
    metrics: pd.DataFrame,
    dm_tests: pd.DataFrame | None = None,
    *,
    min_relative_gain: float = 0.05,
) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    dm_tests = dm_tests if dm_tests is not None else pd.DataFrame()
    benchmark_names = {"random_walk", "ar1", "moving_average_12m", "seasonal_naive"}
    for (variable, horizon), group in metrics.groupby(["variable", "horizon"], dropna=False):
        ranked = group.sort_values(["rmse", "mae"], ascending=True)
        raw_best = ranked.iloc[0]
        base = group.loc[group["model"] == "VAR"]
        base_rmse = float(base["rmse"].iloc[0]) if not base.empty else np.nan
        best_benchmark = group.loc[group["model"].isin(benchmark_names)].sort_values("rmse").head(1)
        best_benchmark_model = str(best_benchmark["model"].iloc[0]) if not best_benchmark.empty else None
        best_benchmark_rmse = float(best_benchmark["rmse"].iloc[0]) if not best_benchmark.empty else np.nan

        dm_support = ""
        has_dm_support = False
        if not dm_tests.empty and str(raw_best["model"]) in benchmark_names:
            match = dm_tests[
                (dm_tests["variable"] == variable)
                & (dm_tests["horizon"] == horizon)
                & (dm_tests["benchmark"] == raw_best["model"])
            ]
            if not match.empty:
                test = match.iloc[0]
                if pd.notna(test.get("pvalue")) and float(test["pvalue"]) < 0.10 and float(test["dm_stat"]) > 0:
                    has_dm_support = True
                    dm_support = " Diferença contra VAR tem suporte no teste Diebold-Mariano."

        raw_gain_vs_base = (
            (base_rmse - float(raw_best["rmse"])) / base_rmse
            if np.isfinite(base_rmse) and base_rmse > 0
            else np.nan
        )
        preserve_base = (
            not base.empty
            and str(raw_best["model"]) != "VAR"
            and np.isfinite(raw_gain_vs_base)
            and raw_gain_vs_base < min_relative_gain
            and not has_dm_support
        )
        best = base.iloc[0] if preserve_base else raw_best

        if str(best["model"]) == "VAR":
            recommendation_type = "base_var"
            reason = (
                f"VAR base preservado; melhor alternativa ganha menos de {min_relative_gain:.0%} em RMSE."
                if preserve_base
                else "VAR base tem o menor RMSE no backtest."
            )
        elif str(best["model"]).startswith("VAR_"):
            recommendation_type = "alternative_var"
            reason = "Especificação VAR alternativa vence por RMSE fora da amostra."
        else:
            recommendation_type = "benchmark"
            reason = f"Benchmark {best['model']} vence por RMSE fora da amostra.{dm_support}"

        rows.append(
            {
                "variable": variable,
                "horizon": int(horizon),
                "recommended_model": str(best["model"]),
                "recommendation_type": recommendation_type,
                "rmse": float(best["rmse"]),
                "mae": float(best["mae"]),
                "base_var_rmse": base_rmse,
                "best_benchmark_model": best_benchmark_model,
                "best_benchmark_rmse": best_benchmark_rmse,
                "rmse_gain_vs_base_var": raw_gain_vs_base if not preserve_base else 0.0,
                "reason": reason,
            }
        )
    return pd.DataFrame(rows).sort_values(["variable", "horizon"]).reset_index(drop=True)
