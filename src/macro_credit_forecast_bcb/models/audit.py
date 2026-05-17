from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.tsa.vector_ar.var_model import VARResults


def status_rank(status: str) -> int:
    return {"pass": 0, "warning": 1, "fail": 2}.get(status, 1)


def worst_status(statuses: list[str]) -> str:
    if not statuses:
        return "warning"
    return max(statuses, key=status_rank)


def run_transformation_audit(raw: pd.DataFrame, model: pd.DataFrame) -> pd.DataFrame:
    """Summarize whether raw and modeled variables look internally consistent."""
    mappings = {
        "ipca": ("ipca", "level_rate"),
        "selic_meta": ("selic", "level_rate"),
        "spread_credito_total": ("spread", "level_rate"),
        "inadimplencia_total": ("inadimplencia", "level_rate"),
        "concessoes_credito_total": ("dlog_concessoes_reais", "deflate_log_diff"),
    }
    rows: list[dict[str, object]] = []
    for raw_name, (model_name, transformation) in mappings.items():
        raw_series = raw[raw_name].dropna() if raw_name in raw else pd.Series(dtype=float)
        model_series = model[model_name].dropna() if model_name in model else pd.Series(dtype=float)
        status = "pass"
        notes: list[str] = []
        if raw_series.empty or model_series.empty:
            status = "fail"
            notes.append("Série bruta ou transformada ausente.")
        elif transformation == "deflate_log_diff":
            if "concessoes_reais" not in model.columns:
                status = "fail"
                notes.append("Nível real de concessões não foi reconstruído.")
            elif model_series.abs().max() > 50:
                status = "warning"
                notes.append("Crescimento real mensal tem outliers acima de 50%.")
        elif raw_series.index.intersection(model_series.index).empty:
            status = "fail"
            notes.append("Série bruta e série modelável não têm datas em comum.")

        rows.append(
            {
                "raw_variable": raw_name,
                "model_variable": model_name,
                "transformation": transformation,
                "raw_nobs": int(raw_series.shape[0]),
                "model_nobs": int(model_series.shape[0]),
                "raw_min": float(raw_series.min()) if not raw_series.empty else np.nan,
                "raw_max": float(raw_series.max()) if not raw_series.empty else np.nan,
                "model_min": float(model_series.min()) if not model_series.empty else np.nan,
                "model_max": float(model_series.max()) if not model_series.empty else np.nan,
                "status": status,
                "notes": " ".join(notes),
            }
        )
    return pd.DataFrame(rows)


def run_residual_diagnostics(model_result: VARResults) -> pd.DataFrame:
    """Run equation-level residual checks for a fitted VAR."""
    residuals = pd.DataFrame(model_result.resid, columns=model_result.names)
    rows: list[dict[str, object]] = []
    ljung_lag = max(model_result.k_ar + 1, min(12, max(2, model_result.nobs // 4)))
    for variable in residuals.columns:
        series = residuals[variable].dropna().astype(float)
        status = "pass"
        notes: list[str] = []
        lb_pvalue = np.nan
        arch_pvalue = np.nan
        outlier_share = np.nan
        try:
            lb = acorr_ljungbox(series, lags=[ljung_lag], return_df=True)
            lb_pvalue = float(lb["lb_pvalue"].iloc[-1])
            if lb_pvalue < 0.01:
                status = "fail"
                notes.append("Autocorrelação residual forte.")
            elif lb_pvalue < 0.05:
                status = "warning"
                notes.append("Autocorrelação residual marginal.")
        except Exception as exc:
            status = worst_status([status, "warning"])
            notes.append(f"Ljung-Box indisponível: {exc}")
        try:
            arch_lags = min(12, max(2, len(series) // 5))
            arch_result = het_arch(series, nlags=arch_lags)
            arch_pvalue = float(arch_result[1])
            if arch_pvalue < 0.05:
                status = worst_status([status, "warning"])
                notes.append("Evidência de heterocedasticidade ARCH.")
        except Exception as exc:
            status = worst_status([status, "warning"])
            notes.append(f"ARCH indisponível: {exc}")

        std = float(series.std(ddof=1)) if len(series) > 1 else np.nan
        if pd.notna(std) and std > 0:
            outlier_share = float((series.abs() > 3 * std).mean())
            if outlier_share > 0.05:
                status = worst_status([status, "warning"])
                notes.append("Mais de 5% dos resíduos excedem 3 desvios-padrão.")

        rows.append(
            {
                "variable": variable,
                "ljung_box_lag": ljung_lag,
                "ljung_box_pvalue": lb_pvalue,
                "arch_pvalue": arch_pvalue,
                "outlier_share": outlier_share,
                "status": status,
                "notes": " ".join(notes),
            }
        )
    return pd.DataFrame(rows)


def _hac_variance(values: np.ndarray, max_lag: int) -> float:
    values = values - np.nanmean(values)
    values = values[np.isfinite(values)]
    nobs = len(values)
    if nobs <= 1:
        return np.nan
    gamma0 = float(np.dot(values, values) / nobs)
    variance = gamma0
    for lag in range(1, min(max_lag, nobs - 1) + 1):
        weight = 1.0 - lag / (max_lag + 1.0)
        cov = float(np.dot(values[lag:], values[:-lag]) / nobs)
        variance += 2.0 * weight * cov
    return variance / nobs


def diebold_mariano_test(
    var_errors: pd.Series,
    benchmark_errors: pd.Series,
    *,
    horizon: int,
    loss: str = "squared",
) -> dict[str, float | str | int]:
    aligned = pd.concat([var_errors, benchmark_errors], axis=1, keys=["var", "benchmark"]).dropna()
    if aligned.shape[0] < 10:
        return {
            "dm_stat": np.nan,
            "pvalue": np.nan,
            "nobs": int(aligned.shape[0]),
            "loss": loss,
            "status": "insufficient_data",
        }
    if loss == "absolute":
        diff = aligned["var"].abs() - aligned["benchmark"].abs()
    else:
        diff = aligned["var"] ** 2 - aligned["benchmark"] ** 2
    values = diff.to_numpy(dtype=float)
    variance = _hac_variance(values, max_lag=max(0, int(horizon) - 1))
    if not np.isfinite(variance) or variance <= 0:
        return {
            "dm_stat": np.nan,
            "pvalue": np.nan,
            "nobs": int(aligned.shape[0]),
            "loss": loss,
            "status": "zero_variance",
        }
    dm_stat = float(np.nanmean(values) / math.sqrt(variance))
    pvalue = float(2.0 * stats.norm.sf(abs(dm_stat)))
    return {
        "dm_stat": dm_stat,
        "pvalue": pvalue,
        "nobs": int(aligned.shape[0]),
        "loss": loss,
        "status": "ok",
    }


def run_forecast_accuracy_tests(backtest_records: pd.DataFrame) -> pd.DataFrame:
    if backtest_records.empty or "VAR" not in set(backtest_records.get("model", [])):
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    benchmarks = sorted(set(backtest_records["model"]) - {"VAR"})
    for (variable, horizon), group in backtest_records.groupby(["variable", "horizon"], dropna=False):
        var_group = group.loc[group["model"] == "VAR"].copy()
        var_group["error"] = var_group["forecast"] - var_group["actual"]
        var_errors = var_group.set_index(["origin", "target_date"])["error"]
        for benchmark in benchmarks:
            bench_group = group.loc[group["model"] == benchmark].copy()
            bench_group["error"] = bench_group["forecast"] - bench_group["actual"]
            bench_errors = bench_group.set_index(["origin", "target_date"])["error"]
            test = diebold_mariano_test(var_errors, bench_errors, horizon=int(horizon))
            rows.append(
                {
                    "variable": variable,
                    "horizon": int(horizon),
                    "benchmark": benchmark,
                    **test,
                    "interpretation": _dm_interpretation(test),
                }
            )
    return pd.DataFrame(rows)


def _dm_interpretation(test: dict[str, Any]) -> str:
    if test.get("status") != "ok" or pd.isna(test.get("pvalue")):
        return "Teste inconclusivo."
    dm_stat = float(test["dm_stat"])
    pvalue = float(test["pvalue"])
    if pvalue >= 0.10:
        return "Sem diferença estatisticamente clara."
    if dm_stat < 0:
        return "VAR tem perda menor que o benchmark."
    return "VAR tem perda maior que o benchmark."


def interval_coverage(backtest_records: pd.DataFrame) -> pd.DataFrame:
    if backtest_records.empty:
        return pd.DataFrame()
    var_records = backtest_records.loc[backtest_records["model"] == "VAR"].copy()
    rows: list[dict[str, object]] = []
    for (variable, horizon), group in var_records.groupby(["variable", "horizon"], dropna=False):
        row: dict[str, object] = {"variable": variable, "horizon": int(horizon), "nobs": int(group.shape[0])}
        for level in ["68", "95"]:
            lower = f"lower_{level}"
            upper = f"upper_{level}"
            valid = group[[lower, upper, "actual"]].dropna()
            if valid.empty:
                row[f"coverage_{level}"] = np.nan
                row[f"avg_width_{level}"] = np.nan
                continue
            row[f"coverage_{level}"] = float(
                ((valid["actual"] >= valid[lower]) & (valid["actual"] <= valid[upper])).mean()
            )
            row[f"avg_width_{level}"] = float((valid[upper] - valid[lower]).mean())
        rows.append(row)
    return pd.DataFrame(rows)


def enhanced_accuracy_metrics(backtest_records: pd.DataFrame) -> pd.DataFrame:
    if backtest_records.empty:
        return pd.DataFrame()
    frame = backtest_records.copy()
    frame["error"] = frame["forecast"] - frame["actual"]
    frame["abs_error"] = frame["error"].abs()
    frame["sq_error"] = frame["error"] ** 2
    denominator = (frame["forecast"].abs() + frame["actual"].abs()) / 2
    frame["smape_component"] = np.where(denominator > 0, frame["abs_error"] / denominator, np.nan)

    scales: dict[str, float] = {}
    for variable, group in frame.loc[frame["model"] == "VAR"].groupby("variable"):
        actual_path = group.sort_values(["target_date", "origin"]).drop_duplicates("target_date")
        scale = actual_path["actual"].diff().abs().mean()
        scales[str(variable)] = float(scale) if pd.notna(scale) and scale > 0 else np.nan

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
    metrics["mase"] = metrics.apply(
        lambda row: float(row["mae"] / scales.get(str(row["variable"]), np.nan))
        if pd.notna(scales.get(str(row["variable"]), np.nan))
        else np.nan,
        axis=1,
    )
    random_walk = metrics.loc[metrics["model"] == "random_walk", ["variable", "horizon", "rmse"]].rename(
        columns={"rmse": "random_walk_rmse"}
    )
    metrics = metrics.merge(random_walk, on=["variable", "horizon"], how="left")
    metrics["theil_u"] = metrics["rmse"] / metrics["random_walk_rmse"]
    metrics["rmse_rank"] = metrics.groupby(["variable", "horizon"])["rmse"].rank(method="min")
    metrics["is_best_rmse"] = metrics["rmse_rank"] == 1
    return metrics.drop(columns=["random_walk_rmse"])


def build_model_scorecard(
    *,
    data_quality: pd.DataFrame,
    stationarity: pd.DataFrame,
    model_summary: dict[str, Any],
    residual_diagnostics: pd.DataFrame,
    audit_metrics: pd.DataFrame,
    coverage: pd.DataFrame,
    dm_tests: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    data_status = "warning" if data_quality.empty else worst_status(
        ["fail" if s == "fail" else "warning" if s == "warning" else "pass" for s in data_quality["status"]]
    )
    rows.append(
        {
            "dimension": "Dados",
            "status": data_status,
            "evidence": "Relatório de qualidade ausente." if data_quality.empty else "Checks de escala e completude executados.",
            "recommendation": "Reexecutar refresh e revisar data_quality_report.csv." if data_status != "pass" else "Manter checks no refresh.",
        }
    )

    ambiguous = int((stationarity.get("chosen_order", pd.Series(dtype=str)) == "ambiguous").sum())
    rows.append(
        {
            "dimension": "Estacionariedade",
            "status": "warning" if ambiguous else "pass",
            "evidence": f"{ambiguous} série(s) com ordem ambígua." if ambiguous else "ADF/KPSS sem ambiguidades materiais.",
            "recommendation": "Testar transformações alternativas para séries ambíguas." if ambiguous else "Manter documentação das transformações.",
        }
    )

    diagnostics = model_summary.get("diagnostics", {})
    stable = bool(diagnostics.get("is_stable", False))
    selected_lag = int(diagnostics.get("lag_order", 0) or 0)
    rows.append(
        {
            "dimension": "Especificação",
            "status": "pass" if stable and selected_lag > 0 else "fail",
            "evidence": f"Modelo {'estável' if stable else 'instável'}; lag selecionado {selected_lag}.",
            "recommendation": "Testar maxlags=12 e VAR menor como robustez." if stable else "Não usar forecasts até corrigir instabilidade.",
        }
    )

    residual_status = "pass"
    whiteness = diagnostics.get("whiteness", {})
    normality = diagnostics.get("normality", {})
    if whiteness.get("conclusion") == "reject" and normality.get("conclusion") == "reject":
        residual_status = "fail"
    elif whiteness.get("conclusion") == "reject" or normality.get("conclusion") == "reject":
        residual_status = "warning"
    if not residual_diagnostics.empty:
        residual_status = worst_status([residual_status, *residual_diagnostics["status"].tolist()])
    rows.append(
        {
            "dimension": "Resíduos",
            "status": residual_status,
            "evidence": f"Whiteness={whiteness.get('conclusion')}; normalidade={normality.get('conclusion')}.",
            "recommendation": "Revisar defasagens, sazonalidade, outliers e possível VAR menor.",
        }
    )

    var_metrics = audit_metrics.loc[audit_metrics["model"] == "VAR"] if not audit_metrics.empty else pd.DataFrame()
    best_share = float(var_metrics["is_best_rmse"].mean()) if not var_metrics.empty else np.nan
    forecast_status = "fail" if var_metrics.empty else "pass" if best_share >= 0.5 else "warning"
    rows.append(
        {
            "dimension": "Forecast",
            "status": forecast_status,
            "evidence": "Backtest ausente." if var_metrics.empty else f"VAR é melhor por RMSE em {best_share:.0%} dos pares variável-horizonte.",
            "recommendation": "Usar benchmark vencedor por variável/horizonte quando VAR perde de forma persistente.",
        }
    )

    if coverage.empty:
        interval_status = "warning"
        interval_evidence = "Coverage indisponível."
    else:
        poor_95 = coverage["coverage_95"].dropna() < 0.75
        interval_status = "fail" if poor_95.any() else "pass"
        interval_evidence = f"Cobertura 95% média: {coverage['coverage_95'].mean():.0%}."
    rows.append(
        {
            "dimension": "Intervalos",
            "status": interval_status,
            "evidence": interval_evidence,
            "recommendation": "Considerar bootstrap residual se coverage ficar abaixo do esperado.",
        }
    )

    significant_losses = 0
    if not dm_tests.empty and "interpretation" in dm_tests:
        significant_losses = int(
            ((dm_tests["pvalue"] < 0.10) & (dm_tests["dm_stat"] > 0)).sum()
        )
    rows.append(
        {
            "dimension": "Robustez",
            "status": "warning" if significant_losses else "pass",
            "evidence": f"{significant_losses} teste(s) DM sugerem perda maior do VAR contra benchmark.",
            "recommendation": "Priorizar benchmarks onde DM indicar perda menor e significativa.",
        }
    )

    return pd.DataFrame(rows)


def run_econometric_audit(
    history: pd.DataFrame,
    forecast: pd.DataFrame,
    backtest_records: pd.DataFrame,
    backtest_metrics: pd.DataFrame,
    *,
    raw: pd.DataFrame | None = None,
    data_quality: pd.DataFrame | None = None,
    stationarity: pd.DataFrame | None = None,
    model_summary: dict[str, Any] | None = None,
    residual_diagnostics: pd.DataFrame | None = None,
) -> dict[str, Any]:
    del forecast, backtest_metrics
    audit_metrics = enhanced_accuracy_metrics(backtest_records)
    coverage = interval_coverage(backtest_records)
    dm_tests = run_forecast_accuracy_tests(backtest_records)
    transformation_audit = run_transformation_audit(raw, history) if raw is not None else pd.DataFrame()
    scorecard = build_model_scorecard(
        data_quality=data_quality if data_quality is not None else pd.DataFrame(),
        stationarity=stationarity if stationarity is not None else pd.DataFrame(),
        model_summary=model_summary or {},
        residual_diagnostics=residual_diagnostics if residual_diagnostics is not None else pd.DataFrame(),
        audit_metrics=audit_metrics,
        coverage=coverage,
        dm_tests=dm_tests,
    )
    overall_status = worst_status(scorecard["status"].tolist()) if not scorecard.empty else "warning"
    return {
        "overall_status": overall_status,
        "scorecard": scorecard,
        "audit_metrics": audit_metrics,
        "interval_coverage": coverage,
        "model_comparison_tests": dm_tests,
        "transformation_audit": transformation_audit,
        "summary": {
            "overall_status": overall_status,
            "dimensions": scorecard.to_dict(orient="records"),
            "generated_at": pd.Timestamp.utcnow().isoformat(),
        },
    }
