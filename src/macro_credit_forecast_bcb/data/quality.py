from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


class DataQualityError(RuntimeError):
    """Raised when input data fail hard plausibility checks."""


@dataclass(frozen=True)
class PlausibilityRule:
    variable: str
    min_value: float | None = None
    max_value: float | None = None
    positive: bool = False
    max_abs_pct_change: float | None = None


RAW_PLAUSIBILITY_RULES: dict[str, PlausibilityRule] = {
    "ipca": PlausibilityRule("ipca", min_value=-5.0, max_value=10.0),
    "selic_meta": PlausibilityRule("selic_meta", min_value=0.0, max_value=30.0),
    "spread_credito_total": PlausibilityRule("spread_credito_total", min_value=0.0, max_value=100.0),
    "inadimplencia_total": PlausibilityRule("inadimplencia_total", min_value=0.0, max_value=20.0),
    "concessoes_credito_total": PlausibilityRule(
        "concessoes_credito_total",
        positive=True,
        max_abs_pct_change=0.75,
    ),
}


def _scale_flags(series: pd.Series, rule: PlausibilityRule) -> list[str]:
    values = series.dropna().astype(float)
    flags: list[str] = []
    if values.empty:
        return ["empty_series"]
    if rule.min_value is not None and float(values.min()) < rule.min_value:
        flags.append(f"min_below_{rule.min_value:g}")
    if rule.max_value is not None and float(values.max()) > rule.max_value:
        flags.append(f"max_above_{rule.max_value:g}")
    if rule.positive and (values <= 0).any():
        flags.append("non_positive_values")
    if rule.max_abs_pct_change is not None and len(values) > 1:
        max_jump = values.pct_change().replace([np.inf, -np.inf], np.nan).abs().max()
        if pd.notna(max_jump) and float(max_jump) > rule.max_abs_pct_change:
            flags.append(f"large_pct_change_{float(max_jump):.2f}")
    return flags


def build_data_quality_report(
    frame: pd.DataFrame,
    rules: dict[str, PlausibilityRule] | None = None,
) -> pd.DataFrame:
    rules = rules or RAW_PLAUSIBILITY_RULES
    duplicate_dates = int(frame.index.duplicated().sum()) if isinstance(frame.index, pd.Index) else 0
    rows: list[dict[str, object]] = []

    for variable in frame.columns:
        series = frame[variable]
        values = series.dropna().astype(float)
        rule = rules.get(variable, PlausibilityRule(variable))
        flags = _scale_flags(series, rule)
        hard_fail = any(
            flag.startswith(("min_below", "max_above", "non_positive", "empty_series"))
            for flag in flags
        )
        warning = any(flag.startswith("large_pct_change") for flag in flags)
        status = "fail" if hard_fail else "warning" if warning or duplicate_dates else "ok"

        rows.append(
            {
                "variable": variable,
                "first_date": values.index.min() if not values.empty else pd.NaT,
                "last_date": values.index.max() if not values.empty else pd.NaT,
                "nobs": int(values.shape[0]),
                "duplicate_dates": duplicate_dates,
                "missing_values": int(series.isna().sum()),
                "min": float(values.min()) if not values.empty else np.nan,
                "max": float(values.max()) if not values.empty else np.nan,
                "latest_value": float(values.iloc[-1]) if not values.empty else np.nan,
                "scale_flags": ";".join(flags) if flags else "",
                "status": status,
            }
        )

    return pd.DataFrame(rows)


def assert_data_quality(report: pd.DataFrame, *, allow_quality_warnings: bool = False) -> None:
    if report.empty:
        raise DataQualityError("Data quality report is empty")
    failing = report.loc[report["status"] == "fail"]
    if not failing.empty and not allow_quality_warnings:
        details = failing[["variable", "min", "max", "latest_value", "scale_flags"]].to_dict(
            orient="records"
        )
        raise DataQualityError(f"Hard data quality checks failed: {details}")


def summarize_quality_report(report: pd.DataFrame) -> dict[str, object]:
    if report.empty:
        return {
            "status": "fail",
            "checked_at": pd.Timestamp.utcnow().isoformat(),
            "variables_checked": 0,
            "failures": 0,
            "warnings": 0,
        }
    failures = int((report["status"] == "fail").sum())
    warnings = int((report["status"] == "warning").sum())
    status = "fail" if failures else "warning" if warnings else "ok"
    return {
        "status": status,
        "checked_at": pd.Timestamp.utcnow().isoformat(),
        "variables_checked": int(report.shape[0]),
        "failures": failures,
        "warnings": warnings,
    }

