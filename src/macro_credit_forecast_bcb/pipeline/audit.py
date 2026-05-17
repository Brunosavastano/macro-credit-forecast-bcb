from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from macro_credit_forecast_bcb.features.transformations import model_columns
from macro_credit_forecast_bcb.models.audit import run_econometric_audit, run_residual_diagnostics
from macro_credit_forecast_bcb.models.robustness import (
    build_transformation_candidates,
    default_var_specifications,
    evaluate_var_specifications,
    rolling_specification_backtest,
)
from macro_credit_forecast_bcb.models.var_model import fit_var
from macro_credit_forecast_bcb.utils.config import load_model_config
from macro_credit_forecast_bcb.utils.logging import configure_logging
from macro_credit_forecast_bcb.utils.paths import FORECAST_DATA_DIR, OUTPUTS_DIR, PROCESSED_DATA_DIR, ensure_project_dirs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run econometric accuracy audit.")
    parser.add_argument("--data", default=None, help="Path to processed model dataset parquet.")
    parser.add_argument("--raw-data", default=None, help="Path to raw monthly dataset parquet.")
    parser.add_argument("--forecast", default=None, help="Path to forecast parquet.")
    parser.add_argument("--records", default=None, help="Path to backtest records parquet.")
    parser.add_argument("--metrics", default=None, help="Path to backtest metrics parquet.")
    parser.add_argument(
        "--skip-robustness-backtest",
        action="store_true",
        help="Skip rolling backtest for alternative VAR specifications.",
    )
    return parser.parse_args()


def _json_safe(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if pd.isna(value) if not isinstance(value, (list, dict, str, bytes)) else False:
        return None
    return value


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def main() -> None:
    configure_logging()
    ensure_project_dirs()
    args = parse_args()
    config = load_model_config()

    data_path = Path(args.data) if args.data else PROCESSED_DATA_DIR / "monthly_macro_credit.parquet"
    raw_path = Path(args.raw_data) if args.raw_data else PROCESSED_DATA_DIR / "monthly_macro_credit_raw.parquet"
    forecast_path = Path(args.forecast) if args.forecast else FORECAST_DATA_DIR / "forecast_12m.parquet"
    records_path = Path(args.records) if args.records else OUTPUTS_DIR / "backtest_records.parquet"
    metrics_path = Path(args.metrics) if args.metrics else OUTPUTS_DIR / "backtest_metrics.parquet"

    required = [data_path, raw_path, forecast_path, records_path, metrics_path]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing audit inputs. Run refresh, forecast and backtest first: {missing}")

    history = pd.read_parquet(data_path)
    history.index = pd.to_datetime(history.index)
    inferred_freq = pd.infer_freq(history.index)
    if inferred_freq:
        history = history.asfreq(inferred_freq)

    raw = pd.read_parquet(raw_path)
    raw.index = pd.to_datetime(raw.index)
    forecast = pd.read_parquet(forecast_path)
    backtest_records = pd.read_parquet(records_path)
    backtest_metrics = pd.read_parquet(metrics_path)
    data_quality = pd.read_csv(OUTPUTS_DIR / "data_quality_report.csv") if (OUTPUTS_DIR / "data_quality_report.csv").exists() else pd.DataFrame()
    stationarity = pd.read_csv(OUTPUTS_DIR / "stationarity_report.csv") if (OUTPUTS_DIR / "stationarity_report.csv").exists() else pd.DataFrame()
    model_summary = _read_json(OUTPUTS_DIR / "model_summary.json")

    lag_order = int(model_summary.get("diagnostics", {}).get("lag_order") or 1)
    result = fit_var(history[model_columns()], lag_order)
    residual_diagnostics = run_residual_diagnostics(result)
    variables = model_columns()
    model_config = config.get("model", {})
    backtest_config = config.get("backtest", {})
    specifications = default_var_specifications(
        variables,
        base_maxlags=int(model_config.get("maxlags", 6)),
        robustness_maxlags=int(model_config.get("robustness_maxlags", 12)),
        criterion=str(model_config.get("criterion", "bic")),
    )
    specification_diagnostics = evaluate_var_specifications(history[variables], specifications)
    transformation_candidates = build_transformation_candidates(raw, history)
    if args.skip_robustness_backtest:
        robustness_records = pd.DataFrame()
        robustness_failures = pd.DataFrame()
    else:
        robustness_records, robustness_failures = rolling_specification_backtest(
            history[variables],
            specifications,
            horizons=backtest_config.get("horizons", [1, 3, 6, 12]),
            initial_window=int(backtest_config.get("initial_window", 72)),
            expanding=bool(backtest_config.get("expanding", True)),
        )

    audit = run_econometric_audit(
        history,
        forecast,
        backtest_records,
        backtest_metrics,
        raw=raw,
        data_quality=data_quality,
        stationarity=stationarity,
        model_summary=model_summary,
        residual_diagnostics=residual_diagnostics,
        robustness_records=robustness_records,
        specification_diagnostics=specification_diagnostics,
        transformation_candidates=transformation_candidates,
    )

    audit["audit_metrics"].to_parquet(OUTPUTS_DIR / "econometric_audit_metrics.parquet", index=False)
    audit["interval_coverage"].to_parquet(OUTPUTS_DIR / "interval_coverage.parquet", index=False)
    audit["model_comparison_tests"].to_parquet(OUTPUTS_DIR / "model_comparison_tests.parquet", index=False)
    audit["model_recommendations"].to_parquet(OUTPUTS_DIR / "model_recommendations.parquet", index=False)
    audit["scorecard"].to_parquet(OUTPUTS_DIR / "econometric_scorecard.parquet", index=False)
    audit["transformation_audit"].to_parquet(OUTPUTS_DIR / "transformation_audit.parquet", index=False)
    audit["transformation_candidates"].to_parquet(OUTPUTS_DIR / "transformation_candidates.parquet", index=False)
    audit["specification_diagnostics"].to_parquet(OUTPUTS_DIR / "specification_diagnostics.parquet", index=False)
    robustness_records.to_parquet(OUTPUTS_DIR / "robustness_backtest_records.parquet", index=False)
    robustness_failures.to_parquet(OUTPUTS_DIR / "robustness_backtest_failures.parquet", index=False)
    residual_diagnostics.to_parquet(OUTPUTS_DIR / "residual_equation_diagnostics.parquet", index=False)
    with (OUTPUTS_DIR / "econometric_audit.json").open("w", encoding="utf-8") as file:
        json.dump(_json_safe(audit["summary"]), file, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
