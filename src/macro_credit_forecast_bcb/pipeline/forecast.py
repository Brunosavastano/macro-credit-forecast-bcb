from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from macro_credit_forecast_bcb.features.transformations import model_columns
from macro_credit_forecast_bcb.models.forecast import forecast_var
from macro_credit_forecast_bcb.models.model_selection import information_criteria_to_records, select_var_lag
from macro_credit_forecast_bcb.models.var_model import fit_var, var_diagnostics
from macro_credit_forecast_bcb.models.vecm_model import select_vecm_rank
from macro_credit_forecast_bcb.utils.config import load_model_config
from macro_credit_forecast_bcb.utils.logging import configure_logging
from macro_credit_forecast_bcb.utils.paths import FORECAST_DATA_DIR, OUTPUTS_DIR, PROCESSED_DATA_DIR, ensure_project_dirs

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit VAR/VECM candidates and generate 12-month forecasts.")
    parser.add_argument("--data", default=None, help="Path to processed model dataset parquet.")
    parser.add_argument("--horizon", type=int, default=None, help="Forecast horizon in months.")
    return parser.parse_args()


def _json_safe(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, pd.DataFrame):
        return value.to_dict(orient="records")
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    return value


def save_diagnostics(diagnostics: dict[str, object], path: Path) -> None:
    rows = []
    for key, value in diagnostics.items():
        rows.append({"metric": key, "value": json.dumps(_json_safe(value), ensure_ascii=True)})
    pd.DataFrame(rows).to_parquet(path)


def main() -> None:
    configure_logging()
    ensure_project_dirs()
    args = parse_args()
    model_config = load_model_config()
    data_path = Path(args.data) if args.data else PROCESSED_DATA_DIR / "monthly_macro_credit.parquet"
    if not data_path.exists():
        raise FileNotFoundError(f"Processed dataset not found: {data_path}. Run make refresh first.")

    frame = pd.read_parquet(data_path)
    frame.index = pd.to_datetime(frame.index)
    inferred_freq = pd.infer_freq(frame.index)
    if inferred_freq:
        frame = frame.asfreq(inferred_freq)
    variables = model_columns()
    data = frame[variables].dropna()
    horizon = args.horizon or int(model_config.get("model", {}).get("horizon", 12))
    maxlags = int(model_config.get("model", {}).get("maxlags", 6))
    criterion = str(model_config.get("model", {}).get("criterion", "bic"))

    selection = select_var_lag(data, maxlags=maxlags, criterion=criterion)
    result = fit_var(data, int(selection["selected_lag"]))
    forecast = forecast_var(result, data, steps=horizon)
    diagnostics = var_diagnostics(result)

    k_ar_diff = max(int(selection["selected_lag"]) - 1, 1)
    vecm_candidates = [col for col in ["selic", "spread", "concessoes_reais", "inadimplencia"] if col in frame.columns]
    vecm_rank = select_vecm_rank(
        frame[vecm_candidates],
        det_order=int(model_config.get("model", {}).get("vecm_det_order", 0)),
        k_ar_diff=k_ar_diff,
    )

    FORECAST_DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    forecast.to_parquet(FORECAST_DATA_DIR / "forecast_12m.parquet", index=False)
    selection_table = selection["ic_table"]
    if isinstance(selection_table, pd.DataFrame):
        selection_table.to_parquet(OUTPUTS_DIR / "var_lag_selection.parquet", index=False)
    save_diagnostics(diagnostics, OUTPUTS_DIR / "diagnostics.parquet")

    summary = {
        "model_selected": f"VAR({int(selection['selected_lag'])})",
        "selection_criterion": criterion,
        "variables": variables,
        "sample_start": data.index.min(),
        "sample_end": data.index.max(),
        "nobs": int(data.shape[0]),
        "forecast_horizon": horizon,
        "information_criteria": information_criteria_to_records(selection),
        "diagnostics": diagnostics,
        "vecm_candidate": vecm_rank,
        "methodological_note": (
            "VECM is treated as a candidate only when integration and Johansen evidence justify it; "
            "the default model is VAR in stationary transformations."
        ),
    }
    with (OUTPUTS_DIR / "model_summary.json").open("w", encoding="utf-8") as file:
        json.dump(_json_safe(summary), file, indent=2, ensure_ascii=True)
    LOGGER.info("Saved forecast and model summary")


if __name__ == "__main__":
    main()
