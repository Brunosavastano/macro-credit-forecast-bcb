from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from macro_credit_forecast_bcb.features.transformations import model_columns
from macro_credit_forecast_bcb.models.backtest import rolling_backtest
from macro_credit_forecast_bcb.utils.config import load_model_config
from macro_credit_forecast_bcb.utils.logging import configure_logging
from macro_credit_forecast_bcb.utils.paths import OUTPUTS_DIR, PROCESSED_DATA_DIR, ensure_project_dirs

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run rolling/expanding forecast backtest.")
    parser.add_argument("--data", default=None, help="Path to processed model dataset parquet.")
    parser.add_argument("--initial-window", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    configure_logging()
    ensure_project_dirs()
    args = parse_args()
    config = load_model_config()
    data_path = Path(args.data) if args.data else PROCESSED_DATA_DIR / "monthly_macro_credit.parquet"
    if not data_path.exists():
        raise FileNotFoundError(f"Processed dataset not found: {data_path}. Run make refresh first.")

    frame = pd.read_parquet(data_path)
    frame.index = pd.to_datetime(frame.index)
    inferred_freq = pd.infer_freq(frame.index)
    if inferred_freq:
        frame = frame.asfreq(inferred_freq)
    backtest_config = config.get("backtest", {})
    records, metrics = rolling_backtest(
        frame[model_columns()],
        horizons=backtest_config.get("horizons", [1, 3, 6, 12]),
        initial_window=args.initial_window or int(backtest_config.get("initial_window", 72)),
        expanding=bool(backtest_config.get("expanding", True)),
        maxlags=int(config.get("model", {}).get("maxlags", 6)),
        criterion=str(config.get("model", {}).get("criterion", "bic")),
    )

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    records.to_parquet(OUTPUTS_DIR / "backtest_records.parquet", index=False)
    metrics.to_parquet(OUTPUTS_DIR / "backtest_metrics.parquet", index=False)
    LOGGER.info("Saved backtest records and metrics to %s", OUTPUTS_DIR)


if __name__ == "__main__":
    main()
