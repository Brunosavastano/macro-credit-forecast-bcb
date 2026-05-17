from __future__ import annotations

import argparse
import logging

from macro_credit_forecast_bcb.data.build_dataset import build_monthly_dataset, save_dataset_bundle
from macro_credit_forecast_bcb.data.focus_client import download_focus_snapshot
from macro_credit_forecast_bcb.data.quality import assert_data_quality
from macro_credit_forecast_bcb.features.stationarity import run_stationarity_tests
from macro_credit_forecast_bcb.features.transformations import model_columns
from macro_credit_forecast_bcb.utils.config import load_model_config, load_series_config
from macro_credit_forecast_bcb.utils.logging import configure_logging
from macro_credit_forecast_bcb.utils.paths import FORECAST_DATA_DIR, OUTPUTS_DIR, PROCESSED_DATA_DIR, ensure_project_dirs

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download BCB data and build the monthly model dataset.")
    parser.add_argument("--start", default=None, help="Start date, e.g. 2011-03-01.")
    parser.add_argument("--end", default=None, help="End date. Defaults to today.")
    parser.add_argument("--skip-focus", action="store_true", help="Skip Focus/OData snapshot download.")
    parser.add_argument(
        "--allow-quality-warnings",
        action="store_true",
        help="Continue even when hard data quality checks flag implausible scale.",
    )
    return parser.parse_args()


def main() -> None:
    configure_logging()
    ensure_project_dirs()
    args = parse_args()
    series_config = load_series_config()
    model_config = load_model_config()
    start = args.start or model_config.get("data", {}).get("start", "2011-03-01")
    end = args.end or model_config.get("data", {}).get("end")

    bundle = build_monthly_dataset(series_config, start=start, end=end)
    assert_data_quality(bundle.quality_report, allow_quality_warnings=args.allow_quality_warnings)
    save_dataset_bundle(bundle, PROCESSED_DATA_DIR)
    bundle.quality_report.to_csv(OUTPUTS_DIR / "data_quality_report.csv", index=False)
    LOGGER.info("Saved data quality report to %s", OUTPUTS_DIR / "data_quality_report.csv")

    stationarity = run_stationarity_tests(bundle.model_dataset[model_columns()])
    stationarity.to_csv(OUTPUTS_DIR / "stationarity_report.csv", index=False)
    LOGGER.info("Saved stationarity report to %s", OUTPUTS_DIR / "stationarity_report.csv")

    if not args.skip_focus and model_config.get("focus", {}).get("enabled", True):
        top = int(model_config.get("focus", {}).get("top", 5000))
        focus_frames = download_focus_snapshot(top=top)
        for name, frame in focus_frames.items():
            if frame.empty:
                continue
            frame.to_parquet(FORECAST_DATA_DIR / f"focus_{name}.parquet")
        LOGGER.info("Saved Focus snapshots to %s", FORECAST_DATA_DIR)


if __name__ == "__main__":
    main()
