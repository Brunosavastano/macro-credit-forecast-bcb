from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from macro_credit_forecast_bcb.data.sgs_client import get_sgs_series
from macro_credit_forecast_bcb.features.transformations import (
    build_model_dataset,
    validate_time_series_frame,
)
from macro_credit_forecast_bcb.utils.dates import parse_date, to_month_end_index

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class DatasetBundle:
    raw_monthly: pd.DataFrame
    model_dataset: pd.DataFrame
    metadata: pd.DataFrame


def monthly_conversion(
    series: pd.Series,
    frequency: str,
    conversion: str | None = None,
) -> pd.Series:
    series = series.sort_index()
    series = series[~series.index.duplicated(keep="last")]

    if frequency.upper() == "D":
        if conversion not in {None, "end_of_month"}:
            raise ValueError(f"Unsupported daily monthly conversion: {conversion}")
        monthly = series.resample("ME").last()
    else:
        monthly = series.copy()
        monthly.index = to_month_end_index(monthly.index)
        monthly = monthly.groupby(monthly.index).last()

    monthly = monthly.dropna()
    monthly.index.name = "date"
    return monthly


def download_sgs_config(
    series_config: dict,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    start_ts = parse_date(start)
    end_ts = parse_date(end)
    pieces: list[pd.Series] = []
    metadata_rows: list[dict[str, object]] = []

    for key, spec in series_config["series"].items():
        code = int(spec["code"])
        frequency = str(spec.get("frequency", "M"))
        series = get_sgs_series(
            code,
            start_ts,
            end_ts,
            name=key,
            frequency=frequency,
        )
        monthly = monthly_conversion(
            series,
            frequency=frequency,
            conversion=spec.get("monthly_conversion"),
        )
        monthly.name = key
        pieces.append(monthly)
        metadata_rows.append(
            {
                "variable": key,
                "code": code,
                "name": spec.get("name", key),
                "frequency": frequency,
                "unit": spec.get("unit", ""),
                "transform": spec.get("transform", ""),
                "first_date": monthly.index.min(),
                "last_date": monthly.index.max(),
                "nobs": int(monthly.notna().sum()),
            }
        )

    raw_monthly = pd.concat(pieces, axis=1).sort_index()
    raw_monthly.index.name = "date"
    metadata = pd.DataFrame(metadata_rows)
    return validate_time_series_frame(raw_monthly), metadata


def align_common_sample(raw_monthly: pd.DataFrame, start: str | pd.Timestamp) -> pd.DataFrame:
    sample = raw_monthly.loc[raw_monthly.index >= parse_date(start).to_period("M").to_timestamp("M")]
    first_common = sample.dropna(how="any").index.min()
    last_common = sample.dropna(how="any").index.max()
    if pd.isna(first_common) or pd.isna(last_common):
        raise ValueError("No complete common monthly sample after applying the start date")
    return sample.loc[first_common:last_common].dropna(how="any")


def build_monthly_dataset(
    series_config: dict,
    *,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp | None = None,
) -> DatasetBundle:
    raw_monthly, metadata = download_sgs_config(series_config, start=start, end=end)
    common = align_common_sample(raw_monthly, start)
    model_dataset = build_model_dataset(common)
    return DatasetBundle(raw_monthly=common, model_dataset=model_dataset, metadata=metadata)


def save_dataset_bundle(bundle: DatasetBundle, processed_dir: Path) -> None:
    processed_dir.mkdir(parents=True, exist_ok=True)
    bundle.raw_monthly.to_parquet(processed_dir / "monthly_macro_credit_raw.parquet")
    bundle.model_dataset.to_parquet(processed_dir / "monthly_macro_credit.parquet")
    bundle.metadata.to_csv(processed_dir / "series_metadata.csv", index=False)
    LOGGER.info("Saved monthly datasets to %s", processed_dir)

