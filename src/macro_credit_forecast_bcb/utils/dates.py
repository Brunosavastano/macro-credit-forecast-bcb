from __future__ import annotations

from datetime import date, datetime

import pandas as pd


def parse_date(value: str | date | datetime | pd.Timestamp | None) -> pd.Timestamp:
    if value is None:
        return pd.Timestamp.today().normalize()
    return pd.Timestamp(value)


def format_bcb_date(value: str | date | datetime | pd.Timestamp) -> str:
    ts = parse_date(value)
    return ts.strftime("%d/%m/%Y")


def to_month_end_index(index: pd.Index) -> pd.DatetimeIndex:
    dates = pd.to_datetime(index)
    return dates.to_period("M").to_timestamp("M")


def next_month_ends(last_date: pd.Timestamp, periods: int) -> pd.DatetimeIndex:
    start = pd.Timestamp(last_date).to_period("M").to_timestamp("M") + pd.offsets.MonthEnd(1)
    return pd.date_range(start=start, periods=periods, freq="ME")

