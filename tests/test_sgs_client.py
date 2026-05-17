from __future__ import annotations

import pandas as pd

from macro_credit_forecast_bcb.data.sgs_client import _parse_sgs_payload


def test_parse_sgs_payload_handles_brazilian_decimal_and_duplicates() -> None:
    payload = [
        {"data": "30/11/2023", "valor": "0.88"},
        {"data": "31/12/2023", "valor": "14.75"},
        {"data": "31/01/2024", "valor": "1,25"},
        {"data": "31/01/2024", "valor": "1,30"},
        {"data": "29/02/2024", "valor": "2.345,67"},
    ]

    series = _parse_sgs_payload(payload, name="test")

    assert series.name == "test"
    assert isinstance(series.index, pd.DatetimeIndex)
    assert series.loc[pd.Timestamp("2023-11-30")] == 0.88
    assert series.loc[pd.Timestamp("2023-12-31")] == 14.75
    assert series.loc[pd.Timestamp("2024-01-31")] == 1.30
    assert series.loc[pd.Timestamp("2024-02-29")] == 2345.67
    assert len(series) == 4
