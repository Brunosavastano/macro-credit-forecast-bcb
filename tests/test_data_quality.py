from __future__ import annotations

import pandas as pd
import pytest

from macro_credit_forecast_bcb.data.quality import (
    DataQualityError,
    assert_data_quality,
    build_data_quality_report,
)


def test_data_quality_rejects_implausible_scale() -> None:
    index = pd.date_range("2026-01-31", periods=3, freq="ME")
    frame = pd.DataFrame(
        {
            "ipca": [33.0, 70.0, 88.0],
            "selic_meta": [1500.0, 1500.0, 1475.0],
            "spread_credito_total": [2176.0, 2211.0, 2184.0],
            "concessoes_credito_total": [648496.0, 609344.0, 732939.0],
            "inadimplencia_total": [426.0, 444.0, 433.0],
        },
        index=index,
    )

    report = build_data_quality_report(frame)

    assert set(report.loc[report["status"] == "fail", "variable"]) == {
        "ipca",
        "selic_meta",
        "spread_credito_total",
        "inadimplencia_total",
    }
    with pytest.raises(DataQualityError):
        assert_data_quality(report)


def test_data_quality_accepts_plausible_scale() -> None:
    index = pd.date_range("2026-01-31", periods=3, freq="ME")
    frame = pd.DataFrame(
        {
            "ipca": [0.33, 0.70, 0.88],
            "selic_meta": [15.0, 15.0, 14.75],
            "spread_credito_total": [21.76, 22.11, 21.84],
            "concessoes_credito_total": [648496.0, 609344.0, 732939.0],
            "inadimplencia_total": [4.26, 4.44, 4.33],
        },
        index=index,
    )

    report = build_data_quality_report(frame)

    assert "fail" not in set(report["status"])
    assert_data_quality(report)

