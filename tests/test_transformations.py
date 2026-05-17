from __future__ import annotations

import numpy as np
import pandas as pd

from macro_credit_forecast_bcb.features.transformations import (
    build_model_dataset,
    compound_ipca_index,
    rolling_compounded_rate,
)


def test_compound_ipca_index_and_rolling_rate() -> None:
    index = pd.date_range("2024-01-31", periods=12, freq="ME")
    ipca = pd.Series([1.0] * 12, index=index, name="ipca")

    price_index = compound_ipca_index(ipca)
    ipca_12m = rolling_compounded_rate(ipca, 12)

    assert round(price_index.iloc[-1], 6) == round(100 * (1.01**12), 6)
    assert round(ipca_12m.iloc[-1], 6) == round(((1.01**12) - 1) * 100, 6)


def test_build_model_dataset_creates_expected_columns() -> None:
    index = pd.date_range("2020-01-31", periods=24, freq="ME")
    raw = pd.DataFrame(
        {
            "ipca": np.repeat(0.5, len(index)),
            "selic_meta": np.linspace(2.0, 12.0, len(index)),
            "spread_credito_total": np.linspace(15.0, 20.0, len(index)),
            "concessoes_credito_total": np.linspace(1000.0, 1500.0, len(index)),
            "inadimplencia_total": np.linspace(2.0, 4.0, len(index)),
        },
        index=index,
    )

    model = build_model_dataset(raw)

    assert {"ipca", "selic", "spread", "dlog_concessoes_reais", "inadimplencia"}.issubset(model.columns)
    assert model["dlog_concessoes_reais"].notna().all()

