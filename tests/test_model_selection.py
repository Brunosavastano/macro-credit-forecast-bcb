from __future__ import annotations

import numpy as np
import pandas as pd

from macro_credit_forecast_bcb.models.model_selection import select_var_lag


def test_select_var_lag_returns_positive_lag() -> None:
    rng = np.random.default_rng(42)
    index = pd.date_range("2015-01-31", periods=80, freq="ME")
    x = rng.normal(size=(80, 3))
    frame = pd.DataFrame(x, index=index, columns=["a", "b", "c"])

    selection = select_var_lag(frame, maxlags=3, criterion="bic")

    assert selection["selected_lag"] >= 1
    assert not selection["ic_table"].empty

