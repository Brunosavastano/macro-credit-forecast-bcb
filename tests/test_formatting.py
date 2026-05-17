from __future__ import annotations

import pandas as pd

from macro_credit_forecast_bcb.viz.formatting import format_table_for_display, format_value


def test_format_value_uses_variable_units_and_decimals() -> None:
    assert format_value("ipca", 0.88) == "0.88%"
    assert format_value("selic", 14.75) == "14.75%"
    assert format_value("spread", 21.84) == "21.84 p.p."
    assert format_value("concessoes_reais", 732939.4) == "R$ 732,939"


def test_format_table_for_display_formats_forecast_by_variable() -> None:
    frame = pd.DataFrame(
        {
            "date": ["2026-03-31"],
            "variable": ["ipca"],
            "forecast": [0.88],
            "lower_68": [0.5],
            "upper_68": [1.0],
        }
    )

    table = format_table_for_display(frame)

    assert table.loc[0, "variable"] == "IPCA mensal"
    assert table.loc[0, "forecast"] == "0.88%"

