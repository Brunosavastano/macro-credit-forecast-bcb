from __future__ import annotations

import pandas as pd

from macro_credit_forecast_bcb.viz.formatting import format_value, label, unit


EXECUTIVE_VARIABLES = ["ipca", "ipca_12m", "selic", "spread", "concessoes_reais", "inadimplencia"]


def latest_values_table(
    history: pd.DataFrame,
    forecast: pd.DataFrame,
    horizon: int = 12,
    *,
    formatted: bool = False,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for variable in EXECUTIVE_VARIABLES:
        latest_value = history[variable].dropna().iloc[-1] if variable in history else None
        forecast_match = forecast[
            (forecast["variable"] == variable) & (forecast["horizon"] == horizon)
        ]
        forecast_value = forecast_match["forecast"].iloc[0] if not forecast_match.empty else None
        latest_display = format_value(variable, latest_value) if formatted else latest_value
        forecast_display = format_value(variable, forecast_value) if formatted else forecast_value
        rows.append(
            {
                "Indicador": label(variable),
                "Unidade": unit(variable),
                "Último observado": latest_display,
                f"Forecast h={horizon}": forecast_display,
            }
        )
    return pd.DataFrame(rows)
