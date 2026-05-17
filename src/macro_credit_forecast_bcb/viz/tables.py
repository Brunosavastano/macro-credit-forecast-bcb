from __future__ import annotations

import pandas as pd

from macro_credit_forecast_bcb.viz.charts import label, unit


def latest_values_table(history: pd.DataFrame, forecast: pd.DataFrame, horizon: int = 12) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for variable in ["ipca", "ipca_12m", "selic", "spread", "dlog_concessoes_reais", "inadimplencia"]:
        latest_value = history[variable].dropna().iloc[-1] if variable in history else None
        forecast_match = forecast[
            (forecast["variable"] == variable) & (forecast["horizon"] == horizon)
        ]
        forecast_value = forecast_match["forecast"].iloc[0] if not forecast_match.empty else None
        rows.append(
            {
                "Indicador": label(variable),
                "Unidade": unit(variable),
                "Ultimo observado": latest_value,
                f"Forecast h={horizon}": forecast_value,
            }
        )
    return pd.DataFrame(rows)

