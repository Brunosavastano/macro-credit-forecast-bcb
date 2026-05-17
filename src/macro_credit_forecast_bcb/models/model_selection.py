from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR


def feasible_maxlags(frame: pd.DataFrame, requested: int) -> int:
    nobs, nvars = frame.dropna().shape
    if nobs < nvars * 3:
        return 1
    conservative = max(1, (nobs - 1) // (nvars + 2))
    return max(1, min(int(requested), conservative))


def select_var_lag(
    frame: pd.DataFrame,
    *,
    maxlags: int = 6,
    criterion: str = "bic",
) -> dict[str, object]:
    """Select VAR lag by fitting feasible lag orders and comparing ICs."""
    data = frame.dropna().astype(float)
    if data.shape[0] < 20:
        raise ValueError("At least 20 observations are required for VAR lag selection")

    maxlags = feasible_maxlags(data, maxlags)
    rows: list[dict[str, float | int]] = []
    model = VAR(data)
    for lag in range(1, maxlags + 1):
        try:
            result = model.fit(lag)
            rows.append(
                {
                    "lag": lag,
                    "aic": float(result.aic),
                    "bic": float(result.bic),
                    "hqic": float(result.hqic),
                    "fpe": float(result.fpe),
                }
            )
        except Exception:
            continue
    if not rows:
        raise ValueError("No feasible VAR lag could be estimated")

    table = pd.DataFrame(rows)
    criterion = criterion.lower()
    if criterion not in table.columns:
        raise ValueError(f"Unsupported criterion: {criterion}")
    selected_row = table.loc[table[criterion].idxmin()]
    selected_lag = int(selected_row["lag"])
    return {
        "selected_lag": selected_lag,
        "criterion": criterion,
        "maxlags_requested": int(maxlags),
        "ic_table": table,
        "selected_orders": {
            metric: int(table.loc[table[metric].idxmin(), "lag"])
            for metric in ["aic", "bic", "hqic", "fpe"]
            if metric in table
        },
    }


def information_criteria_to_records(selection: dict[str, object]) -> list[dict[str, object]]:
    table = selection.get("ic_table")
    if isinstance(table, pd.DataFrame):
        return table.replace({np.nan: None}).to_dict(orient="records")
    return []

