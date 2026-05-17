from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.vector_ar.vecm import VECM, coint_johansen, select_coint_rank


def select_vecm_rank(
    frame: pd.DataFrame,
    *,
    det_order: int = 0,
    k_ar_diff: int = 1,
    signif: float = 0.05,
) -> dict[str, object]:
    data = frame.dropna().astype(float)
    if data.shape[0] < 30 or data.shape[1] < 2:
        return {"rank": 0, "reason": "Insufficient observations or variables for Johansen test"}
    try:
        rank_result = select_coint_rank(
            data,
            det_order=det_order,
            k_ar_diff=k_ar_diff,
            signif=signif,
            method="trace",
        )
        johansen = coint_johansen(data, det_order=det_order, k_ar_diff=k_ar_diff)
        return {
            "rank": int(rank_result.rank),
            "trace_stats": [float(x) for x in np.asarray(johansen.lr1)],
            "crit_values_95": [float(x) for x in np.asarray(johansen.cvt)[:, 1]],
            "det_order": det_order,
            "k_ar_diff": k_ar_diff,
            "signif": signif,
        }
    except Exception as exc:
        return {"rank": 0, "reason": str(exc)}


def fit_vecm(frame: pd.DataFrame, rank: int, *, k_ar_diff: int = 1, deterministic: str = "co"):
    if rank <= 0:
        raise ValueError("VECM rank must be positive")
    data = frame.dropna().astype(float)
    return VECM(data, k_ar_diff=k_ar_diff, coint_rank=rank, deterministic=deterministic).fit()

