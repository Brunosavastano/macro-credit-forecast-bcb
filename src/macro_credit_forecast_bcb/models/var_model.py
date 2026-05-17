from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR
from statsmodels.tsa.vector_ar.var_model import VARResults


def fit_var(frame: pd.DataFrame, lag: int) -> VARResults:
    data = frame.dropna().astype(float)
    if lag < 1:
        raise ValueError("VAR lag must be at least 1")
    return VAR(data).fit(lag)


def _test_result_payload(result: Any) -> dict[str, object]:
    if result is None:
        return {}
    return {
        "statistic": getattr(result, "test_statistic", None),
        "pvalue": getattr(result, "pvalue", None),
        "df": getattr(result, "df", None),
        "conclusion": getattr(result, "conclusion", None),
    }


def var_diagnostics(result: VARResults, *, max_whiteness_lag: int = 12) -> dict[str, object]:
    resid = pd.DataFrame(result.resid, columns=result.names)
    diagnostics: dict[str, object] = {
        "is_stable": bool(result.is_stable(verbose=False)),
        "nobs": int(result.nobs),
        "lag_order": int(result.k_ar),
        "aic": float(result.aic),
        "bic": float(result.bic),
        "hqic": float(result.hqic),
        "fpe": float(result.fpe),
        "roots_abs_min": float(np.min(np.abs(result.roots))) if len(result.roots) else None,
        "residual_correlation": resid.corr().to_dict(),
    }
    try:
        whiteness_lag = max(result.k_ar + 1, min(max_whiteness_lag, result.nobs // 4))
        diagnostics["whiteness"] = _test_result_payload(result.test_whiteness(nlags=whiteness_lag))
    except Exception as exc:
        diagnostics["whiteness_error"] = str(exc)
    try:
        diagnostics["normality"] = _test_result_payload(result.test_normality())
    except Exception as exc:
        diagnostics["normality_error"] = str(exc)
    return diagnostics

