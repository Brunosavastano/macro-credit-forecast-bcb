from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import timedelta
from typing import Iterable

import pandas as pd
import requests

from macro_credit_forecast_bcb.utils.dates import format_bcb_date, parse_date

LOGGER = logging.getLogger(__name__)
SGS_URL = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.{code}/dados"


class BCBAPIError(RuntimeError):
    """Raised when a BCB API request cannot be completed safely."""


@dataclass(frozen=True)
class SGSRequest:
    code: int
    start: pd.Timestamp
    end: pd.Timestamp


def _to_float(value: object) -> float:
    if value is None:
        return float("nan")
    text = str(value).strip()
    if not text:
        return float("nan")

    if "," in text and "." in text:
        if text.rfind(",") > text.rfind("."):
            text = text.replace(".", "").replace(",", ".")
        else:
            text = text.replace(",", "")
    elif "," in text:
        text = text.replace(".", "").replace(",", ".")
    elif text.count(".") > 1:
        parts = text.split(".")
        text = "".join(parts[:-1]) + "." + parts[-1]

    return float(text)


def _parse_sgs_payload(payload: list[dict[str, object]], name: str | None = None) -> pd.Series:
    if not payload:
        return pd.Series(dtype="float64", name=name)

    frame = pd.DataFrame(payload)
    required = {"data", "valor"}
    missing = required.difference(frame.columns)
    if missing:
        raise BCBAPIError(f"Unexpected SGS payload, missing columns: {sorted(missing)}")

    frame["date"] = pd.to_datetime(frame["data"], dayfirst=True, errors="coerce")
    frame["value"] = frame["valor"].map(_to_float)
    frame = frame.dropna(subset=["date"]).sort_values("date")
    frame = frame.drop_duplicates(subset=["date"], keep="last")

    series = pd.Series(frame["value"].to_numpy(), index=frame["date"], name=name)
    series.index = pd.DatetimeIndex(series.index)
    return series.astype(float)


def _date_windows(start: pd.Timestamp, end: pd.Timestamp, years: int = 5) -> Iterable[tuple[pd.Timestamp, pd.Timestamp]]:
    current = start
    step_days = int(years * 365.25)
    while current <= end:
        window_end = min(current + timedelta(days=step_days), end)
        yield current, window_end
        current = window_end + timedelta(days=1)


def _request_sgs(
    request: SGSRequest,
    session: requests.Session | None = None,
    timeout: int = 60,
    retries: int = 3,
) -> pd.Series:
    http = session or requests.Session()
    params = {
        "formato": "json",
        "dataInicial": format_bcb_date(request.start),
        "dataFinal": format_bcb_date(request.end),
    }
    url = SGS_URL.format(code=request.code)
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            response = http.get(url, params=params, timeout=timeout)
            break
        except requests.RequestException as exc:
            last_error = exc
            if attempt == retries:
                raise BCBAPIError(
                    f"SGS request failed for code {request.code} after {retries} attempts: {exc}"
                ) from exc
            sleep_seconds = 1.5 * attempt
            LOGGER.warning(
                "SGS request retry %s/%s for code %s after error: %s",
                attempt,
                retries,
                request.code,
                exc,
            )
            time.sleep(sleep_seconds)
    else:  # pragma: no cover - defensive
        raise BCBAPIError(f"SGS request failed for code {request.code}: {last_error}")

    if response.status_code != 200:
        raise BCBAPIError(
            f"SGS request failed for code {request.code}: "
            f"HTTP {response.status_code} - {response.text[:200]}"
        )
    try:
        payload = response.json()
    except ValueError as exc:
        raise BCBAPIError(f"Invalid JSON returned by SGS code {request.code}") from exc
    return _parse_sgs_payload(payload)


def get_sgs_series(
    code: int,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp | None = None,
    *,
    name: str | None = None,
    frequency: str | None = None,
    session: requests.Session | None = None,
    timeout: int = 60,
) -> pd.Series:
    """Download one SGS series as a numeric pandas Series.

    Daily SGS series are requested in windows to respect BCB period limits.
    """
    start_ts = parse_date(start)
    end_ts = parse_date(end)
    if start_ts > end_ts:
        raise ValueError("start must be before end")

    LOGGER.info("Downloading SGS %s from %s to %s", code, start_ts.date(), end_ts.date())
    if str(frequency or "").upper() == "D":
        pieces = [
            _request_sgs(SGSRequest(code, window_start, window_end), session, timeout)
            for window_start, window_end in _date_windows(start_ts, end_ts)
        ]
        series = pd.concat(pieces).sort_index() if pieces else pd.Series(dtype="float64")
        series = series[~series.index.duplicated(keep="last")]
    else:
        series = _request_sgs(SGSRequest(code, start_ts, end_ts), session, timeout)

    series.name = name or str(code)
    return series.astype(float)
