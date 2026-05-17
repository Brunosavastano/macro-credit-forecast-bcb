from __future__ import annotations

import logging
from typing import Any
from urllib.parse import quote

import pandas as pd
import requests

LOGGER = logging.getLogger(__name__)
FOCUS_BASE_URL = "https://olinda.bcb.gov.br/olinda/servico/Expectativas/versao/v1/odata"


class FocusAPIError(RuntimeError):
    """Raised when the Focus/OData API cannot be queried safely."""


def _format_filter_value(value: Any) -> str:
    if isinstance(value, str):
        escaped = value.replace("'", "''")
        return f"'{escaped}'"
    return str(value)


def build_odata_filter(filters: dict[str, Any] | None = None) -> str | None:
    if not filters:
        return None
    clauses = []
    for key, value in filters.items():
        if " " in key:
            clauses.append(f"{key} {_format_filter_value(value)}")
        else:
            clauses.append(f"{key} eq {_format_filter_value(value)}")
    return " and ".join(clauses)


def get_focus_expectations(
    resource: str,
    filters: dict[str, Any] | None = None,
    *,
    top: int = 1000,
    orderby: str = "Data desc",
    session: requests.Session | None = None,
    timeout: int = 30,
) -> pd.DataFrame:
    """Query a Focus expectations OData resource.

    The function returns the API columns unchanged, with date-like columns parsed
    when present. Callers can decide how to map Focus horizons to model horizons.
    """
    http = session or requests.Session()
    params = {"$top": top, "$orderby": orderby, "$format": "json"}
    odata_filter = build_odata_filter(filters)
    if odata_filter:
        params["$filter"] = odata_filter

    encoded_resource = quote(resource, safe="")
    url = f"{FOCUS_BASE_URL}/{encoded_resource}"
    LOGGER.info("Downloading Focus resource %s", resource)
    response = http.get(url, params=params, timeout=timeout)
    if response.status_code != 200:
        raise FocusAPIError(
            f"Focus request failed for {resource}: "
            f"HTTP {response.status_code} - {response.text[:200]}"
        )
    try:
        payload = response.json()
    except ValueError as exc:
        raise FocusAPIError(f"Invalid JSON returned by Focus resource {resource}") from exc

    records = payload.get("value")
    if records is None:
        raise FocusAPIError(f"Unexpected Focus payload for {resource}: missing 'value'")

    frame = pd.DataFrame(records)
    for column in frame.columns:
        if column.lower().startswith("data") or column in {"Reuniao", "Referencia"}:
            parsed = pd.to_datetime(frame[column], errors="coerce")
            if parsed.notna().any():
                frame[column] = parsed
    return frame


def download_focus_snapshot(top: int = 5000) -> dict[str, pd.DataFrame]:
    """Download the Focus resources used by the dashboard, best-effort."""
    resources = {
        "ipca_mensal": (
            "ExpectativaMercadoMensais",
            {"Indicador": "IPCA"},
            "Data desc",
        ),
        "ipca_12m": (
            "ExpectativasMercadoInflacao12Meses",
            {"Indicador": "IPCA"},
            "Data desc",
        ),
        "selic_reunioes": (
            "ExpectativasMercadoSelic",
            None,
            "Data desc",
        ),
        "anuais": (
            "ExpectativasMercadoAnuais",
            None,
            "Data desc",
        ),
    }
    output: dict[str, pd.DataFrame] = {}
    for name, (resource, filters, orderby) in resources.items():
        try:
            output[name] = get_focus_expectations(
                resource,
                filters=filters,
                top=top,
                orderby=orderby,
            )
        except Exception as exc:  # pragma: no cover - depends on external API
            LOGGER.warning("Focus download skipped for %s: %s", name, exc)
            output[name] = pd.DataFrame()
    return output

