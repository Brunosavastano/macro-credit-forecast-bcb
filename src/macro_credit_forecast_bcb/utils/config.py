from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from macro_credit_forecast_bcb.utils.paths import CONFIG_DIR


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as file:
        loaded = yaml.safe_load(file) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Config file must contain a mapping: {path}")
    return loaded


def load_series_config(path: str | Path | None = None) -> dict[str, Any]:
    return load_yaml(path or CONFIG_DIR / "series_sgs.yaml")


def load_model_config(path: str | Path | None = None) -> dict[str, Any]:
    return load_yaml(path or CONFIG_DIR / "model.yaml")

