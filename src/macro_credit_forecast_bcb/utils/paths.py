from __future__ import annotations

import os
from pathlib import Path


def project_root(start: Path | None = None) -> Path:
    """Resolve the project root without depending on an absolute local path."""
    env_root = os.getenv("MACRO_FORECAST_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()

    current = (start or Path(__file__)).resolve()
    for parent in [current, *current.parents]:
        if (parent / "pyproject.toml").exists() or (
            parent / "spec_macro_credit_forecast_bcb.md"
        ).exists():
            return parent
    return Path.cwd().resolve()


ROOT = project_root()
CONFIG_DIR = ROOT / "config"
DATA_DIR = ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
FORECAST_DATA_DIR = DATA_DIR / "forecasts"
OUTPUTS_DIR = ROOT / "outputs"


def ensure_project_dirs() -> None:
    for path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, FORECAST_DATA_DIR, OUTPUTS_DIR]:
        path.mkdir(parents=True, exist_ok=True)

