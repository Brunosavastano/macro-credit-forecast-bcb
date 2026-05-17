from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class VariableDisplay:
    label: str
    unit: str
    decimals: int = 2
    tickformat: str = ".2f"
    prefix: str = ""
    suffix: str = ""


VARIABLE_DISPLAY: dict[str, VariableDisplay] = {
    "ipca": VariableDisplay("IPCA mensal", "% m/m", decimals=2, tickformat=".2f", suffix="%"),
    "ipca_12m": VariableDisplay("IPCA acumulado em 12 meses", "% 12m", decimals=2, tickformat=".2f", suffix="%"),
    "selic": VariableDisplay("Selic meta", "% a.a.", decimals=2, tickformat=".2f", suffix="%"),
    "spread": VariableDisplay("Spread medio de credito", "p.p.", decimals=2, tickformat=".2f", suffix=" p.p."),
    "dlog_concessoes_reais": VariableDisplay(
        "Crescimento real das concessoes",
        "% m/m",
        decimals=2,
        tickformat=".2f",
        suffix="%",
    ),
    "inadimplencia": VariableDisplay("Inadimplencia", "%", decimals=2, tickformat=".2f", suffix="%"),
    "concessoes_reais": VariableDisplay(
        "Concessoes reais",
        "R$ milhoes, precos constantes",
        decimals=0,
        tickformat=",.0f",
        prefix="R$ ",
    ),
}


def variable_display_metadata(variable: str) -> VariableDisplay:
    return VARIABLE_DISPLAY.get(variable, VariableDisplay(variable, ""))


def label(variable: str) -> str:
    return variable_display_metadata(variable).label


def unit(variable: str) -> str:
    return variable_display_metadata(variable).unit


def tickformat(variable: str) -> str:
    return variable_display_metadata(variable).tickformat


def format_value(variable: str, value: Any) -> str:
    if value is None or pd.isna(value):
        return "-"
    metadata = variable_display_metadata(variable)
    numeric = float(value)
    formatted = f"{numeric:,.{metadata.decimals}f}"
    return f"{metadata.prefix}{formatted}{metadata.suffix}"


def format_metric(value: Any, decimals: int = 3) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{float(value):,.{decimals}f}"


def format_table_for_display(frame: pd.DataFrame, variable_column: str = "variable") -> pd.DataFrame:
    table = frame.copy()
    if variable_column in table.columns:
        table[variable_column] = table[variable_column].map(label)
    for column in table.columns:
        if column == "date":
            table[column] = pd.to_datetime(table[column]).dt.strftime("%Y-%m-%d")
        elif column in {"forecast", "lower_68", "upper_68", "lower_95", "upper_95", "actual"}:
            if variable_column in frame.columns:
                table[column] = [
                    format_value(variable, value)
                    for variable, value in zip(frame[variable_column], frame[column], strict=False)
                ]
            else:
                table[column] = table[column].map(lambda value: format_metric(value, 3))
        elif np.issubdtype(table[column].dtype, np.number):
            table[column] = table[column].map(lambda value: format_metric(value, 3))
    return table

