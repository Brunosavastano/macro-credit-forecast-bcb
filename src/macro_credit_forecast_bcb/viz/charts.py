from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px


VARIABLE_LABELS = {
    "ipca": "IPCA mensal",
    "ipca_12m": "IPCA acumulado em 12 meses",
    "selic": "Selic meta",
    "spread": "Spread medio de credito",
    "dlog_concessoes_reais": "Crescimento real das concessoes",
    "inadimplencia": "Inadimplencia",
    "concessoes_reais": "Concessoes reais",
}

VARIABLE_UNITS = {
    "ipca": "% m/m",
    "ipca_12m": "% 12m",
    "selic": "% a.a.",
    "spread": "p.p.",
    "dlog_concessoes_reais": "% m/m real",
    "inadimplencia": "%",
    "concessoes_reais": "R$ milhoes, precos constantes",
}


def label(variable: str) -> str:
    return VARIABLE_LABELS.get(variable, variable)


def unit(variable: str) -> str:
    return VARIABLE_UNITS.get(variable, "")


def history_forecast_chart(
    history: pd.DataFrame,
    forecast: pd.DataFrame,
    variable: str,
    *,
    interval: str = "95",
    title: str | None = None,
) -> go.Figure:
    fig = go.Figure()
    title = title or f"{label(variable)}: historico e forecast"
    hist = history[[variable]].dropna() if variable in history.columns else pd.DataFrame()
    if not hist.empty:
        fig.add_trace(
            go.Scatter(
                x=hist.index,
                y=hist[variable],
                mode="lines",
                name="Historico",
                line=dict(color="#0f172a", width=2),
            )
        )

    fc = forecast.loc[forecast["variable"] == variable].sort_values("date")
    if not fc.empty:
        lower = f"lower_{interval}"
        upper = f"upper_{interval}"
        if lower in fc.columns and upper in fc.columns and fc[lower].notna().any():
            fig.add_trace(
                go.Scatter(
                    x=fc["date"],
                    y=fc[upper],
                    mode="lines",
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=fc["date"],
                    y=fc[lower],
                    mode="lines",
                    fill="tonexty",
                    fillcolor="rgba(37, 99, 235, 0.16)",
                    line=dict(width=0),
                    name=f"IC {interval}%",
                    hoverinfo="skip",
                )
            )
        fig.add_trace(
            go.Scatter(
                x=fc["date"],
                y=fc["forecast"],
                mode="lines+markers",
                name="Forecast",
                line=dict(color="#2563eb", width=3),
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="Data",
        yaxis_title=unit(variable),
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=30, r=20, t=70, b=35),
    )
    return fig


def metrics_heatmap(metrics: pd.DataFrame, variable: str, metric: str = "rmse") -> go.Figure:
    subset = metrics.loc[metrics["variable"] == variable]
    if subset.empty:
        return go.Figure().update_layout(title=f"Sem metricas para {label(variable)}")
    pivot = subset.pivot_table(index="model", columns="horizon", values=metric, aggfunc="mean")
    fig = px.imshow(
        pivot,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="Blues",
        title=f"{metric.upper()} por modelo e horizonte - {label(variable)}",
    )
    fig.update_layout(template="plotly_white", margin=dict(l=30, r=20, t=70, b=35))
    return fig


def residual_correlation_chart(correlation: pd.DataFrame | dict) -> go.Figure:
    corr = pd.DataFrame(correlation)
    fig = px.imshow(
        corr,
        zmin=-1,
        zmax=1,
        color_continuous_scale="RdBu",
        text_auto=".2f",
        title="Correlacao dos residuos",
    )
    fig.update_layout(template="plotly_white", margin=dict(l=30, r=20, t=70, b=35))
    return fig


def forecast_table(forecast: pd.DataFrame) -> pd.DataFrame:
    if forecast.empty:
        return forecast
    table = forecast.copy()
    table["date"] = pd.to_datetime(table["date"]).dt.date
    return table[
        [
            "date",
            "horizon",
            "variable",
            "model",
            "forecast",
            "lower_68",
            "upper_68",
            "lower_95",
            "upper_95",
        ]
    ]

