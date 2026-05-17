from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "app"))

import pandas as pd
import streamlit as st

from app_utils import (
    apply_app_style,
    dataframe_download,
    load_audit_summary,
    load_audit_table,
    missing_artifacts_message,
    render_status_pill,
)
from macro_credit_forecast_bcb.viz.charts import VARIABLE_LABELS, metrics_heatmap
from macro_credit_forecast_bcb.viz.formatting import format_metric, label


st.set_page_config(page_title="Econometric Audit", layout="wide")
apply_app_style()
st.title("Econometric Audit")
st.caption("Auditoria de acurácia, resíduos, intervalos e robustez contra benchmarks.")

summary = load_audit_summary()
scorecard = load_audit_table("econometric_scorecard.parquet")
audit_metrics = load_audit_table("econometric_audit_metrics.parquet")
coverage = load_audit_table("interval_coverage.parquet")
dm_tests = load_audit_table("model_comparison_tests.parquet")
residuals = load_audit_table("residual_equation_diagnostics.parquet")
transformations = load_audit_table("transformation_audit.parquet")
recommendations = load_audit_table("model_recommendations.parquet")
specifications = load_audit_table("specification_diagnostics.parquet")
transformation_candidates = load_audit_table("transformation_candidates.parquet")

if not summary or scorecard.empty:
    missing_artifacts_message()
    st.code("python -m macro_credit_forecast_bcb.pipeline.audit", language="bash")
    st.stop()

cols = st.columns([0.25, 0.75])
with cols[0]:
    render_status_pill(summary.get("overall_status", "warning"))
with cols[1]:
    st.caption("O scorecard não troca o modelo automaticamente; ele aponta onde a evidência é fraca.")

st.subheader("Scorecard")
st.dataframe(
    scorecard,
    use_container_width=True,
    hide_index=True,
    column_config={
        "dimension": st.column_config.TextColumn("Dimensão"),
        "status": st.column_config.TextColumn("Status"),
        "evidence": st.column_config.TextColumn("Evidência"),
        "recommendation": st.column_config.TextColumn("Recomendação"),
    },
)

st.subheader("Recomendação operacional por horizonte")
if recommendations.empty:
    st.info("Recomendações ainda não foram geradas. Reexecute o pipeline de auditoria.")
else:
    rec_display = recommendations.copy()
    rec_display["variable"] = rec_display["variable"].map(label)
    st.dataframe(
        rec_display.sort_values(["variable", "horizon"]),
        use_container_width=True,
        hide_index=True,
        column_config={
            "horizon": st.column_config.NumberColumn("Horizonte", format="%d"),
            "rmse": st.column_config.NumberColumn("RMSE recomendado", format="%.4f"),
            "mae": st.column_config.NumberColumn("MAE recomendado", format="%.4f"),
            "base_var_rmse": st.column_config.NumberColumn("RMSE VAR base", format="%.4f"),
            "best_benchmark_rmse": st.column_config.NumberColumn("RMSE melhor benchmark", format="%.4f"),
            "rmse_gain_vs_base_var": st.column_config.NumberColumn("Ganho vs VAR base", format="%.1%"),
        },
    )

st.subheader("Ranking de acurácia")
if audit_metrics.empty:
    st.info("Métricas de auditoria não encontradas.")
else:
    variable_options = [v for v in VARIABLE_LABELS if v in audit_metrics["variable"].unique()]
    variable = st.selectbox("Indicador", variable_options, format_func=label)
    metric = st.selectbox("Métrica", ["rmse", "mae", "mase", "theil_u", "smape"], index=0)
    st.plotly_chart(metrics_heatmap(audit_metrics, variable, metric=metric), use_container_width=True)
    table = audit_metrics.loc[audit_metrics["variable"] == variable].sort_values(["horizon", "rmse_rank"]).copy()
    display = table.copy()
    display["variable"] = display["variable"].map(label)
    for column in ["mae", "rmse", "smape", "mase", "theil_u", "rmse_rank"]:
        if column in display:
            display[column] = display[column].map(lambda value: format_metric(value, 4))
    st.dataframe(display, use_container_width=True, hide_index=True)
    dataframe_download(table, f"audit_metrics_{variable}.csv")

st.subheader("Cobertura dos intervalos")
if coverage.empty:
    st.info("Coverage indisponível. Reexecute o backtest para salvar intervalos do VAR.")
else:
    coverage_display = coverage.copy()
    coverage_display["variable"] = coverage_display["variable"].map(label)
    st.dataframe(
        coverage_display,
        use_container_width=True,
        hide_index=True,
        column_config={
            "coverage_68": st.column_config.NumberColumn("Coverage 68%", format="%.1%"),
            "coverage_95": st.column_config.NumberColumn("Coverage 95%", format="%.1%"),
            "avg_width_68": st.column_config.NumberColumn("Largura 68%", format="%.4f"),
            "avg_width_95": st.column_config.NumberColumn("Largura 95%", format="%.4f"),
        },
    )

st.subheader("Diebold-Mariano: VAR vs benchmarks")
if dm_tests.empty:
    st.info("Testes Diebold-Mariano indisponíveis.")
else:
    dm_display = dm_tests.copy()
    dm_display["variable"] = dm_display["variable"].map(label)
    st.dataframe(
        dm_display.sort_values(["variable", "horizon", "pvalue"]),
        use_container_width=True,
        hide_index=True,
        column_config={
            "dm_stat": st.column_config.NumberColumn("DM stat", format="%.3f"),
            "pvalue": st.column_config.NumberColumn("p-value", format="%.4f"),
        },
    )

st.subheader("Robustez de especificações")
if specifications.empty:
    st.info("Diagnósticos das especificações alternativas indisponíveis.")
else:
    st.dataframe(
        specifications,
        use_container_width=True,
        hide_index=True,
        column_config={
            "selected_lag": st.column_config.NumberColumn("Lag", format="%d"),
            "maxlags": st.column_config.NumberColumn("Maxlags", format="%d"),
            "aic": st.column_config.NumberColumn("AIC", format="%.3f"),
            "bic": st.column_config.NumberColumn("BIC", format="%.3f"),
            "roots_abs_min": st.column_config.NumberColumn("Menor |raiz|", format="%.3f"),
        },
    )

st.subheader("Transformações candidatas")
if transformation_candidates.empty:
    st.info("Candidatos de transformação indisponíveis.")
else:
    candidates = transformation_candidates.copy()
    for column in ["adf_pvalue", "kpss_pvalue"]:
        if column in candidates:
            candidates[column] = pd.to_numeric(candidates[column], errors="coerce")
    st.dataframe(
        candidates.sort_values(["model_variable", "current_model", "candidate"], ascending=[True, False, True]),
        use_container_width=True,
        hide_index=True,
        column_config={
            "current_model": st.column_config.CheckboxColumn("Atual"),
            "adf_pvalue": st.column_config.NumberColumn("ADF p-value", format="%.4f"),
            "kpss_pvalue": st.column_config.NumberColumn("KPSS p-value", format="%.4f"),
        },
    )

st.subheader("Resíduos e transformações")
left, right = st.columns(2)
with left:
    st.write("Diagnósticos por equação")
    st.dataframe(
        residuals,
        use_container_width=True,
        hide_index=True,
        column_config={
            "ljung_box_pvalue": st.column_config.NumberColumn("Ljung-Box p-value", format="%.4f"),
            "arch_pvalue": st.column_config.NumberColumn("ARCH p-value", format="%.4f"),
            "outlier_share": st.column_config.NumberColumn("Outliers", format="%.1%"),
        },
    )
with right:
    st.write("Auditoria de transformações")
    st.dataframe(transformations, use_container_width=True, hide_index=True)
