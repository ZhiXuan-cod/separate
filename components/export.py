# components/export.py
# NOTE: go_to is passed as an argument — do NOT import from app.py (circular import).

from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from utils.helpers import text_to_simple_pdf_bytes
from utils.state import reset_ml_state


def export_page(go_to) -> None:
    """Export page — generate and download a PDF summary report."""
    if not st.session_state.training_complete:
        st.warning("⚠️ Train a model first.")
        return

    st.markdown('<h2 class="sub-header">💾 Export Results</h2>', unsafe_allow_html=True)

    problem_type: str = st.session_state.problem_type or ""
    metrics = st.session_state.cluster_metrics or {}

    # ── Model summary (always visible, no button) ─────────────────────────────
    st.markdown("### 📊 Model Summary")
    if problem_type == "Clustering":
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Algorithm", metrics.get("algorithm", "N/A"))
        col_b.metric("Clusters Found", metrics.get("num_clusters", "N/A"))
        sil = metrics.get("silhouette_score")
        col_c.metric("Silhouette Score", f"{sil:.4f}" if sil is not None else "N/A")

        # Cluster size breakdown
        sizes = metrics.get("cluster_sizes", {})
        if sizes:
            size_df = pd.DataFrame(
                [("Noise" if k == -1 else f"Cluster {k}", v)
                 for k, v in sorted(sizes.items())],
                columns=["Cluster", "Points"],
            )
            st.dataframe(size_df, use_container_width=True)
    else:
        col_a, col_b = st.columns(2)
        col_a.write(f"**Task:** {problem_type}")
        col_a.write(f"**Target column:** {st.session_state.target_column}")
        col_a.write(f"**Model:** {st.session_state.model}")

        preds  = st.session_state.predictions
        y_true = st.session_state.test_labels
        if preds is not None and y_true is not None:
            preds, y_true = np.array(preds), np.array(y_true)
            with col_b:
                st.markdown("**Test-set performance:**")
                if problem_type == "Classification":
                    st.write(f"• Accuracy: **{accuracy_score(y_true, preds):.4f}**")
                    st.write(f"• F1 (weighted): **{f1_score(y_true, preds, average='weighted', zero_division=0):.4f}**")
                elif problem_type == "Regression":
                    st.write(f"• R²: **{r2_score(y_true, preds):.4f}**")
                    st.write(f"• MAE: **{mean_absolute_error(y_true, preds):.4f}**")
                    st.write(f"• RMSE: **{np.sqrt(mean_squared_error(y_true, preds)):.4f}**")

    # ── PDF report ────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📄 Download Report")
    if st.button("Generate PDF Report", key="generate_report"):
        report = _build_report(problem_type, metrics)
        st.code(report, language="text")
        st.download_button(
            "📥 Download PDF",
            text_to_simple_pdf_bytes(report, title="ML Model Report"),
            "ml_model_report.pdf",
            mime="application/pdf",
            key="download_pdf",
        )

    # ── Session summary ────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📋 Session Summary")
    st.dataframe(
        pd.DataFrame.from_dict({
            "Data loaded":       str(st.session_state.data is not None),
            "Problem type":      str(problem_type) or "N/A",
            "Target column":     str(st.session_state.target_column or "N/A"),
            "Model trained":     str(st.session_state.training_complete),
            "Results available": str(
                st.session_state.predictions is not None
                or st.session_state.cluster_labels is not None
            ),
        }, orient="index", columns=["Status"]),
        use_container_width=True,
    )

    # ── Start over ─────────────────────────────────────────────────────────────
    st.markdown("---")
    if st.button("🔄 Start Over (clear all data)", type="secondary", key="start_over"):
        reset_ml_state()
        go_to("data_upload")


# ── Report builder ────────────────────────────────────────────────────────────

def _fmt(val, spec: str) -> str:
    if val is None:
        return "N/A"
    try:
        return format(float(val), spec)
    except (TypeError, ValueError):
        return str(val)


def _build_report(problem_type: str, metrics: dict) -> str:
    ts    = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    shape = st.session_state.data.shape if st.session_state.data is not None else "N/A"

    if problem_type == "Clustering":
        return (
            f"Clustering Report (AutoML)\n"
            f"Generated     : {ts}\n\n"
            f"Algorithm     : {metrics.get('algorithm', 'N/A')}\n"
            f"Dataset shape : {shape}\n"
            f"Clusters      : {metrics.get('num_clusters', 'N/A')}\n"
            f"Silhouette    : {_fmt(metrics.get('silhouette_score'), '.4f')}\n"
            f"Calinski-H    : {_fmt(metrics.get('calinski_harabasz'), '.2f')}\n"
            f"Davies-Bouldin: {_fmt(metrics.get('davies_bouldin'), '.4f')}\n"
        )

    preds  = st.session_state.predictions
    y_true = st.session_state.test_labels
    perf   = ""
    if preds is not None and y_true is not None:
        preds, y_true = np.array(preds), np.array(y_true)
        if problem_type == "Classification":
            perf = (
                f"Accuracy      : {accuracy_score(y_true, preds):.4f}\n"
                f"F1 (weighted) : {f1_score(y_true, preds, average='weighted', zero_division=0):.4f}\n"
            )
        elif problem_type == "Regression":
            perf = (
                f"R² Score      : {r2_score(y_true, preds):.4f}\n"
                f"MAE           : {mean_absolute_error(y_true, preds):.4f}\n"
                f"RMSE          : {np.sqrt(mean_squared_error(y_true, preds)):.4f}\n"
            )

    return (
        f"ML Model Report (AutoML)\n"
        f"Generated     : {ts}\n\n"
        f"Task          : {problem_type}\n"
        f"Target        : {st.session_state.target_column}\n"
        f"Dataset shape : {shape}\n"
        f"Model         : {st.session_state.model}\n"
        + (f"\n--- Test-set Performance ---\n{perf}" if perf else "")
    )
