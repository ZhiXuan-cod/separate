# components/evaluation.py
#
# Design: each task type has a public show_*_results() function.
#   - training.py calls these right after training to display inline results.
#   - evaluation_page() calls the same functions for the dedicated page.
# This guarantees charts are identical in both places with no code duplication.

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    calinski_harabasz_score,
    classification_report,
    confusion_matrix,
    davies_bouldin_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    silhouette_score,
)

from utils.helpers import metric_row, pca_scatter_fig


# ── Evaluation page entry point ───────────────────────────────────────────────

def evaluation_page() -> None:
    """Dedicated evaluation page — same charts as shown right after training."""
    if not st.session_state.training_complete:
        st.warning("⚠️ Train a model first.")
        return

    problem_type: str = st.session_state.problem_type or ""
    st.markdown('<h2 class="sub-header">📈 Model Evaluation</h2>', unsafe_allow_html=True)

    if not problem_type:
        # Should not happen after training, but guard gracefully
        st.info("No problem type set — please go to Data Upload and configure a task.")
        return

    if problem_type == "Clustering":
        show_clustering_results()
    elif problem_type == "Classification":
        y_true, preds = _load_supervised()
        if y_true is not None:
            show_classification_results(y_true, preds)
    elif problem_type == "Regression":
        y_true, preds = _load_supervised()
        if y_true is not None:
            show_regression_results(y_true, preds)
    else:
        # Unknown value — defensive fallback
        st.error(
            f"Unrecognised problem type '{problem_type}'. "
            "Please retrain from Data Upload."
        )


# ── Internal helper ───────────────────────────────────────────────────────────

def _load_supervised() -> tuple:
    """Return (y_true, preds) as numpy arrays, or (None, None) if not found."""
    y_true = st.session_state.test_labels
    preds  = st.session_state.predictions
    if y_true is None or preds is None:
        st.error("Results not found — please retrain the model.")
        return None, None
    return np.array(y_true), np.array(preds)


# ── Classification ────────────────────────────────────────────────────────────

def show_classification_results(y_true: np.ndarray, preds: np.ndarray) -> None:
    """Display classification metrics and charts.

    Called by both training_page() (inline) and evaluation_page() (dedicated view).
    """
    y_true = np.array(y_true)
    preds  = np.array(preds)

    acc  = accuracy_score(y_true, preds)
    prec = precision_score(y_true, preds, average="weighted", zero_division=0)
    rec  = recall_score(y_true, preds, average="weighted", zero_division=0)
    f1   = f1_score(y_true, preds, average="weighted", zero_division=0)

    metric_row([
        ("Accuracy",            f"{acc:.4f}",  "How often the model is correct overall."),
        ("Precision",           f"{prec:.4f}", "Of all positive predictions, how many were right."),
        ("Recall",              f"{rec:.4f}",  "Of all actual positives, how many were found."),
        ("F1 Score (weighted)", f"{f1:.4f}",   "Balance between precision and recall."),
    ])

    # Warn when the test set itself is heavily imbalanced
    vc = pd.Series(y_true).value_counts()
    if len(vc) >= 2 and vc.max() / vc.min() > 5:
        st.warning(
            "⚠️ Class imbalance detected in the test set. "
            "Weighted metrics are shown — check per-class recall in the table below."
        )

    # Confusion matrix: each cell shows how many samples were classified as what
    st.markdown("#### Where did the model make mistakes?")
    classes = sorted(set(np.concatenate([y_true, preds])))
    cm = confusion_matrix(y_true, preds, labels=classes)
    st.plotly_chart(
        px.imshow(
            cm,
            x=[str(c) for c in classes],
            y=[str(c) for c in classes],
            text_auto=True,
            color_continuous_scale="Blues",
            labels={"x": "Predicted", "y": "Actual"},
            title="Confusion Matrix — rows = actual class, columns = predicted class",
        ),
        use_container_width=True,
    )

    # Per-class breakdown table
    st.markdown("#### Per-class breakdown")
    report_df = pd.DataFrame(
        classification_report(y_true, preds, output_dict=True, zero_division=0)
    ).transpose().round(4)
    st.dataframe(report_df, use_container_width=True)


# ── Regression ────────────────────────────────────────────────────────────────

def show_regression_results(y_true: np.ndarray, preds: np.ndarray) -> None:
    """Display regression metrics and charts.

    Called by both training_page() (inline) and evaluation_page() (dedicated view).
    """
    y_true = np.array(y_true)
    preds  = np.array(preds)

    r2   = r2_score(y_true, preds)
    mae  = mean_absolute_error(y_true, preds)
    rmse = np.sqrt(mean_squared_error(y_true, preds))
    nonzero = y_true != 0
    mape = (
        np.mean(np.abs((y_true[nonzero] - preds[nonzero]) / y_true[nonzero])) * 100
        if nonzero.any() else float("nan")
    )

    metric_row([
        ("R² Score", f"{r2:.4f}",
         "Closer to 1.0 = better. 0.0 = no better than always predicting the mean."),
        ("MAE",  f"{mae:.4f}",
         "Average error in the same units as your target column."),
        ("RMSE", f"{rmse:.4f}",
         "Like MAE but penalises large errors more heavily."),
        ("MAPE", f"{mape:.2f}%" if not np.isnan(mape) else "N/A",
         "Average % error — skips rows where the actual value is zero."),
    ])

    residuals = y_true - preds
    col_a, col_b = st.columns(2)

    with col_a:
        # Points on the red line = perfect predictions
        _min = float(min(y_true.min(), preds.min()))
        _max = float(max(y_true.max(), preds.max()))
        fig_av = px.scatter(
            x=y_true, y=preds, opacity=0.6,
            labels={"x": "Actual", "y": "Predicted"},
            title="Actual vs Predicted — points on the red line are perfect",
        )
        fig_av.add_trace(go.Scatter(
            x=[_min, _max], y=[_min, _max],
            mode="lines", name="Perfect fit",
            line=dict(dash="dash", color="red"),
        ))
        st.plotly_chart(fig_av, use_container_width=True)

    with col_b:
        # Random scatter around 0 = errors are unbiased
        fig_res = px.scatter(
            x=preds, y=residuals, opacity=0.6,
            labels={"x": "Predicted", "y": "Error (Actual − Predicted)"},
            title="Errors vs Predicted — random scatter near 0 is ideal",
        )
        fig_res.add_hline(y=0, line_dash="dash", line_color="red")
        st.plotly_chart(fig_res, use_container_width=True)

    # Error distribution — should be roughly bell-shaped and centred at 0
    st.markdown("#### Error distribution")
    fig_hist = px.histogram(
        x=residuals, nbins=50, marginal="box",
        labels={"x": "Error (Actual − Predicted)"},
        title="How errors are spread — centred near 0 means the model is unbiased",
    )
    fig_hist.add_vline(x=0, line_dash="dash", line_color="red")
    st.plotly_chart(fig_hist, use_container_width=True)

    mean_res = float(np.mean(residuals))
    if abs(mean_res) > 0.1 * float(np.std(residuals)):
        st.caption(
            f"ℹ️ Average error = {mean_res:.4f} — a non-zero average means the model "
            "consistently over- or under-predicts."
        )


# ── Clustering ────────────────────────────────────────────────────────────────

def show_clustering_results() -> None:
    """Display clustering metrics and charts.

    Called by both training_page() (inline) and evaluation_page() (dedicated view).
    Reads cluster labels and scaled data directly from session state.
    """
    labels = st.session_state.cluster_labels
    if labels is None:
        st.error("No cluster labels found — please retrain.")
        return

    labels = np.array(labels)
    df: pd.DataFrame = st.session_state.data.copy()
    numeric_df = df.select_dtypes(include=[np.number]).dropna(axis=1, how="all")

    if numeric_df.shape[1] < 2:
        st.warning("Not enough numeric columns to compute clustering metrics.")
        _cluster_summary_table(df, labels)
        return

    # Use stored X_scaled; recompute only if it is missing or stale
    X_scaled = st.session_state.clustering_X_scaled
    scaler   = st.session_state.clustering_scaler
    if X_scaled is None or X_scaled.shape[0] != len(df):
        try:
            X_imp    = SimpleImputer(strategy="mean").fit_transform(numeric_df)
            X_scaled = scaler.transform(X_imp)
        except Exception:
            X_scaled = numeric_df.fillna(numeric_df.mean()).values
        st.session_state.clustering_X_scaled = X_scaled

    # DBSCAN marks noise points as -1; exclude them from metric computation
    eval_mask   = labels != -1
    n_noise     = int((~eval_mask).sum())
    eval_X      = X_scaled[eval_mask]
    eval_labels = labels[eval_mask]

    if len(set(eval_labels)) < 2:
        st.error(
            "Fewer than 2 distinct clusters found. "
            "Try a different search scope or dataset."
        )
        return

    try:
        sil = silhouette_score(eval_X, eval_labels)
        ch  = calinski_harabasz_score(eval_X, eval_labels)
        db  = davies_bouldin_score(eval_X, eval_labels)
    except Exception as e:
        st.error(f"Could not compute clustering metrics: {e}")
        return

    metric_row([
        ("Silhouette Score",  f"{sil:.4f}",
         "Range −1 to 1. Higher = clusters are well-separated and compact."),
        ("Calinski-Harabász", f"{ch:.2f}",
         "Higher = clusters are denser and better separated."),
        ("Davies-Bouldin",    f"{db:.4f}",
         "Lower = clusters are more distinct from each other."),
    ])

    if n_noise:
        st.caption(
            f"ℹ️ {n_noise} point(s) labelled as noise by DBSCAN — "
            "excluded from metric computation but shown in the charts below."
        )

    # Cluster size bar chart
    st.markdown("#### How many points are in each cluster?")
    vc = pd.Series(labels).value_counts().sort_index()
    label_names = ["Noise" if i == -1 else f"Cluster {i}" for i in vc.index]
    st.plotly_chart(
        px.bar(
            x=label_names, y=vc.values, color=label_names,
            labels={"x": "Cluster", "y": "Number of points"},
            title="Points per cluster",
        ),
        use_container_width=True,
    )

    # PCA scatter — compressed 2D view of how clusters are separated
    if X_scaled.shape[1] >= 2:
        cluster_names = ["Noise" if l == -1 else f"Cluster {l}" for l in labels]
        fig, caption = pca_scatter_fig(
            X_scaled, labels=cluster_names,
            label_col="Cluster",
            title="Cluster assignments in 2D (all features compressed into 2 axes)",
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(caption)

    _cluster_summary_table(df, labels)


def _cluster_summary_table(df: pd.DataFrame, labels: np.ndarray) -> None:
    """Show per-cluster average values and a sample of the labelled rows."""
    df_show = df.copy()
    df_show.insert(0, "Cluster", ["Noise" if l == -1 else f"Cluster {l}" for l in labels])

    st.markdown("#### Sample rows with cluster labels (first 100)")
    st.dataframe(df_show.head(100), use_container_width=True)

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if num_cols:
        st.markdown("#### Average values per cluster")
        summary = (
            df_show[["Cluster"] + num_cols]
            .groupby("Cluster")
            .agg(["mean", "std"])
        )
        summary.columns = [" ".join(c).strip() for c in summary.columns]
        st.dataframe(summary.round(3), use_container_width=True)
