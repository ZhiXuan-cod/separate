# components/training.py
import numpy as np
import pandas as pd
import streamlit as st

from utils.ml_utils import (
    PYCARET_AVAILABLE,
    _pycaret_setup_safe,
    auto_clustering,
    train_fallback_model,
)

# Shared result-display functions — training and evaluation pages use the
# same functions so charts are always identical in both places.
from components.evaluation import (
    show_classification_results,
    show_clustering_results,
    show_regression_results,
)

if PYCARET_AVAILABLE:
    from utils.ml_utils import (
        clf_compare, clf_predict, clf_setup,
        reg_compare, reg_predict, reg_setup,
    )


def training_page() -> None:
    """AutoML training — runs the model and immediately shows full results."""
    if st.session_state.data is None:
        st.warning("⚠️ Please upload data first.")
        return

    problem_type = st.session_state.problem_type

    # Guard: problem type must be set before training can begin
    if problem_type is None:
        st.info("Please complete **Data Upload** first — select a problem type and target column.")
        return

    if problem_type == "Clustering":
        _run_clustering()
    else:
        _run_supervised()


# ── Clustering ────────────────────────────────────────────────────────────────

def _run_clustering() -> None:
    st.markdown(
        '<h2 class="sub-header">🎯 Automated Clustering</h2>',
        unsafe_allow_html=True,
    )

    df: pd.DataFrame = st.session_state.data.copy()
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] < 2:
        st.error(f"Need at least 2 numeric features ({numeric_df.shape[1]} found).")
        return

    # Let the user choose speed vs thoroughness
    cluster_mode = st.radio(
        "How thorough should the search be?",
        options=[
            "Fast — KMeans only (a few seconds)",
            "Standard — KMeans + Hierarchical (~30 s)",
            "Full — all algorithms (several minutes)",
        ],
        index=1,
    )
    st.caption(
        f"Dataset: {df.shape[0]:,} rows · {numeric_df.shape[1]} numeric features. "
        "The algorithm automatically picks the best number of clusters."
    )

    if not st.button("🚀 Start Clustering", type="primary"):
        return

    if cluster_mode.startswith("Fast"):
        max_k, skip_hier, skip_birch, skip_dbscan = 5, True, True, True
    elif cluster_mode.startswith("Standard"):
        max_k, skip_hier, skip_birch, skip_dbscan = 8, False, True, True
    else:
        max_k, skip_hier, skip_birch, skip_dbscan = 10, False, False, False

    progress = st.progress(0, text="Starting…")
    status   = st.empty()
    try:
        status.info("Preparing and scaling features…")
        progress.progress(0.1)
        status.info("Testing different clustering configurations…")
        progress.progress(0.2)

        model, labels, algo_name, score, metrics, scaler, X_scaled = auto_clustering(
            df, max_clusters=max_k,
            skip_hierarchical=skip_hier, skip_birch=skip_birch, skip_dbscan=skip_dbscan,
        )

        progress.progress(1.0)
        status.empty()
        progress.empty()

        # Persist results to session state
        st.session_state.update({
            "cluster_labels":      labels,
            "clustering_model":    model,
            "clustering_scaler":   scaler,
            "clustering_X_scaled": X_scaled,
            "training_complete":   True,
            "problem_type":        "Clustering",
            "cluster_metrics": {
                "algorithm":         algo_name,
                "num_clusters":      metrics["num_clusters"],
                "silhouette_score":  metrics["silhouette"],
                "calinski_harabasz": metrics["calinski_harabasz"],
                "davies_bouldin":    metrics["davies_bouldin"],
                "cluster_sizes":     metrics["cluster_sizes"],
            },
        })

        st.success(
            f"🎉 Done! Best algorithm: **{algo_name}** · "
            f"{metrics['num_clusters']} clusters found · "
            f"Silhouette score: {score:.4f}"
        )

        # Show full results immediately — users can also revisit in Model Evaluation
        st.markdown("---")
        st.markdown("### 📊 Results")
        show_clustering_results()

    except Exception as exc:
        progress.empty()
        status.empty()
        st.error(f"Clustering failed: {exc}")
        st.exception(exc)


# ── Supervised (Classification / Regression) ──────────────────────────────────

def _run_supervised() -> None:
    if st.session_state.target_column is None:
        st.warning("⚠️ Please set a target column in Data Upload first.")
        return

    st.markdown(
        '<h2 class="sub-header">📐 Automated Model Training</h2>',
        unsafe_allow_html=True,
    )

    df: pd.DataFrame  = st.session_state.data.copy()
    target_col: str   = st.session_state.target_column
    problem_type: str = st.session_state.problem_type

    if target_col not in df.columns:
        st.error(f"Target column '{target_col}' not found in the dataset.")
        return

    # Drop rows where the target is missing — cannot train on unknown labels
    n_missing = int(df[target_col].isnull().sum())
    if n_missing:
        st.warning(f"Dropping {n_missing} row(s) with a missing target value.")
        df = df.dropna(subset=[target_col])
        if len(df) < 10:
            st.error("Not enough rows remain after dropping missing target values.")
            return

    if problem_type == "Regression" and not pd.api.types.is_numeric_dtype(df[target_col]):
        st.error(f"'{target_col}' must be numeric for Regression.")
        return

    n_classes = df[target_col].nunique() if problem_type == "Classification" else None

    # Configuration summary so the user knows what is about to run
    st.info(
        f"**Task:** {problem_type}  ·  **Target:** `{target_col}`  ·  "
        f"**Rows:** {len(df):,}  ·  **Features:** {len(df.columns) - 1}"
        + (f"  ·  **Classes:** {n_classes}" if n_classes else "")
        + "  ·  **CV folds:** 3  ·  **Models tried:** LR, RandomForest, LightGBM"
    )

    if not PYCARET_AVAILABLE:
        st.info("PyCaret not installed — will use scikit-learn RandomForest as fallback.")

    if not st.button("🚀 Start Training", type="primary"):
        return

    with st.spinner("Training… this may take a minute."):
        try:
            trained = False
            model = preds = y_true = None

            if PYCARET_AVAILABLE:
                try:
                    sort_metric    = "Accuracy" if problem_type == "Classification" else "R2"
                    include_models = ["lr", "rf", "lightgbm"]
                    setup_args = dict(
                        data=df, target=target_col, train_size=0.8,
                        session_id=42, verbose=False, log_experiment=False,
                        n_jobs=-1, html=False, preprocess=True,
                    )

                    if problem_type == "Classification":
                        _pycaret_setup_safe(clf_setup, **setup_args)
                        best = clf_compare(
                            verbose=False, sort=sort_metric,
                            include=include_models, n_select=1, fold=3,
                        )
                        best    = best[0] if isinstance(best, list) else best
                        pred_df = clf_predict(best)
                        preds   = pred_df["prediction_label"].values
                        y_true  = pred_df[target_col].values
                    else:
                        _pycaret_setup_safe(reg_setup, **setup_args)
                        best = reg_compare(
                            verbose=False, sort=sort_metric,
                            include=include_models, n_select=1, fold=3,
                        )
                        best    = best[0] if isinstance(best, list) else best
                        pred_df = reg_predict(best)
                        preds   = pred_df["prediction_label"].values
                        y_true  = pred_df[target_col].values

                    model   = best
                    trained = True

                except Exception as pycaret_err:
                    # PyCaret failed — proceed to RandomForest fallback
                    st.warning(
                        f"PyCaret could not complete training ({pycaret_err}). "
                        "Falling back to scikit-learn RandomForest."
                    )

            if not trained:
                model, preds, y_true = train_fallback_model(df, target_col, problem_type)

            # Convert once to numpy arrays and store
            preds_arr  = np.array(preds)
            y_true_arr = np.array(y_true)

            st.session_state.update({
                "model":             model,
                "predictions":       preds_arr,
                "test_labels":       y_true_arr,
                "training_complete": True,
            })

            algo_label = "PyCaret AutoML" if trained else "scikit-learn RandomForest"
            st.success(f"🎉 Training complete using **{algo_label}**!")

            # Show full results immediately — users can also revisit in Model Evaluation
            st.markdown("---")
            st.markdown("### 📊 Results")
            if problem_type == "Classification":
                show_classification_results(y_true_arr, preds_arr)
            elif problem_type == "Regression":
                show_regression_results(y_true_arr, preds_arr)

        except Exception as exc:
            st.error(f"Training failed: {exc}")
            st.exception(exc)
