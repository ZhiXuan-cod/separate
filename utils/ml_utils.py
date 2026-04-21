# utils/ml_utils.py
import inspect
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, AgglomerativeClustering, Birch, KMeans
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ── PyCaret (optional) — single import point for the whole app ────────────────
PYCARET_AVAILABLE = False
try:
    from pycaret.classification import compare_models as clf_compare
    from pycaret.classification import get_config as clf_get_config
    from pycaret.classification import predict_model as clf_predict
    from pycaret.classification import setup as clf_setup
    from pycaret.regression import compare_models as reg_compare
    from pycaret.regression import get_config as reg_get_config
    from pycaret.regression import predict_model as reg_predict
    from pycaret.regression import setup as reg_setup
    PYCARET_AVAILABLE = True
except ImportError:
    pass


def _pycaret_setup_safe(setup_fn, **kwargs):
    """
    Call a PyCaret setup function, silently dropping keyword arguments that
    the installed version does not accept (handles API differences across
    PyCaret versions).
    """
    sig = inspect.signature(setup_fn)
    filtered = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return setup_fn(**filtered)


# ── Task-detection helpers ────────────────────────────────────────────────────

def is_classification_possible(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Detect columns that could serve as a classification target.
    Criteria: object/category dtype, OR numeric with fewer than 20 unique values.
    """
    candidates = []
    for col in df.columns:
        dtype = df[col].dtype
        unique_vals = df[col].nunique(dropna=False)
        if dtype in ["object", "category"]:
            candidates.append(col)
        elif np.issubdtype(dtype, np.number) and unique_vals < 20:
            candidates.append(col)
    return bool(candidates), candidates


def is_regression_possible(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Detect columns that could serve as a regression target.
    Criteria: numeric dtype with at least 20 unique values.
    """
    candidates = [
        col for col in df.columns
        if np.issubdtype(df[col].dtype, np.number)
        and df[col].nunique(dropna=False) >= 20
    ]
    return bool(candidates), candidates


def is_clustering_possible(
    df: pd.DataFrame,
    min_rows: int = 10,
    min_numeric_features: int = 2,
) -> Tuple[bool, str]:
    """
    Check whether the dataset has enough rows and numeric features for
    clustering. Returns (is_possible, reason_message).
    """
    if len(df) < min_rows:
        return False, f"Need at least {min_rows} rows (currently {len(df)})"

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) < min_numeric_features:
        return (
            False,
            f"Need at least {min_numeric_features} numeric features "
            f"(currently {len(numeric_cols)})",
        )

    constant_cols = [col for col in numeric_cols if df[col].var() == 0]
    if constant_cols:
        return (
            False,
            f"Constant (zero-variance) numeric features: "
            f"{', '.join(constant_cols[:3])}",
        )

    return True, "Suitable for clustering"


# ── Auto-clustering ───────────────────────────────────────────────────────────

def auto_clustering(
    df: pd.DataFrame,
    max_clusters: int = 10,
    skip_hierarchical: bool = False,
    skip_birch: bool = False,
    skip_dbscan: bool = False,
):
    """
    Try multiple clustering algorithms and hyperparameters; select the best
    by silhouette score.

    Uses Calinski-Harabász (O(n), cheap) as a fast pre-filter before calling
    silhouette_score (O(n²), expensive) to avoid redundant heavy computation.

    Returns
    -------
    (best_model, best_labels, best_name, best_score, metrics, scaler, X_scaled)
    """
    numeric_df = df.select_dtypes(include=[np.number])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(numeric_df)

    best_score = -1.0
    best_ch = 0.0          # best Calinski-Harabász seen so far (cheap pre-filter)
    best_model = None
    best_labels = None
    best_name = ""

    def _try_update(model, labels, name: str) -> None:
        """Evaluate a candidate and update bests if it improves silhouette."""
        nonlocal best_score, best_ch, best_model, best_labels, best_name

        if len(set(labels)) < 2:
            return

        # Fast pre-filter: skip expensive silhouette when CH is clearly worse.
        ch = calinski_harabasz_score(X_scaled, labels)
        if best_model is not None and ch < best_ch * 0.7:
            return

        score = silhouette_score(X_scaled, labels)
        if score > best_score:
            best_score = score
            best_ch = ch
            best_model = model
            best_labels = labels
            best_name = name

    # 1. KMeans (always run)
    for k in range(2, min(max_clusters, len(df) - 1) + 1):
        try:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            _try_update(km, km.fit_predict(X_scaled), f"KMeans (k={k})")
        except Exception:
            continue

    # 2. Agglomerative / Hierarchical
    if not skip_hierarchical:
        for linkage in ["ward", "complete", "average"]:
            for k in range(2, min(max_clusters, len(df) - 1) + 1):
                try:
                    hc = AgglomerativeClustering(n_clusters=k, linkage=linkage)
                    _try_update(
                        hc,
                        hc.fit_predict(X_scaled),
                        f"Agglomerative (k={k}, linkage={linkage})",
                    )
                except Exception:
                    continue

    # 3. BIRCH
    if not skip_birch:
        for k in range(2, min(max_clusters, len(df) - 1) + 1):
            try:
                birch = Birch(n_clusters=k)
                _try_update(birch, birch.fit_predict(X_scaled), f"BIRCH (k={k})")
            except Exception:
                continue

    # 4. DBSCAN
    if not skip_dbscan:
        for eps in np.linspace(0.1, 1.5, 10):
            try:
                db = DBSCAN(eps=eps, min_samples=5)
                labels = db.fit_predict(X_scaled)
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                if n_clusters >= 2:
                    mask = labels != -1
                    if mask.sum() >= 2:
                        _try_update(
                            db,
                            labels,
                            f"DBSCAN (eps={eps:.2f})",
                        )
            except Exception:
                continue

    # Fallback: KMeans k=2
    if best_model is None:
        best_model = KMeans(n_clusters=2, random_state=42, n_init=10)
        best_labels = best_model.fit_predict(X_scaled)
        best_name = "KMeans (k=2, fallback)"
        best_score = (
            silhouette_score(X_scaled, best_labels)
            if len(set(best_labels)) > 1 else 0.0
        )

    n_unique = len(set(best_labels))
    metrics = {
        "silhouette": best_score,
        "calinski_harabasz": (
            calinski_harabasz_score(X_scaled, best_labels) if n_unique > 1 else 0.0
        ),
        "davies_bouldin": (
            davies_bouldin_score(X_scaled, best_labels) if n_unique > 1 else 0.0
        ),
        "num_clusters": n_unique,
        "algorithm": best_name,
        "cluster_sizes": pd.Series(best_labels).value_counts().to_dict(),
    }

    return best_model, best_labels, best_name, best_score, metrics, scaler, X_scaled


# ── Fallback supervised training ──────────────────────────────────────────────

def train_fallback_model(
    df: pd.DataFrame, target_col: str, problem_type: str
):
    """
    Train a RandomForest model when PyCaret is unavailable.
    Returns (model, predictions, y_true_array).
    """
    X = pd.get_dummies(df.drop(columns=[target_col]), drop_first=True)
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    if problem_type == "Classification":
        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    else:
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)

    model.fit(X_train, y_train)
    return model, model.predict(X_test), y_test.values
