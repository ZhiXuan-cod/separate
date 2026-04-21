# components/upload.py
import hashlib
import io

import pandas as pd
import streamlit as st

from utils.ml_utils import (
    is_classification_possible,
    is_clustering_possible,
    is_regression_possible,
)
from utils.state import reset_ml_state


@st.cache_data(show_spinner=False)
def _load_csv(raw_bytes: bytes) -> pd.DataFrame | None:
    """Parse CSV bytes trying multiple encodings. Cached by content hash."""
    for enc in ["utf-8", "latin1", "iso-8859-1", "cp1252"]:
        try:
            return pd.read_csv(io.BytesIO(raw_bytes), encoding=enc)
        except UnicodeDecodeError:
            continue
        except Exception:
            break
    return None


def upload_page() -> None:
    st.markdown('<h2 class="sub-header">📁 Upload Your Dataset</h2>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file is not None:
        raw     = uploaded_file.read()
        file_id = hashlib.md5(raw).hexdigest()

        if st.session_state.get("_last_file_id") != file_id:
            df = _load_csv(raw)
            if df is None:
                st.error("Could not parse the CSV — please check its encoding.")
                return
            if df.empty:
                st.error("The uploaded file is empty.")
                return
            reset_ml_state()
            st.session_state["_last_file_id"] = file_id
            st.session_state.data = df
            st.success(f"✔️ Loaded **{len(df):,} rows × {len(df.columns)} columns**")

    if st.session_state.data is None:
        st.info("No data loaded yet — upload a CSV file above.")
        return

    df: pd.DataFrame      = st.session_state.data
    current_file_id: str  = st.session_state.get("_last_file_id", "")

    # Data quality notices
    total_missing = df.isnull().sum().sum()
    if total_missing > 0:
        pct = total_missing / df.size * 100
        st.info(f"ℹ️ **{int(total_missing):,}** missing value(s) ({pct:.1f}%) — handled automatically.")

    fully_null_cols = df.columns[df.isnull().all()].tolist()
    if fully_null_cols:
        st.warning(f"⚠️ Entirely empty columns (will be ignored): {', '.join(f'`{c}`' for c in fully_null_cols)}")

    class_ok, class_candidates = is_classification_possible(df)
    reg_ok,   reg_candidates   = is_regression_possible(df)
    clust_ok, clust_msg        = is_clustering_possible(df)

    available_tasks = (
        (["Classification"] if class_ok else [])
        + (["Regression"]   if reg_ok   else [])
        + (["Clustering"]   if clust_ok else [])
    )

    if not available_tasks:
        st.error("No ML task is possible with this dataset — please try a different file.")
        return

    # ── Task & target selection ───────────────────────────────────────────────
    st.markdown("### 📌 Configure Your Task")

    col_a, col_b = st.columns(2)
    with col_a:
        st.caption("**Classification candidates:** " + (", ".join(class_candidates) or "none"))
    with col_b:
        st.caption("**Regression candidates:** "     + (", ".join(reg_candidates)   or "none"))

    current_problem = st.session_state.get("problem_type")
    default_idx     = available_tasks.index(current_problem) if current_problem in available_tasks else 0
    problem_type    = st.selectbox("Problem type:", available_tasks, index=default_idx)

    if problem_type == "Clustering":
        if not clust_ok:
            st.error(f"Clustering unavailable: {clust_msg}")
        else:
            st.caption("Unsupervised — no target column needed. The algorithm finds groups automatically.")
            if st.button("Set Clustering Task", type="primary", key="set_clustering"):
                _reset_and_restore(df, current_file_id)
                st.session_state.target_column = None
                st.session_state.problem_type  = "Clustering"
                st.success("✅ Clustering task set.")
    else:
        candidates = class_candidates if problem_type == "Classification" else reg_candidates
        if not candidates:
            st.error(f"No suitable target column found for {problem_type}.")
            return

        current_target     = st.session_state.get("target_column")
        target_default_idx = candidates.index(current_target) if current_target in candidates else 0
        target_col         = st.selectbox(
            f"Which column do you want to predict? ({problem_type}):",
            candidates, index=target_default_idx,
        )

        with st.expander(f"Preview `{target_col}`"):
            col_info = df[target_col]
            ca, cb, cc = st.columns(3)
            ca.metric("Unique values", col_info.nunique())
            cb.metric("Missing",       int(col_info.isna().sum()))
            cc.metric("Type",          str(col_info.dtype))
            if pd.api.types.is_numeric_dtype(col_info):
                st.dataframe(col_info.describe().to_frame(), use_container_width=True)
            else:
                st.dataframe(col_info.value_counts().head(10).to_frame(), use_container_width=True)

        if st.button("Set Target", type="primary", key="set_target"):
            if problem_type == "Classification" and df[target_col].nunique() > 50:
                st.warning(
                    f"⚠️ '{target_col}' has {df[target_col].nunique()} unique values — "
                    "consider Regression instead."
                )
            if problem_type == "Regression" and not pd.api.types.is_numeric_dtype(df[target_col]):
                st.error(f"'{target_col}' is not numeric — Regression requires a numeric target.")
                return

            _reset_and_restore(df, current_file_id)
            st.session_state.target_column = target_col
            st.session_state.problem_type  = problem_type
            st.success(f"✅ Target set to **{target_col}** ({problem_type}).")

    # ── Data preview ──────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📄 Data Preview")
    st.dataframe(df.head(10), use_container_width=True)

    with st.expander("📊 Column Statistics"):
        st.write(f"**Shape:** {df.shape}")
        summary = pd.DataFrame({
            "Column":    df.columns,
            "Type":      df.dtypes.astype(str).values,
            "Non-null":  df.count().values,
            "Missing":   df.isnull().sum().values,
            "Missing %": (df.isnull().sum().values / len(df) * 100).round(1),
            "Unique":    df.nunique().values,
        })
        st.dataframe(summary, use_container_width=True)


def _reset_and_restore(df: pd.DataFrame, file_id: str) -> None:
    reset_ml_state()
    st.session_state.data             = df
    st.session_state["_last_file_id"] = file_id
