# components/eda.py
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from utils.helpers import pca_scatter_fig


def eda_page() -> None:
    """EDA page — shows only charts that help users understand their data."""
    if st.session_state.data is None:
        st.warning("⚠️ Please upload data first.")
        return

    st.markdown(
        '<h2 class="sub-header">🔍 Exploratory Data Analysis</h2>',
        unsafe_allow_html=True,
    )

    df: pd.DataFrame = st.session_state.data.copy()
    problem_type     = st.session_state.problem_type
    target_col       = st.session_state.target_column

    # Exclude the target from feature-level analysis so it doesn't skew charts
    feature_cols = (
        [c for c in df.columns if c != target_col]
        if target_col and target_col in df.columns
        else list(df.columns)
    )
    feature_df = df[feature_cols]

    # ── Overview ──────────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows",           f"{len(df):,}")
    c2.metric("Columns",        len(df.columns))
    c3.metric("Missing Values", f"{int(df.isnull().sum().sum()):,}")
    c4.metric("Memory (MB)",    f"{df.memory_usage(deep=True).sum() / 1024**2:.2f}")

    # Missing-value detail — only visible when there are missing values
    missing_per_col    = df.isnull().sum()
    cols_with_missing  = missing_per_col[missing_per_col > 0]
    if not cols_with_missing.empty:
        with st.expander(f"⚠️ {len(cols_with_missing)} column(s) have missing values"):
            miss_df = pd.DataFrame({
                "Column":    cols_with_missing.index,
                "Count":     cols_with_missing.values,
                "Missing %": (cols_with_missing.values / len(df) * 100).round(2),
            }).sort_values("Missing %", ascending=False)
            st.dataframe(miss_df, use_container_width=True)

    # ── Numerical feature analysis ────────────────────────────────────────────
    numerical_cols = feature_df.select_dtypes(include=[np.number]).columns.tolist()
    if numerical_cols:
        st.markdown("### 📊 Numerical Features")
        sel_num = st.selectbox("Select a column:", numerical_cols, key="eda_sel_num")
        if sel_num:
            col_a, col_b = st.columns(2)
            with col_a:
                # Histogram: shows overall distribution and skewness at a glance
                st.plotly_chart(
                    px.histogram(
                        df, x=sel_num, nbins=50, marginal="box",
                        title=f"Distribution — {sel_num}",
                    ),
                    use_container_width=True,
                )
            with col_b:
                # Box plot: shows median, spread, and potential outliers
                st.plotly_chart(
                    px.box(df, y=sel_num, title=f"Spread & Outliers — {sel_num}"),
                    use_container_width=True,
                )
            st.dataframe(df[sel_num].describe().to_frame(), use_container_width=True)

    # ── Categorical feature analysis ──────────────────────────────────────────
    categorical_cols = feature_df.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()
    if categorical_cols:
        st.markdown("### 📊 Categorical Features")
        sel_cat = st.selectbox("Select a column:", categorical_cols, key="eda_sel_cat")
        if sel_cat:
            vc = df[sel_cat].value_counts().head(20)
            col_a, col_b = st.columns(2)
            with col_a:
                st.plotly_chart(
                    px.bar(
                        x=vc.index.astype(str), y=vc.values,
                        labels={"x": sel_cat, "y": "Count"},
                        title=f"Category Counts — {sel_cat}",
                    ),
                    use_container_width=True,
                )
            with col_b:
                st.plotly_chart(
                    px.pie(
                        names=vc.index.astype(str), values=vc.values,
                        title=f"Proportions — {sel_cat}",
                    ),
                    use_container_width=True,
                )
            st.caption(
                f"`{sel_cat}`: {df[sel_cat].nunique()} unique values "
                f"(showing top {len(vc)})."
            )

    # ── Correlation matrix ────────────────────────────────────────────────────
    # Include the numeric target so users can immediately see which features
    # are most related to what they want to predict.
    corr_cols = feature_df.select_dtypes(include=[np.number]).columns.tolist()
    if (
        target_col
        and target_col in df.columns
        and pd.api.types.is_numeric_dtype(df[target_col])
        and target_col not in corr_cols
    ):
        corr_cols = corr_cols + [target_col]

    if len(corr_cols) > 1:
        st.markdown("### 🔗 Feature Correlation")
        corr = df[corr_cols].corr()
        st.plotly_chart(
            px.imshow(
                corr,
                x=corr.columns, y=corr.columns,
                color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                text_auto=".2f", aspect="auto",
                title=(
                    "How strongly each feature is related to the others"
                    + (" (target included)" if target_col in corr_cols else "")
                ),
            ),
            use_container_width=True,
        )
        if target_col and target_col in corr.columns:
            top = (
                corr[target_col]
                .drop(target_col)
                .abs()
                .sort_values(ascending=False)
                .head(5)
            )
            st.caption(
                f"**Features most related to `{target_col}`:** "
                + ", ".join(f"`{c}` ({v:.2f})" for c, v in top.items())
            )

    # ── Problem-specific analysis ─────────────────────────────────────────────
    if problem_type == "Classification" and target_col and target_col in df.columns:
        _eda_classification(df, target_col, numerical_cols)

    elif problem_type == "Regression" and target_col and target_col in df.columns:
        _eda_regression(df, target_col, numerical_cols)

    elif problem_type == "Clustering":
        _eda_clustering(feature_df)


# ── Task-specific sections ────────────────────────────────────────────────────

def _eda_classification(
    df: pd.DataFrame, target_col: str, numerical_cols: list
) -> None:
    """Show class distribution and how numeric features vary across classes."""
    st.markdown(f"### 🎯 Target: `{target_col}` (Classification)")
    vc = df[target_col].value_counts()

    col_a, col_b = st.columns(2)
    with col_a:
        st.plotly_chart(
            px.bar(
                x=vc.index.astype(str), y=vc.values,
                labels={"x": target_col, "y": "Count"},
                title=f"Samples per class ({len(vc)} classes)",
            ),
            use_container_width=True,
        )
    with col_b:
        st.plotly_chart(
            px.pie(
                names=vc.index.astype(str), values=vc.values,
                title="Class proportions",
            ),
            use_container_width=True,
        )

    # Warn when one class dominates — affects model fairness
    if len(vc) >= 2 and vc.max() / vc.min() > 5:
        st.warning(
            f"⚠️ Class imbalance: the largest class is {vc.max() / vc.min():.1f}× "
            "bigger than the smallest. The model may be biased towards the majority class."
        )

    # Box plots: show whether features separate the classes — useful for spotting
    # which columns will be most helpful to the model.
    if numerical_cols:
        st.markdown("#### How features vary across classes")

        # Cast the target to string so x-axis labels and legend always match,
        # regardless of whether the target is stored as int or object.
        df_plot = df.copy()
        df_plot[target_col] = df_plot[target_col].astype(str)

        for fc in numerical_cols[:3]:
            st.plotly_chart(
                px.box(
                    df_plot, x=target_col, y=fc, color=target_col,
                    title=f"`{fc}` values grouped by class",
                ),
                use_container_width=True,
            )


def _eda_regression(
    df: pd.DataFrame, target_col: str, numerical_cols: list
) -> None:
    """Show the target distribution and the features most correlated with it."""
    st.markdown(f"### 🎯 Target: `{target_col}` (Regression)")

    if not pd.api.types.is_numeric_dtype(df[target_col]):
        st.error(f"'{target_col}' is not numeric — cannot run regression analysis.")
        return

    # Target distribution: helps spot skewness or outliers before training
    st.plotly_chart(
        px.histogram(
            df, x=target_col, nbins=50, marginal="box",
            title=f"Distribution of `{target_col}` (the value you want to predict)",
        ),
        use_container_width=True,
    )
    st.dataframe(df[target_col].describe().to_frame(), use_container_width=True)

    # Scatter plots: show how top features relate to the target
    if numerical_cols:
        st.markdown("#### Features most related to the target")
        corr_with_target = (
            df[numerical_cols + [target_col]]
            .corr()[target_col]
            .drop(target_col)
            .abs()
            .sort_values(ascending=False)
        )
        for fc in corr_with_target.index[:3]:
            r = df[[fc, target_col]].corr().iloc[0, 1]
            st.plotly_chart(
                px.scatter(
                    df, x=fc, y=target_col, trendline="ols",
                    title=f"`{fc}` vs `{target_col}` — correlation: {r:.3f}",
                ),
                use_container_width=True,
            )


def _eda_clustering(feature_df: pd.DataFrame) -> None:
    """Show a 2D PCA overview so the user can see whether natural groups exist."""
    st.markdown("### 🧩 Data Structure Preview")
    num_df = feature_df.select_dtypes(include=[np.number]).dropna(axis=1, how="all")

    if num_df.shape[1] < 2:
        st.info("Need at least 2 numeric features to show a visual preview.")
        return

    # PCA compresses all features into 2 dimensions for visualisation.
    # If natural groups exist they will often already be visible here.
    X_imp    = SimpleImputer(strategy="mean").fit_transform(num_df)
    X_scaled = StandardScaler().fit_transform(X_imp)

    fig, caption = pca_scatter_fig(
        X_scaled,
        title="2D view of your data — natural groups may already be visible",
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(caption)
