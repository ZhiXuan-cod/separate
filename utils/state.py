# utils/state.py
import streamlit as st

_ML_DEFAULTS = {
    "data": None,
    "target_column": None,
    "problem_type": None,
    "model": None,
    "predictions": None,
    "test_labels": None,
    "training_complete": False,
    "cluster_labels": None,
    "cluster_metrics": None,
    "clustering_model": None,
    "clustering_scaler": None,
    "clustering_X_scaled": None,
    "_last_file_id": None,
}


def init_ml_state() -> None:
    """Initialise ML state keys with safe defaults (call once at startup)."""
    for k, v in _ML_DEFAULTS.items():
        if k not in st.session_state:
            st.session_state[k] = v


def reset_ml_state() -> None:
    """Reset all ML state to defaults (new upload or Start Over)."""
    st.session_state.update(_ML_DEFAULTS)
