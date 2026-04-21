# utils/db.py
import streamlit as st

try:
    from supabase import create_client
    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False


def init_supabase() -> None:
    """
    Initialise the Supabase client and store it in session state.
    Shows a clear, actionable error for every failure mode.
    """
    if not SUPABASE_AVAILABLE:
        st.session_state.supabase = None
        st.warning(
            "⚠️ Supabase not installed. Authentication will not work. "
            "Run: `pip install supabase`"
        )
        return

    if "supabase" in st.session_state:
        return  # already initialised — skip

    try:
        url = st.secrets["supabase"]["url"]
        key = st.secrets["supabase"]["key"]
    except KeyError:
        st.error(
            "Supabase secrets not configured. "
            "Add `[supabase]` with `url` and `key` to `.streamlit/secrets.toml`."
        )
        st.session_state.supabase = None
        return

    try:
        st.session_state.supabase = create_client(url, key)
    except Exception as exc:
        st.error(f"Supabase connection failed: {exc}")
        st.session_state.supabase = None
