import streamlit as st

# ── Page config MUST be the very first Streamlit call ────────────────────────
st.set_page_config(
    page_title="No-Code ML Platform",
    page_icon="💻",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Internal imports (after set_page_config) ──────────────────────────────────
from utils.db import init_supabase
from utils.helpers import set_global_css
from utils.state import init_ml_state

from components.dashboard import dashboard_page
from components.front import front_page
from components.login import login_page

# ── Bootstrap session state ───────────────────────────────────────────────────
for key, default in [
    ("page", "front"),
    ("logged_in", False),
    ("user_name", ""),
    ("user_email", ""),
]:
    if key not in st.session_state:
        st.session_state[key] = default

init_ml_state()

# ── Infrastructure ────────────────────────────────────────────────────────────
init_supabase()
set_global_css()


# ── Navigation helper ─────────────────────────────────────────────────────────
def go_to(page: str) -> None:
    """Set the active page and trigger a rerun."""
    if st.session_state.page != page:
        st.session_state.page = page
        st.rerun()


# ── Routing ───────────────────────────────────────────────────────────────────
# app.py only cares about authentication state — all page-level logic lives
# inside the component modules.
page = st.session_state.page

if page == "front":
    front_page(go_to)

elif page == "login":
    login_page(go_to)

elif st.session_state.logged_in:
    # Normalise the legacy "dashboard" alias to the first workflow step.
    if page == "dashboard":
        st.session_state.page = "data_upload"
    dashboard_page(go_to)

else:
    # Any protected page reached while logged out → send to login.
    go_to("login")
