# components/dashboard.py
import streamlit as st

from utils.helpers import asset, set_bg_image_local
from utils.ml_utils import PYCARET_AVAILABLE
from utils.state import reset_ml_state

from components.account import account_page
from components.eda import eda_page
from components.evaluation import evaluation_page
from components.export import export_page
from components.training import training_page
from components.upload import upload_page

# ── Workflow definition ───────────────────────────────────────────────────────
WORKFLOW_PAGES = [
    "data_upload",
    "eda",
    "model_training",
    "model_evaluation",
    "export_results",
    "account",
]

PAGE_LABELS = {
    "data_upload":      "📁 Data Upload",
    "eda":              "🔍 Exploratory Data Analysis",
    "model_training":   "📐 AutoML Training",
    "model_evaluation": "📈 Model Evaluation",
    "export_results":   "💾 Export Results",
    "account":          "👤 Account Settings",
}

# Map each page key to the function that renders it.
_PAGE_RENDERERS = {
    "data_upload":      upload_page,
    "eda":              eda_page,
    "model_training":   training_page,
    "model_evaluation": evaluation_page,
    "account":          account_page,
}


def dashboard_page(go_to) -> None:
    """
    Main dashboard shell: sidebar navigation + active-page renderer.
    All workflow sub-page routing is handled here so app.py stays thin.
    """
    set_bg_image_local(asset("purple.png"))

    st.markdown(
        f"<h1 style='color: black;'>Welcome, {st.session_state.user_name}!</h1>",
        unsafe_allow_html=True,
    )

    # Ensure the current page is a valid workflow page
    current_page = st.session_state.page
    if current_page not in WORKFLOW_PAGES:
        current_page = "data_upload"
        st.session_state.page = current_page

    with st.sidebar:
        st.image(
            "https://cdn-icons-png.flaticon.com/512/2103/2103832.png", width=100
        )
        st.markdown("---")
        st.markdown("### Sequential Steps")

        current_index = WORKFLOW_PAGES.index(current_page)
        selected_label = st.radio(
            "Select a step:",
            options=[PAGE_LABELS[p] for p in WORKFLOW_PAGES],
            index=current_index,
            key="sidebar_radio",
            label_visibility="collapsed",
        )

        # Reverse-lookup: label → page key
        selected_page = next(
            p for p, lbl in PAGE_LABELS.items() if lbl == selected_label
        )
        if selected_page != st.session_state.page:
            go_to(selected_page)

        st.markdown("---")

        if not PYCARET_AVAILABLE:
            st.error(
                "⚠️ PyCaret not installed. "
                "Run `pip install pycaret` for full AutoML."
            )

        if st.button("👋🏻 Logout", type="primary", use_container_width=True):
            st.session_state.logged_in = False
            st.session_state.user_name = ""
            st.session_state.user_email = ""
            reset_ml_state()
            go_to("front")

    # ── Render the active page ────────────────────────────────────────────────
    page = st.session_state.page

    if page == "export_results":
        # export_page needs go_to to navigate after "Start Over"
        export_page(go_to)
    elif page in _PAGE_RENDERERS:
        _PAGE_RENDERERS[page]()
    else:
        # Unknown sub-page — fall back to upload
        st.session_state.page = "data_upload"
        upload_page()
