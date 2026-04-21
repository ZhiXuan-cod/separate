import streamlit as st

from utils.auth import authenticate_user, register_user
from utils.helpers import asset, set_bg_image_local


def login_page(go_to) -> None:
    set_bg_image_local(asset("login.jpg"))

    st.markdown(
        """
        <style>
        .stApp { 
            display: flex; 
            align-items: center; 
            justify-content: center; 
        }

        .stTabs [data-baseweb="tab-list"] button {
            color: white;
            font-size: 1.1rem;
        }

        .stTextInput input {
            color: #1f2937 !important;
            background-color: rgba(255,255,255,0.96) !important;
            border-radius: 10px !important;
            border: 1px solid rgba(148, 163, 184, 0.45) !important;
        }

        .stTextInput label { 
            color: white !important; 
        }

        /* 主按钮：蓝灰色，更协调 */
        button[data-testid="stFormSubmitButton"],
        .stForm button {
            background: linear-gradient(135deg, #A9D6FF, #F6E7A1) !important;
            color: black !important;
            font-weight: 700 !important;
            border: none !important;
            border-radius: 50px !important;
            padding: 14px 30px !important;
            font-size: 1.05rem !important;
            box-shadow: 0 6px 18px rgba(63, 95, 125, 0.35) !important;
            width: 100% !important;
            transition: all 0.2s ease !important;
        }

        button[data-testid="stFormSubmitButton"]:hover,
        .stForm button:hover {
            background: linear-gradient(135deg, #5b7a98, #496987) !important;
            transform: translateY(-1px) !important;
        }

        .stTextInput input:focus {
            border-color: #6f8fb1 !important;
            box-shadow: 0 0 0 0.15rem rgba(111, 143, 177, 0.18) !important;
        }

        /* 密码眼睛按钮：只轻量修饰，不隐藏 */
        div[data-testid="stTextInput"] button[title*="password"],
        div[data-testid="stTextInput"] button[aria-label*="password"] {
            background: transparent !important;
            border: none !important;
            box-shadow: none !important;
            width: auto !important;
            height: auto !important;
            margin: 0 !important;
            padding: 0 8px !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
        }

        div[data-testid="stTextInput"] button[title*="password"] svg,
        div[data-testid="stTextInput"] button[aria-label*="password"] svg {
            width: 18px !important;
            height: 18px !important;
            color: #5b6472 !important;
            fill: none !important;
            stroke: currentColor !important;
            opacity: 1 !important;
            visibility: visible !important;
        }

        div[data-testid="stTextInput"] button[title*="password"]:hover,
        div[data-testid="stTextInput"] button[aria-label*="password"]:hover {
            background: transparent !important;
        }

        /* 只隐藏提示文字，不隐藏眼睛按钮 */
        [data-testid="InputInstructions"] {
            display: none !important;
        }

        /* Back to Home 按钮 */
        .back-button-container button {
            background: transparent !important;
            color: white !important;
            border: 1px solid rgba(255,255,255,0.5) !important;
            border-radius: 50px !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    _, col, _ = st.columns([1, 2, 1])
    with col:
        st.markdown('<div class="form-card">', unsafe_allow_html=True)
        st.markdown(
            "<h2 style='color:white;text-align:center;margin-bottom:1.5rem;'>"
            "Login / Register</h2>",
            unsafe_allow_html=True,
        )

        tab_login, tab_register = st.tabs(["Login", "Register"])

        # ── Login ─────────────────────────────────────────────────────────────
        with tab_login:
            with st.form("login_form"):
                email = st.text_input("Email")
                password = st.text_input("Password", type="password")
                submitted = st.form_submit_button(
                    "Login", use_container_width=True,
                )

            if submitted:
                if not email or not password:
                    st.error("Please fill in both fields.")
                else:
                    success, name, email_ret = authenticate_user(email, password)
                    if success:
                        st.session_state.logged_in = True
                        st.session_state.user_name = name
                        st.session_state.user_email = email_ret
                        go_to("data_upload")
                    else:
                        st.error("Invalid email or password.")

        # ── Register ──────────────────────────────────────────────────────────
        with tab_register:
            with st.form("register_form"):
                name = st.text_input("Full Name")
                email = st.text_input("Email")
                password = st.text_input("Password", type="password")
                confirm = st.text_input("Confirm Password", type="password")
                submitted = st.form_submit_button(
                    "Register", use_container_width=True,
                )

            if submitted:
                if not name or not email or not password:
                    st.error("Please fill in all fields.")
                elif password != confirm:
                    st.error("Passwords do not match.")
                elif len(password) < 6:
                    st.error("Password must be at least 6 characters.")
                else:
                    success, msg = register_user(email, password, name)
                    if success:
                        st.success(msg)
                    else:
                        st.error(msg)

        st.markdown('<div class="back-button-container">', unsafe_allow_html=True)
        if st.button("← Back to Home", key="back_home"):
            go_to("front")
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)