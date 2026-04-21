# components/front.py
import base64
import os

import streamlit as st

from utils.helpers import asset, set_bg_image_local


def front_page(go_to) -> None:
    set_bg_image_local(asset("FrontPage.jpg"))

    st.markdown(
        """
        <style>
        .stApp { color: white !important; }
        .stApp * { color: white !important; }
        div.stButton > button { color: white !important; }

        .right-panel {
            background-color: rgba(0,0,0,0.70);
            padding: 3rem 2rem;
            border-radius: 20px;
            text-align: center;
            box-shadow: 0 4px 20px rgba(0,0,0,0.5);
            animation: fadeIn 1s ease-in-out;
            color: white;
            height: 100%;
        }
        .right-panel h1 {
            text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
            font-size: 3rem;
            margin-bottom: 1rem;
        }
        .right-panel p {
            text-shadow: 1px 1px 2px rgba(0,0,0,0.5);
            font-size: 1.2rem;
            opacity: 0.9;
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to   { opacity: 1; transform: translateY(0); }
        }
        .video-container {
            display: flex;
            align-items: center;
            justify-content: center;
            max-height: 400px;
            height: auto;
            margin: auto;
        }
        .video-container video {
            width: 100%;
            height: auto;
            max-height: 400px;
            object-fit: contain;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([1.2, 1.8])

    with col1:
        video_path = asset("animation.mp4")
        if os.path.exists(video_path):
            with open(video_path, "rb") as f:
                video_b64 = base64.b64encode(f.read()).decode()
            st.markdown(
                f"""
                <div class="video-container">
                    <video autoplay loop muted playsinline>
                        <source src="data:video/mp4;base64,{video_b64}"
                                type="video/mp4">
                    </video>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div style="background:rgba(255,255,255,0.2);border-radius:10px;'
                'padding:2rem;text-align:center;">'
                '<span style="font-size:3rem;">📹</span>'
                '<p style="color:white;">Video not found. Please add animation.mp4</p>'
                "</div>",
                unsafe_allow_html=True,
            )

    with col2:
        st.markdown(
            '<div class="right-panel">'
            "<h1>Welcome to<br>No-Code ML Platform</h1>"
            "<p>Accessible Machine Learning without code.</p>"
            "</div>",
            unsafe_allow_html=True,
        )
        if st.button("Get Started", key="get_started", use_container_width=True):
            go_to("login")
