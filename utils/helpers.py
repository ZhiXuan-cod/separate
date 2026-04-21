# utils/helpers.py
import base64
import os
import pathlib
from typing import List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.decomposition import PCA

ROOT = pathlib.Path(__file__).parent.parent


def asset(name: str) -> str:
    """Absolute path to a file inside the assets/ folder."""
    return str(ROOT / "assets" / name)


def get_base64_of_file(file_path: str) -> Optional[str]:
    try:
        with open(file_path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except FileNotFoundError:
        return None


def set_bg_image_local(image_path: str) -> None:
    bin_str = get_base64_of_file(image_path)
    if bin_str:
        ext = os.path.splitext(image_path)[1].lower()
        mime = "image/png" if ext == ".png" else "image/jpeg"
        st.markdown(
            f'<style>.stApp{{background-image:url("data:{mime};base64,{bin_str}");'
            "background-size:cover;background-repeat:no-repeat;"
            "background-attachment:fixed;}}</style>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            "<style>.stApp{background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);}"
            "</style>",
            unsafe_allow_html=True,
        )


def set_global_css() -> None:
    """Inject global CSS — call once from app.py after set_page_config."""
    st.markdown(
        """
        <style>
            .main-header {
                font-size: 2.5rem; color: #1E88E5;
                text-align: center; padding: 1rem; margin-bottom: 2rem;
            }
            .sub-header {
                font-size: 1.5rem; color: #3949AB;
                margin-top: 1.5rem; margin-bottom: 1rem;
            }
            .card {
                background-color: rgba(255,255,255,0.85); border-radius: 10px;
                padding: 1.5rem; margin-bottom: 1rem;
                border-left: 5px solid #1E88E5;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }

            /* Regular + download + form submit buttons — unified gradient */
            div.stButton > button,
            div.stDownloadButton > button,
            div.stFormSubmitButton > button {
                background: linear-gradient(135deg, #2196F3, #9C27B0) !important;
                color: white !important;
                border: none !important;
                padding: 0.75rem 2rem !important;
                font-size: 1.1rem !important;
                border-radius: 50px !important;
                transition: all 0.3s ease !important;
                width: 100% !important;
                margin-top: 0.5rem !important;
            }
            div.stButton > button:hover,
            div.stDownloadButton > button:hover,
            div.stFormSubmitButton > button:hover {
                transform: scale(1.02) !important;
                box-shadow: 0 4px 12px rgba(33,150,243,0.4) !important;
            }

            /* Sidebar: light background, all text black, buttons white text */
            section[data-testid="stSidebar"] {
                background: #ffffe0 !important;
            }
            section[data-testid="stSidebar"] * {
                color: black !important;
            }
            section[data-testid="stSidebar"] div.stButton > button {
                color: white !important;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ── Shared UI helpers ─────────────────────────────────────────────────────────

def metric_row(items: list) -> None:
    """Render metric cards side-by-side.

    items: list of (label, value) or (label, value, help_text).
    """
    cols = st.columns(len(items))
    for col, entry in zip(cols, items):
        col.metric(entry[0], entry[1], help=entry[2] if len(entry) > 2 else None)


def pca_scatter_fig(
    X_scaled: np.ndarray,
    labels=None,
    label_col: str = "Group",
    title: str = "PCA projection",
) -> tuple:
    """2-component PCA scatter shared by EDA, Training, and Evaluation.

    Returns (plotly_fig, caption_str).
    """
    n_components = min(2, X_scaled.shape[1])
    pca = PCA(n_components=n_components)
    pc = pca.fit_transform(X_scaled)
    col_names = ["PC1", "PC2"][:n_components]
    df = pd.DataFrame(pc, columns=col_names)
    ev = pca.explained_variance_ratio_

    if labels is not None and n_components >= 2:
        df[label_col] = [str(l) for l in labels]
        fig = px.scatter(df, x="PC1", y="PC2", color=label_col, title=title, opacity=0.7)
    elif n_components >= 2:
        fig = px.scatter(df, x="PC1", y="PC2", title=title, opacity=0.6)
    else:
        fig = px.scatter(df, x="PC1", y="PC1", title=title, opacity=0.6)

    caption = "  |  ".join(f"PC{i+1}: {v:.1%}" for i, v in enumerate(ev))
    caption += " variance explained."
    return fig, caption


# ── PDF generation ────────────────────────────────────────────────────────────

def text_to_simple_pdf_bytes(text: str, title: str = "ML Model Report") -> bytes:
    """Generate a minimal valid PDF from plain text without third-party libraries."""

    def _escape(s: str) -> str:
        s = s.encode("latin-1", errors="replace").decode("latin-1")
        return s.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")

    page_w, page_h = 612, 792
    margin_x, margin_y = 54, 54
    font_size = 10
    leading = 14
    max_lines = int((page_h - 2 * margin_y) / leading)

    lines = (text or "").splitlines() or ["(empty report)"]
    pages = [lines[i: i + max_lines] for i in range(0, len(lines), max_lines)]

    objects: List[bytes] = []

    def add_obj(obj: bytes) -> int:
        objects.append(obj)
        return len(objects)

    add_obj(b"<< /Type /Catalog /Pages 2 0 R >>")
    add_obj(b"<< /Type /Pages /Kids [] /Count 0 >>")

    page_obj_nums: List[int] = []
    for page_lines in pages:
        y0 = page_h - margin_y
        ops = [
            b"BT",
            b"/F1 %d Tf" % font_size,
            b"1 0 0 1 %d %d Tm" % (margin_x, y0),
        ]
        for i, line in enumerate(page_lines):
            if i > 0:
                ops.append(b"0 -%d Td" % leading)
            ops.append(b"(%s) Tj" % _escape(line).encode("latin-1"))
        ops.append(b"ET")
        stream = b"\n".join(ops) + b"\n"
        content_num = add_obj(
            b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"endstream"
        )
        page_dict = (
            b"<< /Type /Page /Parent 2 0 R "
            b"/MediaBox [0 0 %d %d] " % (page_w, page_h)
            + b"/Resources << /Font << /F1 << /Type /Font /Subtype /Type1 "
            b"/BaseFont /Helvetica >> >> >> "
            b"/Contents %d 0 R >>" % content_num
        )
        page_obj_nums.append(add_obj(page_dict))

    kids = b" ".join(b"%d 0 R" % n for n in page_obj_nums)
    objects[1] = b"<< /Type /Pages /Kids [ %s ] /Count %d >>" % (kids, len(page_obj_nums))
    info_num = add_obj(b"<< /Title (%s) >>" % _escape(title).encode("latin-1"))

    header = b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n"
    body_parts: List[bytes] = []
    offsets: List[int] = []
    cur = len(header)

    for i, obj in enumerate(objects, start=1):
        offsets.append(cur)
        chunk = b"%d 0 obj\n%s\nendobj\n" % (i, obj)
        body_parts.append(chunk)
        cur += len(chunk)

    body = b"".join(body_parts)
    xref_start = len(header) + len(body)
    xref_lines: List[bytes] = [
        b"xref\n0 %d\n" % (len(objects) + 1),
        b"0000000000 65535 f \n",
    ]
    for off in offsets:
        xref_lines.append(b"%010d 00000 n \n" % off)

    trailer = (
        b"trailer\n<< /Size %d /Root 1 0 R /Info %d 0 R >>\n"
        b"startxref\n%d\n%%%%EOF\n" % (len(objects) + 1, info_num, xref_start)
    )
    return header + body + b"".join(xref_lines) + trailer
