from __future__ import annotations
import streamlit as st

from layout import set_background
from navigation import top_nav
from state import reset_image, go_loading


def page_upload() -> None:
    set_background(True)

    top_nav(show_back_to="settings")  # Home + Back to Settings

    st.markdown(
        """
        <div class='fade-in fade-1' style='padding-top:20px;'>
            <h2 style="color:#ffffff;">Upload an image</h2>
            <p style='color:#dddddd;'>
                Choose a photo and run detection with your selected backend.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not st.session_state.detector_backend:
        st.error("Please choose a detector on the Settings page first.")
        return

    # Reset image button
    reset_col, _ = st.columns([1, 4])
    with reset_col:
        st.button("Reset image", key="reset_image", on_click=reset_image)

    # Upload widget
    img_file = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        key=f"uploader_{st.session_state.upload_key}",
        help="Static images only (no live camera).",
    )

    # ---------- START DETECTION (NAVIGATE TO LOADING PAGE) ----------
    def start_detection() -> None:
        if img_file is None:
            return
        st.session_state.uploaded_image_bytes = img_file.getvalue()
        st.session_state.pending_detection = True
        go_loading()

    st.button(
        "Detect",
        type="primary",
        disabled=img_file is None,
        on_click=start_detection,
    )
