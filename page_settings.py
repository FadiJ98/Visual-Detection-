from __future__ import annotations
import streamlit as st

from layout import set_background
from navigation import top_nav
from state import choose_opencv, choose_retina


def page_settings() -> None:
    set_background(True)

    top_nav()  # Home only

    st.markdown(
        """
        <div class='fade-in fade-1' style='text-align:center; padding-top:40px;'>
            <h2 style="color:#ffffff;">Select a detection backend</h2>
            <p style='color:#dddddd; max-width:600px; margin:10px auto 30px auto;'>
                Choose how faces are detected before emotion and identity analysis.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            "<div class='fade-in fade-2' style='text-align:center;'>"
            "<h3 style='color:#ffffff;'>OpenCV</h3>"
            "<p style='color:#dddddd;'>Fast, classic Haar-based detection.</p>"
            "</div>",
            unsafe_allow_html=True,
        )
        st.button(
            "Use OpenCV",
            key="btn_opencv",
            use_container_width=True,
            on_click=choose_opencv,
        )

    with col2:
        st.markdown(
            "<div class='fade-in fade-3' style='text-align:center;'>"
            "<h3 style='color:#ffffff;'>RetinaFace</h3>"
            "<p style='color:#dddddd;'>More accurate modern deep-learning detector.</p>"
            "</div>",
            unsafe_allow_html=True,
        )
        st.button(
            "Use RetinaFace",
            key="btn_retina",
            use_container_width=True,
            on_click=choose_retina,
        )

    if st.session_state.detector_backend:
        st.markdown(
            f"<p style='text-align:center; margin-top:30px; color:#e0e0e0;'>"
            f"Current selection: <strong>{st.session_state.detector_backend}</strong>"
            f"</p>",
            unsafe_allow_html=True,
        )
