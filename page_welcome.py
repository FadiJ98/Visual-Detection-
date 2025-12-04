from __future__ import annotations
import streamlit as st

from layout import inject_global_css, set_background
from state import go_settings


def page_welcome() -> None:
    inject_global_css()
    set_background(True)

    st.markdown(
        """
        <div style='text-align:center; padding-top:80px;'>
            <h1 class='fade-in fade-1' style='font-size:48px; font-weight:700; color:#ffffff;'>
                Welcome to Visual Detection
            </h1>
            <h3 class='fade-in fade-2' style='color:#eeeeee; margin-top:-10px; font-size:18px;'>
                Using DeepFace for face analysis
            </h3>
            <p class='fade-in fade-3 typing-dots'
               style='font-size:20px; margin-top:30px; max-width:600px; margin-left:auto; margin-right:auto; color:#ffffff;'>
                Press <strong>Start</strong> to upload an image and unfold true identities
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    _, col2, _ = st.columns([1, 1, 1])
    with col2:
        st.button(
            "Start",
            type="primary",
            use_container_width=True,
            on_click=go_settings,
        )
