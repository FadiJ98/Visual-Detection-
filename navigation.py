from __future__ import annotations
import streamlit as st

from state import reset_all, go_settings, go_upload


# ---------- COMMON: TOP NAV BAR (HOME + BACK) ----------
def top_nav(show_back_to: str | None = None) -> None:
    """show_back_to: None, 'settings', or 'upload'."""
    cols = st.columns([1, 1, 6])
    with cols[0]:
        st.button("Home", key=f"home_{st.session_state.page}", on_click=reset_all)

    if show_back_to == "settings":
        with cols[1]:
            st.button("Back to Settings", on_click=go_settings)
    elif show_back_to == "upload":
        with cols[1]:
            st.button("Back to Image Upload", on_click=go_upload)
