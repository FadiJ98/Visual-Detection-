from __future__ import annotations
import streamlit as st

from layout import set_black_background
from navigation import top_nav


def page_results() -> None:
    set_black_background()

    top_nav(show_back_to="upload")  # Home + Back to Image Upload

    st.markdown(
        """
        <div class='fade-in fade-1' style='padding-top:20px;'>
            <h2 style="color:#ffffff;">Detection Results</h2>
            <p style='color:#aaaaaa;'>
                Review the detected faces, genders, emotions, and colors.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.session_state.annotated_rgb is None or not st.session_state.results_table:
        st.info("No results to show yet. Go back to upload an image.")
        return

    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Annotated image")
        st.image(st.session_state.annotated_rgb, channels="RGB", use_container_width=True)

    with col_right:
        st.subheader("Analysis")
        st.dataframe(st.session_state.results_table, hide_index=True)
