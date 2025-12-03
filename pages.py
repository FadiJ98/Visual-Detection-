import streamlit as st
from ui import set_background, load_css, top_nav
from state import *
from recognition import run_detection


def page_welcome():
    load_css()
    set_background(True)
    st.title("Welcome to Visual Detection")
    st.button("Start", on_click=go_settings)


def page_settings():
    set_background(True)
    top_nav(reset_all)

    col1, col2 = st.columns(2)
    with col1:
        st.button("Use OpenCV", on_click=choose_opencv)
    with col2:
        st.button("Use RetinaFace", on_click=choose_retina)


def page_upload():
    set_background(True)
    top_nav(reset_all, go_settings, back_to="settings")

    img = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])
    if st.button("Detect") and img:
        st.session_state.uploaded_image_bytes = img.getvalue()
        go_loading()


def page_loading():
    set_background(False)
    top_nav(reset_all)

    annotated, table = run_detection(
        st.session_state.uploaded_image_bytes,
        st.session_state.detector_backend
    )

    st.session_state.annotated_rgb = annotated
    st.session_state.results_table = table
    go_results()
    st.rerun()


def page_results():
    set_background(False)
    top_nav(reset_all, go_upload, back_to="upload")

    st.image(st.session_state.annotated_rgb)
    st.dataframe(st.session_state.results_table)
