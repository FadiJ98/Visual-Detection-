from __future__ import annotations
import streamlit as st


def init_session_state() -> None:
    # ---------- SESSION STATE SETUP ----------
    if "page" not in st.session_state:
        st.session_state.page = "welcome"   # welcome, settings, upload, loading, results

    if "detector_backend" not in st.session_state:
        st.session_state.detector_backend = None

    if "upload_key" not in st.session_state:
        st.session_state.upload_key = 0     # for resetting uploader

    if "results_table" not in st.session_state:
        st.session_state.results_table = []

    if "annotated_rgb" not in st.session_state:
        st.session_state.annotated_rgb = None

    # for loading page
    if "uploaded_image_bytes" not in st.session_state:
        st.session_state.uploaded_image_bytes = None

    if "pending_detection" not in st.session_state:
        st.session_state.pending_detection = False


# ---------- CALLBACK HELPERS ----------
def reset_all() -> None:
    """Reset everything and go back to welcome."""
    st.session_state.page = "welcome"
    st.session_state.detector_backend = None
    st.session_state.upload_key += 1
    st.session_state.results_table = []
    st.session_state.annotated_rgb = None
    st.session_state.uploaded_image_bytes = None
    st.session_state.pending_detection = False


def go_settings() -> None:
    st.session_state.page = "settings"


def go_upload() -> None:
    st.session_state.page = "upload"


def go_results() -> None:
    st.session_state.page = "results"


def go_loading() -> None:
    st.session_state.page = "loading"


def choose_opencv() -> None:
    st.session_state.detector_backend = "opencv"
    st.session_state.page = "upload"


def choose_retina() -> None:
    st.session_state.detector_backend = "retinaface"
    st.session_state.page = "upload"


def reset_image() -> None:
    st.session_state.upload_key += 1
    st.session_state.results_table = []
    st.session_state.annotated_rgb = None
    st.session_state.uploaded_image_bytes = None
    st.session_state.pending_detection = False
