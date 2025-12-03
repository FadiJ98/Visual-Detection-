#import
import streamlit as st

def init_session_state():
    if "page" not in st.session_state:
        st.session_state.page = "welcome"

    if "detector_backend" not in st.session_state:
        st.session_state.detector_backend = None

    if "upload_key" not in st.session_state:
        st.session_state.upload_key = 0

    if "results_table" not in st.session_state:
        st.session_state.results_table = []

    if "annotated_rgb" not in st.session_state:
        st.session_state.annotated_rgb = None

    if "uploaded_image_bytes" not in st.session_state:
        st.session_state.uploaded_image_bytes = None

    if "pending_detection" not in st.session_state:
        st.session_state.pending_detection = False


# ---------- CALLBACKS ----------
def reset_all():
    st.session_state.page = "welcome"
    st.session_state.detector_backend = None
    st.session_state.upload_key += 1
    st.session_state.results_table = []
    st.session_state.annotated_rgb = None
    st.session_state.uploaded_image_bytes = None
    st.session_state.pending_detection = False


def go_settings():
    st.session_state.page = "settings"


def go_upload():
    st.session_state.page = "upload"


def go_results():
    st.session_state.page = "results"


def go_loading():
    st.session_state.page = "loading"


def choose_opencv():
    st.session_state.detector_backend = "opencv"
    st.session_state.page = "upload"


def choose_retina():
    st.session_state.detector_backend = "retinaface"
    st.session_state.page = "upload"


def reset_image():
    st.session_state.upload_key += 1
    st.session_state.results_table = []
    st.session_state.annotated_rgb = None
    st.session_state.uploaded_image_bytes = None
    st.session_state.pending_detection = False
