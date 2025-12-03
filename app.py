import streamlit as st
from state import init_session_state
from pages import (
    page_welcome,
    page_settings,
    page_upload,
    page_loading,
    page_results,
)

# ---------- STREAMLIT PAGE CONFIG ----------
st.set_page_config(page_title="Visual Detection (DeepFace)", layout="wide")

# ---------- INIT SESSION STATE ----------
init_session_state()

# ---------- PAGE ROUTER ----------
if st.session_state.page == "welcome":
    page_welcome()
elif st.session_state.page == "settings":
    page_settings()
elif st.session_state.page == "upload":
    page_upload()
elif st.session_state.page == "loading":
    page_loading()
elif st.session_state.page == "results":
    page_results()
