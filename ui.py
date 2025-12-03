#import
import streamlit as st

BG_GIF = "https://mir-s3-cdn-cf.behance.net/project_modules/disp/9c0722106004343.5f85fead2894a.gif"

def load_css():
    st.markdown("""
    <style>
    @keyframes fadeUp {
        0% { opacity: 0; transform: translateY(12px); }
        100% { opacity: 1; transform: translateY(0); }
    }
    .fade-in { opacity: 0; animation: fadeUp 0.8s ease-out forwards; }
    .fade-1 { animation-delay: 0.1s; }
    .fade-2 { animation-delay: 0.4s; }
    .fade-3 { animation-delay: 0.7s; }
    </style>
    """, unsafe_allow_html=True)


def set_background(use_gif=True):
    if use_gif:
        css = f"""
        <style>
        .stApp {{
            background-image: url('{BG_GIF}');
            background-size: cover;
            background-position: center;
        }}
        </style>
        """
    else:
        css = """
        <style>
        .stApp {
            background-color: #000000;
        }
        </style>
        """
    st.markdown(css, unsafe_allow_html=True)


def top_nav(reset_all, go_settings=None, go_upload=None, back_to=None):
    cols = st.columns([1, 1, 6])

    with cols[0]:
        st.button("Home", on_click=reset_all)

    if back_to == "settings":
        with cols[1]:
            st.button("Back", on_click=go_settings)

    if back_to == "upload":
        with cols[1]:
            st.button("Back", on_click=go_upload)
