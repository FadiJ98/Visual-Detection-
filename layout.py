from __future__ import annotations
import streamlit as st

# ---------- GLOBAL CSS (animations, dots) ----------
def inject_global_css() -> None:
    st.markdown(
        """
        <style>
        @keyframes fadeUp {
            0%   { opacity: 0; transform: translateY(12px); }
            100% { opacity: 1; transform: translateY(0); }
        }

        .fade-in      { opacity: 0; animation: fadeUp 0.8s ease-out forwards; }
        .fade-1       { animation-delay: 0.1s; }
        .fade-2       { animation-delay: 0.4s; }
        .fade-3       { animation-delay: 0.7s; }
        .fade-4       { animation-delay: 1.0s; }

        @keyframes dotsBlink {
            0%   { content: '';   }
            25%  { content: '.';  }
            50%  { content: '..'; }
            75%  { content: '...';}
            100% { content: '';   }
        }
        .typing-dots::after {
            content: '';
            animation: dotsBlink 1.2s steps(4, end) infinite;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ---------- HELPER: SET BACKGROUND ----------
BG_GIF = "https://mir-s3-cdn-cf.behance.net/project_modules/disp/9c0722106004343.5f85fead2894a.gif"


def set_background(use_gif: bool) -> None:
    """Apply / remove the animated background on the whole app."""
    if use_gif:
        css = f"""
        <style>
        .stApp {{
            background-image: url('{BG_GIF}');
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
            background-repeat: no-repeat;
        }}
        </style>
        """
    else:
        css = """
        <style>
        .stApp {
            background-image: none;
            background-color: #0e1117;
        }
        </style>
        """
    st.markdown(css, unsafe_allow_html=True)


def set_black_background() -> None:
    """Pure black background, no image (for loading + results)."""
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #000000 !important;
            background-image: none !important;
        }
        [data-testid="stHeader"] {
            background: rgba(0,0,0,0) !important;
        }
        [data-testid="stToolbar"] {
            display: none !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
