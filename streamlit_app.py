from __future__ import annotations

from typing import List, Dict, Any

import cv2
import numpy as np
import streamlit as st
from deepface import DeepFace

from recognition_deepface import RecognizerDeepFace


# ---------- STREAMLIT PAGE CONFIG ----------
st.set_page_config(page_title="Visual Detection (DeepFace)", layout="wide")

# ---------- GLOBAL CSS (animations, dots) ----------
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


# ---------- COLOR PALETTE (name, BGR) ----------
COLOR_PALETTE: List[tuple[str, tuple[int, int, int]]] = [
    ("Blue",       (255,  80,  20)),
    ("Green",      ( 80, 220,  80)),
    ("Orange",     ( 40, 140, 255)),
    ("Yellow",     ( 40, 230, 255)),
    ("Purple",     (200,  80, 200)),
    ("Brown",      ( 40,  60, 140)),
    ("Gray",       (160, 160, 160)),
    ("Red",        ( 40,  40, 255)),
    ("Olive",      ( 60, 120,  60)),
    ("Maroon",     ( 40,  40, 140)),
    ("Violet",     (230, 130, 230)),
    ("Charcoal",   ( 60,  60,  60)),
    ("Magenta",    (230,  80, 230)),
    ("Bronze",     ( 60, 120, 200)),
    ("Cream",      (210, 220, 230)),
    ("Tan",        (140, 180, 220)),
    ("Teal",       (140, 200, 140)),
    ("Black",      (  0,   0,   0)),
    ("Mustard",    ( 60, 200, 220)),
    ("Navy Blue",  (180,  60,  40)),
    ("Coral",      (120, 160, 255)),
    ("Burgundy",   ( 40,  40, 110)),
    ("Lavender",   (220, 200, 250)),
    ("Mauve",      (200, 180, 220)),
    ("Peach",      (180, 200, 240)),
    ("Rust",       ( 60,  80, 160)),
    ("Gold",       ( 40, 200, 255)),
    ("Pink",       (220, 180, 250)),
    ("Silver",     (200, 200, 200)),
    ("Cyan",       (250, 220,  80)),
]


# ---------- CACHED RECOGNIZER (LAZY-LOADED) ----------
@st.cache_resource
def get_recognizer() -> RecognizerDeepFace:
    """
    Lazily create and cache the RecognizerDeepFace instance.

    This avoids loading Facenet512 + DB at import time, which is heavy
    in constrained environments like Streamlit Cloud.
    """
    return RecognizerDeepFace(model_name="Facenet512")


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


# ---------- PAGE: WELCOME ----------
def page_welcome() -> None:
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


# ---------- PAGE: SETTINGS ----------
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


# ---------- PAGE: UPLOAD ----------
def page_upload() -> None:
    set_background(True)

    top_nav(show_back_to="settings")  # Home + Back to Settings

    st.markdown(
        """
        <div class='fade-in fade-1' style='padding-top:20px;'>
            <h2 style="color:#ffffff;">Upload an image</h2>
            <p style='color:#dddddd;'>
                Choose a photo and run detection with your selected backend.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not st.session_state.detector_backend:
        st.error("Please choose a detector on the Settings page first.")
        return

    # Reset image button
    reset_col, _ = st.columns([1, 4])
    with reset_col:
        st.button("Reset image", key="reset_image", on_click=reset_image)

    # Upload widget
    img_file = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        key=f"uploader_{st.session_state.upload_key}",
        help="Static images only (no live camera).",
    )

    # ---------- START DETECTION (NAVIGATE TO LOADING PAGE) ----------
    def start_detection() -> None:
        if img_file is None:
            return
        st.session_state.uploaded_image_bytes = img_file.getvalue()
        st.session_state.pending_detection = True
        go_loading()

    st.button(
        "Detect",
        type="primary",
        disabled=img_file is None,
        on_click=start_detection,
    )


# ---------- PAGE: LOADING (GIF + ANALYSIS) ----------
def page_loading() -> None:
    set_black_background()
    top_nav()  # just Home

    # Centered GIF (big)
    st.markdown(
        """
        <div style="display:flex; justify-content:center; align-items:center; height:80vh;">
          <div style="text-align:center;">
            <img src="https://miro.medium.com/v2/1*4Tr0FOsdUgkF32T3mdu6pg.gif"
                 style="width:760px; max-width:90vw; border-radius:20px;"/>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Only run detection once when pending_detection is True
    if st.session_state.pending_detection and st.session_state.uploaded_image_bytes:
        with st.spinner("Running DeepFace analysis..."):
            img_bytes = st.session_state.uploaded_image_bytes
            file_bytes = np.frombuffer(img_bytes, np.uint8)
            bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            if bgr is None:
                st.session_state.pending_detection = False
                st.error("Failed to read image. Try another file.")
                return

            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

            try:
                result = DeepFace.analyze(
                    img_path=rgb,
                    actions=["emotion", "gender"],
                    enforce_detection=True,
                    detector_backend=st.session_state.detector_backend,
                )
            except Exception as e:
                st.session_state.pending_detection = False
                st.error(f"DeepFace error: {e}")
                return

            faces_raw = result if isinstance(result, list) else [result]
            annotated = bgr.copy()
            h, w = bgr.shape[:2]

            # --- Build sortable list of faces (top->bottom, left->right) ---
            face_infos: List[Dict[str, Any]] = []
            for r in faces_raw:
                region = r.get("region") or {}
                x = int(region.get("x", 0))
                y = int(region.get("y", 0))
                fw = int(region.get("w", 0))
                fh = int(region.get("h", 0))

                x1 = max(0, x)
                y1 = max(0, y)
                x2 = min(w - 1, x + fw)
                y2 = min(h - 1, y + fh)

                if x2 <= x1 or y2 <= y1:
                    continue

                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0

                face_infos.append(
                    {
                        "r": r,
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                        "cx": cx,
                        "cy": cy,
                    }
                )

            face_infos.sort(key=lambda f: (f["cy"], f["cx"]))

            results_table: List[Dict[str, Any]] = []

            # --- Draw colored boxes + build table ---
            # Lazy-load recognizer here (heavy object, cached via @st.cache_resource)
            recognizer = get_recognizer()

            for idx, info in enumerate(face_infos, start=1):
                r = info["r"]
                x1, y1, x2, y2 = info["x1"], info["y1"], info["x2"], info["y2"]

                face = bgr[y1:y2, x1:x2]

                emotion = r.get("dominant_emotion", "unknown")

                raw_gender = r.get("gender") or r.get("dominant_gender")
                if isinstance(raw_gender, dict) and raw_gender:
                    gender = max(raw_gender, key=raw_gender.get)
                else:
                    gender = raw_gender if raw_gender else "Unknown"

                name, dist = recognizer.infer(face)

                color_name, color_bgr = COLOR_PALETTE[(idx - 1) % len(COLOR_PALETTE)]

                cv2.rectangle(annotated, (x1, y1), (x2, y2), color_bgr, 3)

                results_table.append(
                    {
                        "Face #": idx,
                        "Color": color_name,
                        "Name": name,
                        "Gender": gender,
                        "Emotion": emotion,
                    }
                )

            st.session_state.annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            st.session_state.results_table = results_table

            st.session_state.pending_detection = False
            go_results()

            if hasattr(st, "rerun"):
                st.rerun()
            elif hasattr(st, "experimental_rerun"):
                st.experimental_rerun()


# ---------- PAGE: RESULTS ----------
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
