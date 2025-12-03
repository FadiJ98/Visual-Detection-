# streamlit_app.py
from __future__ import annotations

from typing import List, Dict, Any

import cv2
import numpy as np
import streamlit as st
from deepface import DeepFace

from recognition_deepface import RecognizerDeepFace


# ---------- STREAMLIT PAGE CONFIG ----------
st.set_page_config(page_title="Visual Detection (DeepFace)", layout="wide")

# ---------- GLOBAL CSS (animations, dots, etc.) ----------
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


# ---------- SESSION STATE SETUP ----------
if "page" not in st.session_state:
    st.session_state.page = "welcome"   # welcome, settings, upload, results

if "detector_backend" not in st.session_state:
    st.session_state.detector_backend = None

if "upload_key" not in st.session_state:
    st.session_state.upload_key = 0     # for resetting uploader

if "results_table" not in st.session_state:
    st.session_state.results_table = []

if "annotated_rgb" not in st.session_state:
    st.session_state.annotated_rgb = None


# ---------- CALLBACK HELPERS ----------
def reset_all():
    """Reset everything and go back to welcome."""
    st.session_state.page = "welcome"
    st.session_state.detector_backend = None
    st.session_state.upload_key += 1
    st.session_state.results_table = []
    st.session_state.annotated_rgb = None


def go_settings():
    st.session_state.page = "settings"


def go_upload():
    st.session_state.page = "upload"


def go_results():
    st.session_state.page = "results"


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


# ---------- HELPER: DISTANCE → RANGE ----------
def distance_to_range(dist: float) -> str:
    if dist < 0.35:
        return "Very close"
    elif dist < 0.55:
        return "Close"
    elif dist < 0.75:
        return "Midrange"
    elif dist < 1.0:
        return "Far"
    else:
        return "Very far"


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


# ---------- CACHED RECOGNIZER ----------
@st.cache_resource
def load_recognizer() -> RecognizerDeepFace:
    return RecognizerDeepFace(model_name="Facenet512")


recognizer = load_recognizer()


# ---------- PAGE: WELCOME ----------
def page_welcome():
    st.markdown(
        """
        <div style='text-align:center; padding-top:80px;'>
            <h1 class='fade-in fade-1' style='font-size:48px; font-weight:700;'>
                Welcome to Visual Detection
            </h1>
            <h3 class='fade-in fade-2' style='color:#aaaaaa; margin-top:-10px; font-size:18px;'>
                Using DeepFace for face analysis
            </h3>
            <p class='fade-in fade-3 typing-dots'
               style='font-size:20px; margin-top:30px; max-width:600px; margin-left:auto; margin-right:auto;'>
                Press <strong>Start</strong> to upload an image and unfold true identities
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        st.button(
            "Start",
            type="primary",
            use_container_width=True,
            on_click=go_settings,
        )


# ---------- COMMON: TOP NAV BAR (HOME + BACK) ----------
def top_nav(show_back_to=None):
    """show_back_to: None, 'settings', or 'upload' """
    cols = st.columns([1, 1, 6])
    with cols[0]:
        st.button("Home", key=f"home_{st.session_state.page}", on_click=reset_all)

    if show_back_to == "settings":
        with cols[1]:
            st.button("Back to Settings", on_click=go_settings)
    elif show_back_to == "upload":
        with cols[1]:
            st.button("Back to Image Upload", on_click=go_upload)


# ---------- PAGE: SETTINGS (choose detector) ----------
def page_settings():
    top_nav()  # Home only

    st.markdown(
        """
        <div class='fade-in fade-1' style='text-align:center; padding-top:40px;'>
            <h2>Select a detection backend</h2>
            <p style='color:#aaaaaa; max-width:600px; margin:10px auto 30px auto;'>
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
            "<h3>OpenCV</h3>"
            "<p style='color:#bbbbbb;'>Fast, classic Haar-based detection.</p>"
            "</div>",
            unsafe_allow_html=True,
        )
        st.button("Use OpenCV", key="btn_opencv", use_container_width=True,
                  on_click=choose_opencv)

    with col2:
        st.markdown(
            "<div class='fade-in fade-3' style='text-align:center;'>"
            "<h3>RetinaFace</h3>"
            "<p style='color:#bbbbbb;'>More accurate modern deep-learning detector.</p>"
            "</div>",
            unsafe_allow_html=True,
        )
        st.button("Use RetinaFace", key="btn_retina", use_container_width=True,
                  on_click=choose_retina)

    if st.session_state.detector_backend:
        st.markdown(
            f"<p style='text-align:center; margin-top:30px; color:#888;'>"
            f"Current selection: <strong>{st.session_state.detector_backend}</strong>"
            f"</p>",
            unsafe_allow_html=True,
        )


# ---------- PAGE: UPLOAD ----------
def page_upload():
    top_nav(show_back_to="settings")  # Home + Back to Settings

    st.markdown(
        """
        <div class='fade-in fade-1' style='padding-top:20px;'>
            <h2>Upload an image</h2>
            <p style='color:#aaaaaa;'>
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

    # ---------- DETECT CALLBACK ----------
    def detect_image():
        if img_file is None:
            return

        file_bytes = np.frombuffer(img_file.read(), np.uint8)
        bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if bgr is None:
            st.error("Failed to read image. Try another file.")
            return

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        try:
            result = DeepFace.analyze(
                img_path=rgb,
                actions=["emotion"],
                enforce_detection=True,
                detector_backend=st.session_state.detector_backend,
            )
        except Exception as e:
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

        # Sort by vertical then horizontal position
        face_infos.sort(key=lambda f: (f["cy"], f["cx"]))

        results_table: List[Dict[str, Any]] = []

        # --- Draw colored boxes + build table ---
        for idx, info in enumerate(face_infos, start=1):
            r = info["r"]
            x1, y1, x2, y2 = info["x1"], info["y1"], info["x2"], info["y2"]

            face = bgr[y1:y2, x1:x2]
            emotion = r.get("dominant_emotion", "unknown")

            # Recognition + range
            name, dist = recognizer.infer(face)
            range_label = distance_to_range(float(dist))

            # Pick color from palette
            color_name, color_bgr = COLOR_PALETTE[(idx - 1) % len(COLOR_PALETTE)]

            label = emotion if name == "Unknown" else f"{name} | {emotion}"

            # Rectangle in that color
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color_bgr, 3)

            # Optional: colored label background
            label_bg_y2 = max(0, y1 - 5)
            label_bg_y1 = max(0, label_bg_y2 - 25)
            cv2.rectangle(
                annotated,
                (x1, label_bg_y1),
                (x2, label_bg_y2),
                color_bgr,
                thickness=-1,
            )

            # Label text in black/white depending on brightness
            brightness = 0.299 * color_bgr[2] + 0.587 * color_bgr[1] + 0.114 * color_bgr[0]
            text_color = (0, 0, 0) if brightness > 150 else (255, 255, 255)

            cv2.putText(
                annotated,
                label,
                (x1 + 4, label_bg_y2 - 7),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                text_color,
                2,
                lineType=cv2.LINE_AA,
            )

            results_table.append(
                {
                    "Face #": idx,
                    "Name": name,
                    "Range": range_label,
                    "Emotion": emotion,
                    "Color": color_name,
                }
            )

        st.session_state.annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        st.session_state.results_table = results_table
        go_results()

    # Detect button using callback
    st.button(
        "Detect",
        type="primary",
        disabled=img_file is None,
        on_click=detect_image,
    )


# ---------- PAGE: RESULTS ----------
def page_results():
    top_nav(show_back_to="upload")  # Home + Back to Image Upload

    st.markdown(
        """
        <div class='fade-in fade-1' style='padding-top:20px;'>
            <h2>Detection Results</h2>
            <p style='color:#aaaaaa;'>
                Review the detected faces, emotions, ranges, and colors.
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
        st.image(st.session_state.annotated_rgb, channels="RGB", use_column_width=True)

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
elif st.session_state.page == "results":
    page_results()
