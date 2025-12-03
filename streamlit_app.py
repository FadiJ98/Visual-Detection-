# streamlit_app.py
from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any

import cv2
import numpy as np
import streamlit as st
from deepface import DeepFace

from recognition_deepface import RecognizerDeepFace


# ---------- STREAMLIT PAGE CONFIG ----------
st.set_page_config(page_title="Visual Detection — DeepFace Only", layout="wide")
st.title("Visual Detection — Face Detection + Emotion + Recognition (DeepFace)")


# ---------- CACHED RECOGNIZER ----------
@st.cache_resource
def load_recognizer() -> RecognizerDeepFace:
    # Uses your existing DeepFace-based recognizer class
    return RecognizerDeepFace(model_name="Facenet512")


recognizer = load_recognizer()


# ---------- SIDEBAR ----------
st.sidebar.header("Settings")
detector_backend = st.sidebar.selectbox(
    "Detector backend",
    options=["retinaface", "opencv", "mtcnn"],
    index=0,
    help="If RetinaFace gives errors, try switching to opencv.",
)

threshold = st.sidebar.slider(
    "Recognition distance threshold",
    min_value=0.5,
    max_value=1.5,
    value=float(recognizer.threshold),
    step=0.05,
    help="Lower = stricter matching; higher = more forgiving.",
)

recognizer.threshold = threshold


# ---------- MAIN LAYOUT ----------
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Upload image")
    img_file = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        help="Static images only (no live camera).",
    )

    run_button = st.button("Detect + Analyze", disabled=img_file is None)

results_table: List[Dict[str, Any]] = []

if img_file and run_button:
    # Read image into OpenCV BGR
    file_bytes = np.frombuffer(img_file.read(), np.uint8)
    bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if bgr is None:
        st.error("Failed to read image. Try another file.")
    else:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        try:
            # DeepFace: detect faces + emotions
            result = DeepFace.analyze(
                img_path=rgb,
                actions=["emotion"],
                enforce_detection=True,
                detector_backend=detector_backend,
            )
        except Exception as e:
            st.error(f"DeepFace error: {e}")
            st.stop()

        faces = result if isinstance(result, list) else [result]
        annotated = bgr.copy()
        h, w = bgr.shape[:2]

        for idx, r in enumerate(faces, start=1):
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

            # Crop face
            face = bgr[y1:y2, x1:x2]

            # Emotion
            emotion = r.get("dominant_emotion", "unknown")

            # Recognition via your RecognizerDeepFace
            name, dist = recognizer.infer(face)

            # Draw box + label
            label = emotion if name == "Unknown" else f"{name} | {emotion}"
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            y_text = max(15, y1 - 10)
            cv2.putText(
                annotated,
                label,
                (x1, y_text),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
                lineType=cv2.LINE_AA,
            )

            # Collect row for results table
            results_table.append(
                {
                    "Face #": idx,
                    "Name": name,
                    "Distance": round(float(dist), 3),
                    "Emotion": emotion,
                }
            )

        with col_left:
            st.subheader("Annotated image")
            st.image(
                cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                channels="RGB",
                use_column_width=True,
            )

        with col_right:
            st.subheader("Analysis")
            if results_table:
                st.write(f"Detected **{len(results_table)}** face(s).")
                st.dataframe(results_table, hide_index=True)
            else:
                st.info("No faces found.")


elif img_file and not run_button:
    # Show raw uploaded image before running analysis
    with col_left:
        st.image(img_file, caption="Uploaded image", use_column_width=True)
    with col_right:
        st.info("Click **Detect + Analyze** to run DeepFace.")
