from __future__ import annotations
from typing import List, Dict, Any

import cv2
import numpy as np
import streamlit as st
from deepface import DeepFace
import random

from layout import set_black_background
from navigation import top_nav
from state import go_results
from detection_config import COLOR_PALETTE, FEMALE_NAMES, MALE_NAMES, get_recognizer


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

                # Recognizer prediction
                name, dist = recognizer.infer(face)

                # ---------- FAKE NAME GENERATION WHEN UNKNOWN ----------
                if name == "Unknown":
                    g = str(gender).lower()

                    if "woman" in g or "female" in g:
                        name = random.choice(FEMALE_NAMES)
                    elif "man" in g or "male" in g:
                        name = random.choice(MALE_NAMES)
                    else:
                        name = random.choice(FEMALE_NAMES + MALE_NAMES)

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
