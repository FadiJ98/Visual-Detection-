import cv2
import numpy as np
from deepface import DeepFace
from recognition_deepface import RecognizerDeepFace
import streamlit as st

@st.cache_resource
def load_recognizer():
    return RecognizerDeepFace(model_name="Facenet512")

recognizer = load_recognizer()

COLOR_PALETTE = [
    ("Blue", (255, 80, 20)), ("Green", (80, 220, 80)),
    ("Red", (40, 40, 255)), ("Purple", (200, 80, 200))
]

def run_detection(img_bytes, backend):
    file_bytes = np.frombuffer(img_bytes, np.uint8)
    bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    result = DeepFace.analyze(
        img_path=rgb,
        actions=["emotion", "gender"],
        detector_backend=backend,
        enforce_detection=True
    )

    faces = result if isinstance(result, list) else [result]
    annotated = bgr.copy()

    results = []
    for i, r in enumerate(faces, start=1):
        region = r["region"]
        x, y, w, h = region.values()
        face = bgr[y:y+h, x:x+w]

        name, _ = recognizer.infer(face)
        emotion = r["dominant_emotion"]
        gender = r.get("dominant_gender")

        color_name, color = COLOR_PALETTE[(i - 1) % len(COLOR_PALETTE)]
        cv2.rectangle(annotated, (x, y), (x+w, y+h), color, 3)

        results.append({
            "Face #": i,
            "Name": name,
            "Gender": gender,
            "Emotion": emotion,
            "Color": color_name
        })

    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    return annotated_rgb, results
