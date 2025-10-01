import streamlit as st
import cv2, numpy as np
from pathlib import Path
from src.detection_mediapipe import FaceDetectorMP
from src.face_align import crop_face
from src.recognition_deepface import RecognizerDeepFace

st.set_page_config(page_title="Visual Detection", layout="wide")
st.title("Visual Detection – Face Detect & Recognize")

det = FaceDetectorMP(min_conf=0.5)
rec = RecognizerDeepFace("Facenet512")

col1, col2 = st.columns(2)

with col1:
    img_file = st.file_uploader("Upload image", type=["jpg","png","jpeg"])
    if img_file:
        arr = np.frombuffer(img_file.read(), np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        boxes = det.detect(frame)
        for b in boxes:
            face = crop_face(frame, b, 160)
            vec = rec.embed(face)
            name, dist = rec.match(vec, 0.4)
            x1,y1,x2,y2,_ = b
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.putText(frame, f"{name} ({dist:.2f})",(x1,y1-6),
                        cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)
        st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption="Result")

with col2:
    st.header("Enroll")
    name = st.text_input("Person name")
    enroll_file = st.file_uploader("Upload face for enrollment", type=["jpg","png","jpeg"], key="enroll")
    if enroll_file and name:
        arr = np.frombuffer(enroll_file.read(), np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        boxes = det.detect(frame)
        if boxes:
            face = crop_face(frame, boxes[0], 160)
            rec.enroll(name, rec.embed(face))
            st.success(f"Enrolled {name}!")
