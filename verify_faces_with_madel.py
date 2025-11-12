import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model

model = load_model("face_angle_classifier.h5")
labels = ["front", "left", "right"]  # adjust if your folders differ

mp_face = mp.solutions.face_detection
mp_draw = mp.solutions.drawing_utils

def preprocess_face(face):
    face = cv2.resize(face, (128,128))
    face = face / 255.0
    return np.expand_dims(face, axis=0)

cap = cv2.VideoCapture(0)

with mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.45) as detector:
    while True:
        success, frame = cap.read()
        if not success:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = detector.process(rgb)

        if results.detections:
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                h, w, _ = frame.shape
                x, y, w_box, h_box = int(bbox.xmin * w), int(bbox.ymin * h), int(bbox.width * w), int(bbox.height * h)
                face = frame[y:y+h_box, x:x+w_box]

                if face.size > 0:
                    pred = model.predict(preprocess_face(face))
                    label = labels[np.argmax(pred)]
                    conf = np.max(pred)
                    cv2.putText(frame, f"{label} ({conf:.2f})", (x, y-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                mp_draw.draw_detection(frame, detection)

        cv2.imshow("Face Detection + Verification", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
