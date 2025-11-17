import cv2
import numpy as np
from tensorflow.keras.models import load_model

import mediapipe as mp

MODEL_PATH = "face_angle_classifier.h5"
labels = ["front", "left", "right"]  # adjust if your folders differ

model = load_model(MODEL_PATH)

mp_face = mp.solutions.face_detection
mp_draw = mp.solutions.drawing_utils


def preprocess_face(face_bgr: np.ndarray) -> np.ndarray:
    face = cv2.resize(face_bgr, (128, 128))
    face = face / 255.0
    return np.expand_dims(face, axis=0)


def main() -> None:
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
                    x = int(bbox.xmin * w)
                    y = int(bbox.ymin * h)
                    w_box = int(bbox.width * w)
                    h_box = int(bbox.height * h)

                    face = frame[y : y + h_box, x : x + w_box]
                    if face.size > 0:
                        pred = model.predict(preprocess_face(face), verbose=0)
                        idx = int(np.argmax(pred))
                        label = labels[idx]
                        conf = float(np.max(pred))

                        cv2.putText(
                            frame,
                            f"{label} ({conf:.2f})",
                            (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 255, 0),
                            2,
                        )

                    mp_draw.draw_detection(frame, detection)

            cv2.imshow("Face Detection + Angle Classification", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
