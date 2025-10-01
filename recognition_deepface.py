from deepface import DeepFace
import numpy as np
import os, cv2

class RecognizerDeepFace:
    def __init__(self, model_name="Facenet512", db_dir="data/embeddings"):
        self.model_name = model_name
        self.db_dir = db_dir
        os.makedirs(db_dir, exist_ok=True)
        self.model = DeepFace.build_model(model_name)

    def embed(self, face_bgr):
        if face_bgr is None: return None
        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        vec = DeepFace.represent(img_path = face_rgb, model = self.model, enforce_detection=False)
        # DeepFace returns list of dicts when img_path is array-like; handle common form:
        if isinstance(vec, list) and len(vec) and isinstance(vec[0], dict):
            return np.array(vec[0]["embedding"], dtype=np.float32)
        return np.array(vec, dtype=np.float32)

    def infer(self, face_bgr, thresh=0.4):
        vec = self.embed(face_bgr)
        if vec is None: return ("Unknown", 1.0)
        # Simple nearest neighbor against on-disk .npy vectors: name.npy
        best_name, best_dist = "Unknown", 1.0
        for f in os.listdir(self.db_dir):
            if not f.endswith(".npy"): continue
            name = f[:-4]
            ref = np.load(os.path.join(self.db_dir, f))
            dist = float(np.linalg.norm(vec - ref))
            if dist < best_dist:
                best_dist, best_name = dist, name
        return (best_name if best_dist <= thresh else "Unknown", best_dist)
