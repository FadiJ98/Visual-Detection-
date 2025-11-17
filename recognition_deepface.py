# recognition_deepface.py
from typing import Dict, Tuple
import os
import cv2
import numpy as np
from deepface import DeepFace


class RecognizerDeepFace:
    """
    Simple wrapper around DeepFace for embedding + nearest-neighbor
    identity matching using .npy files on disk.
    """

    def __init__(self, model_name: str = "Facenet512", db_dir: str = "data/embeddings"):
        self.model_name = model_name
        self.db_dir = db_dir
        os.makedirs(db_dir, exist_ok=True)

        # Build once; skip DeepFace’s own detector since we crop already.
        self.model = DeepFace.build_model(model_name)
        self._memory_db: Dict[str, np.ndarray] = {}
        self.load_db()

    # ---------- DB helpers ----------

    def load_db(self) -> None:
        """
        Load all .npy vectors in db_dir into memory.
        Filenames are used as identities (name.npy -> name).
        """
        self._memory_db.clear()
        for f in os.listdir(self.db_dir):
            if not f.endswith(".npy"):
                continue
            name = f[:-4]
            path = os.path.join(self.db_dir, f)
            try:
                vec = np.load(path)
                self._memory_db[name] = vec.astype(np.float32)
            except Exception:
                continue

    def save_embedding(self, name: str, vec: np.ndarray) -> None:
        """
        Save an embedding for a given identity.
        """
        path = os.path.join(self.db_dir, f"{name}.npy")
        np.save(path, vec.astype(np.float32))
        self._memory_db[name] = vec.astype(np.float32)

    # ---------- Core DeepFace calls ----------

    def embed(self, face_bgr: np.ndarray) -> np.ndarray | None:
        """
        Turn a cropped face (BGR) into an embedding vector.
        """
        if face_bgr is None:
            return None

        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        vec = DeepFace.represent(
            img_path=face_rgb,
            model=self.model,
            enforce_detection=False,
        )
        # DeepFace returns list of dicts when img_path is array-like
        if isinstance(vec, list) and len(vec) and isinstance(vec[0], dict):
            return np.array(vec[0]["embedding"], dtype=np.float32)
        return np.array(vec, dtype=np.float32)

    def infer(self, face_bgr: np.ndarray, thresh: float = 0.4) -> Tuple[str, float]:
        """
        Compare a face embedding to all known identities.
        Returns (best_name or "Unknown", distance).
        """
        vec = self.embed(face_bgr)
        if vec is None:
            return "Unknown", 1.0

        best_name = "Unknown"
        best_dist = 1.0

        for name, ref in self._memory_db.items():
            dist = float(np.linalg.norm(vec - ref))
            if dist < best_dist:
                best_dist = dist
                best_name = name

        if best_dist > thresh:
            return "Unknown", best_dist
        return best_name, best_dist
