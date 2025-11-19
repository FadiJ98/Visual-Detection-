# recognition_deepface.py
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from deepface import DeepFace


class RecognizerDeepFace:
    """
    Simple DeepFace-based face recognizer.

    - Loads one image per identity from `db_path`
    - Uses DeepFace.represent(model_name=...) to compute embeddings
    - `infer()` finds the closest embedding and returns (name, distance)
    """

    def __init__(self, model_name: str = "Facenet512", db_path: str = "faces_db", threshold: float = 1.0) -> None:
        self.model_name = model_name
        self.db_path = Path(db_path)
        self.threshold = threshold

        self.db_path.mkdir(parents=True, exist_ok=True)

        self.embeddings: List[np.ndarray] = []
        self.labels: List[str] = []

        self._load_db()

    # ----------------- LOAD DB -----------------
    def _load_db(self) -> None:
        """Load all face images in db_path and compute embeddings."""
        for file in self.db_path.iterdir():
            if file.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue

            img = cv2.imread(str(file))
            if img is None:
                continue

            try:
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                rep = DeepFace.represent(
                    img_path=rgb,
                    model_name=self.model_name,
                    enforce_detection=False,
                )
                if isinstance(rep, list):
                    rep = rep[0]

                vec = np.array(rep["embedding"], dtype="float32")
                self.embeddings.append(vec)
                self.labels.append(file.stem)
            except Exception:
                # Skip any images that cause errors
                continue

    # ----------------- EMBED FACE -----------------
    def embed(self, face_bgr: np.ndarray) -> np.ndarray:
        """Return embedding vector for a cropped BGR face image."""
        rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        rep = DeepFace.represent(
            img_path=rgb,
            model_name=self.model_name,
            enforce_detection=False,
        )
        if isinstance(rep, list):
            rep = rep[0]
        return np.array(rep["embedding"], dtype="float32")

    # ----------------- INFER NAME -----------------
    def infer(self, face_bgr: np.ndarray) -> Tuple[str, float]:
        """
        Compare face to DB. Returns (name, distance).
        If no match or DB empty → ("Unknown", large_distance).
        """
        if face_bgr is None or face_bgr.size == 0:
            return "Unknown", 999.0

        # No database, nothing to match
        if not self.embeddings:
            return "Unknown", 999.0

        try:
            vec = self.embed(face_bgr)
        except Exception:
            return "Unknown", 999.0

        dists = [float(np.linalg.norm(vec - e)) for e in self.embeddings]
        idx = int(np.argmin(dists))
        best_dist = dists[idx]
        best_name = self.labels[idx]

        if best_dist > self.threshold:
            return "Unknown", best_dist

        return best_name, best_dist
