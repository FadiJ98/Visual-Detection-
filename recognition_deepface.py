from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Dict, Any

import cv2
import numpy as np


class RecognizerDeepFace:
    """
    Lightweight face recognizer (no DeepFace models).

    - Recursively loads all face images under `db_path`
    - Filenames store metadata:
        Tom-Cruise_63_1.jpg  ->  name = "Tom Cruise", age = 63
        mr-beast_29_3.png    ->  name = "mr beast",  age = 29

    - For each DB image we build a simple embedding:
        - convert to gray
        - resize to 128x128
        - flatten to a 1D vector
    - infer() compares the uploaded face to all DB embeddings
      using L2 distance and returns (name, distance).
    """

    def __init__(
        self,
        model_name: str = "dummy",          # kept for compatibility
        db_path: str = "faces_db",
        threshold: float = 2000.0,          # tune for “Unknown”
    ) -> None:
        self.db_path = Path(db_path)
        self.threshold = threshold

        self.db_path.mkdir(parents=True, exist_ok=True)

        self.embeddings: List[np.ndarray] = []
        self.labels: List[str] = []                    # display names
        self.profiles: Dict[str, Dict[str, Any]] = {}  # name -> {"age": int | None}

        self._load_db()

    # ----------------- LABEL PARSER -----------------
    @staticmethod
    def _parse_label(stem: str) -> Tuple[str, int | None]:
        """
        Parse filename stem like 'Tom-Cruise_63_1' into:
            name = 'Tom Cruise'
            age  = 63

        If age missing or invalid -> age = None.
        """
        parts = stem.split("_")

        base = parts[0] if parts else stem
        name = base.replace("-", " ").strip()

        age: int | None = None
        if len(parts) >= 2:
            try:
                age = int(parts[1])
            except ValueError:
                age = None

        return name, age

    # ----------------- EMBEDDING HELPER -----------------
    @staticmethod
    def _to_embedding(img_bgr: np.ndarray) -> np.ndarray:
        """
        Turn a BGR face image into a small numeric embedding.
        (Very simple: grayscale + resize + flatten.)
        """
        if img_bgr is None or img_bgr.size == 0:
            return np.zeros((128 * 128,), dtype="float32")

        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        face_resized = cv2.resize(gray, (128, 128), interpolation=cv2.INTER_AREA)
        vec = face_resized.flatten().astype("float32")
        # normalize to reduce brightness differences
        norm = np.linalg.norm(vec) + 1e-8
        return vec / norm

    # ----------------- LOAD DB -----------------
    def _load_db(self) -> None:
        """Recursively load all face images under db_path and compute embeddings."""
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

        for file in self.db_path.rglob("*"):
            if not file.is_file() or file.suffix.lower() not in exts:
                continue

            img = cv2.imread(str(file))
            if img is None:
                continue

            emb = self._to_embedding(img)

            name, age = self._parse_label(file.stem)

            self.embeddings.append(emb)
            self.labels.append(name)

            profile = self.profiles.get(name, {})
            if profile.get("age") is None and age is not None:
                profile["age"] = age
            self.profiles[name] = profile

    # ----------------- INFER NAME -----------------
    def infer(self, face_bgr: np.ndarray) -> Tuple[str, float]:
        """
        Compare face to DB. Returns (name, distance).
        If no match or DB empty → ("Unknown", large_distance).
        """
        if face_bgr is None or face_bgr.size == 0:
            return "Unknown", 9999.0

        if not self.embeddings:
            return "Unknown", 9999.0

        vec = self._to_embedding(face_bgr)

        dists = [float(np.linalg.norm(vec - e)) for e in self.embeddings]
        idx = int(np.argmin(dists))
        best_dist = dists[idx]
        best_name = self.labels[idx]

        if best_dist > self.threshold:
            return "Unknown", best_dist

        return best_name, best_dist

    # ----------------- PROFILE LOOKUP -----------------
    def get_profile(self, name: str) -> Dict[str, Any]:
        """Return stored metadata for a name (currently just age)."""
        return self.profiles.get(name, {})
