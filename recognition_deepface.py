from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Dict, Any

import cv2
import numpy as np
from deepface import DeepFace


class RecognizerDeepFace:
    """
    Simple DeepFace-based face recognizer.

    - Recursively loads all image files under `db_path`
    - Filenames are used for metadata:

        Tom-Cruise_63_1.jpg  ->  name = "Tom Cruise", age = 63
        mr-beast_29_3.png    ->  name = "mr beast",  age = 29

    - Uses DeepFace.represent(model_name=...) to compute embeddings
    - infer() finds the closest embedding and returns (name, distance)
    """

    def __init__(
        self,
        model_name: str = "Facenet512",
        db_path: str = "faces_db",
        threshold: float = 1.0,
    ) -> None:
        self.model_name = model_name
        self.db_path = Path(db_path)
        self.threshold = threshold

        self.db_path.mkdir(parents=True, exist_ok=True)

        self.embeddings: List[np.ndarray] = []
        self.labels: List[str] = []                   # display names
        self.profiles: Dict[str, Dict[str, Any]] = {}  # name -> { "age": int | None }

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
            except Exception:
                # Skip any images that cause errors
                continue

            name, age = self._parse_label(file.stem)

            self.embeddings.append(vec)
            self.labels.append(name)

            # store / update simple profile (age)
            profile = self.profiles.get(name, {})
            if profile.get("age") is None and age is not None:
                profile["age"] = age
            self.profiles[name] = profile

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

    # ----------------- PROFILE LOOKUP -----------------
    def get_profile(self, name: str) -> Dict[str, Any]:
        """Return stored metadata for a name (currently just age)."""
        return self.profiles.get(name, {})
