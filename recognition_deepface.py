from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Dict, Any

import cv2
import numpy as np
from deepface import DeepFace


def parse_filename(stem: str) -> Tuple[str | None, int | None]:
    """
    Expected pattern: safe-name_age_index

    Examples:
        tom-cruise_24_1  ->  name='Tom Cruise', age=24
        mr-beast_29_3    ->  name='Mr Beast',   age=29

    Returns:
        (display_name, age) or (None, None) if pattern not matched.
    """
    parts = stem.split("_")
    if len(parts) < 2:
        return None, None

    raw_name = parts[0]            # "tom-cruise"
    name = raw_name.replace("-", " ").title()  # -> "Tom Cruise"

    age = None
    try:
        age = int(parts[1])
    except Exception:
        age = None

    return name, age


class RecognizerDeepFace:
    """
    Simple DeepFace-based face recognizer.

    - Loads images from `db_path` (including subfolders)
    - Uses DeepFace.represent(model_name=...) to compute embeddings
    - `infer()` finds the closest embedding and returns (name, distance)
    - Filenames can encode metadata: safe-name_age_index
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
        self.labels: List[str] = []  # display names
        # name -> {"age": int | None}
        self.person_meta: Dict[str, Dict[str, Any]] = {}

        self._load_db()

    # ----------------- LOAD DB -----------------
    def _load_db(self) -> None:
        """Load all face images in db_path and compute embeddings."""
        self.embeddings.clear()
        self.labels.clear()
        self.person_meta.clear()

        valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

        # support subfolders under faces_db
        for file in self.db_path.rglob("*"):
            if not file.is_file() or file.suffix.lower() not in valid_exts:
                continue

            img = cv2.imread(str(file))
            if img is None:
                continue

            # parse from filename
            display_name, age = parse_filename(file.stem)
            if display_name is None:
                # fallback: raw stem
                display_name = file.stem

            # init metadata for this name
            if display_name not in self.person_meta:
                self.person_meta[display_name] = {"age": None}
            if age is not None:
                self.person_meta[display_name]["age"] = age

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
                self.labels.append(display_name)
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
        """
        Return stored metadata for a person.
        Example: {"age": 24}
        """
        return self.person_meta.get(name, {})
