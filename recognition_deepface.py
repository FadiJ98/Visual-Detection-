from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Dict, Any

import cv2
import numpy as np
from deepface import DeepFace


def parse_filename(stem: str) -> Tuple[str | None, int | None, str | None]:
    """
    Parse file name of form: safe-name_age_gender_index
    Example: mr-beast_29_male_1  -> ("Mr Beast", 29, "male")
    If pattern doesn't match, returns (None, None, None).
    """
    parts = stem.split("_")
    if len(parts) < 3:
        return None, None, None

    raw_name = parts[0]  # "mr-beast"
    # convert to display name "Mr Beast"
    name = raw_name.replace("-", " ").title()

    age = None
    try:
        age = int(parts[1])
    except Exception:
        age = None

    gender = parts[2].lower()
    return name, age, gender


class RecognizerDeepFace:
    """
    Simple DeepFace-based face recognizer.

    - Looks inside `db_path`
    - Accepts nested folders, e.g.:

        faces_db/
          mr-beast/
            mr-beast_29_male_1.jpg
            mr-beast_29_male_2.jpg
          tom-cruise/
            tom-cruise_62_male_1.jpg

    - Uses DeepFace.represent(model_name=...) to compute embeddings
    - `infer()` finds the closest embedding and returns (name, distance)
    - `get_profile(name)` returns stored age/gender (from filenames)
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
        self.person_meta: Dict[str, Dict[str, Any]] = {}  # name -> {age, gender}

        self._load_db()

    # ----------------- LOAD DB -----------------
    def _load_db(self) -> None:
        """Load all face images in db_path and compute embeddings."""
        self.embeddings.clear()
        self.labels.clear()
        self.person_meta.clear()

        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

        for file in self.db_path.rglob("*"):
            if not file.is_file() or file.suffix.lower() not in exts:
                continue

            img = cv2.imread(str(file))
            if img is None:
                continue

            # Parse name/age/gender from filename if possible
            name, age, gender = parse_filename(file.stem)

            # If parsing failed, fall back to folder name or file stem
            if name is None:
                if file.parent != self.db_path:
                    name = file.parent.name
                else:
                    name = file.stem

            # Store meta if we got it
            if name not in self.person_meta:
                self.person_meta[name] = {"age": None, "gender": None}
            if age is not None:
                self.person_meta[name]["age"] = age
            if gender is not None:
                self.person_meta[name]["gender"] = gender

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
                self.labels.append(name)
            except Exception:
                # Skip any images that cause errors
                continue

    # Allow manual reload after new faces are added
    def reload(self) -> None:
        self._load_db()

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

    # ----------------- PROFILE LOOKUP -----------------
    def get_profile(self, name: str) -> Dict[str, Any]:
        """
        Returns stored metadata for a person:
            {"age": int | None, "gender": str | None}
        """
        return self.person_meta.get(name, {})
