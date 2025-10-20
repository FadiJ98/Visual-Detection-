# recognition_deepface.py
from dataclasses import dataclass
from typing import Dict, Tuple, List, Union, Optional, Literal
import os
import cv2
import numpy as np
from deepface import DeepFace

NDArray = np.ndarray
BBox = Tuple[int, int, int, int]  # (x1, y1, x2, y2)

# -------------------- helpers --------------------

def _l2_normalize(v: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    n = np.linalg.norm(v) + eps
    return v / n

def _to_bgr(image_source: Union[str, NDArray]) -> NDArray:
    """
    Accepts a file path or a BGR/RGB ndarray and returns a BGR ndarray.
    """
    if isinstance(image_source, str):
        img = cv2.imread(image_source, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Could not read image from path: {image_source}")
        return img
    if not isinstance(image_source, np.ndarray):
        raise TypeError("image_source must be a file path (str) or a numpy array.")
    return image_source

def _crop_raw(bgr: NDArray, box: BBox) -> Optional[NDArray]:
    """Raw crop without square/pad; use only if you don’t want face_align.crop_face."""
    if bgr is None or bgr.size == 0 or box is None:
        return None
    x1, y1, x2, y2 = map(int, box)
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    return bgr[y1:y2, x1:x2].copy()

# -------------------- data classes --------------------

@dataclass
class RecognitionResult:
    name: str
    distance: float
    box: BBox

# -------------------- main class --------------------

class RecognizerDeepFace:
    """
    DeepFace-based face recognition for static images.
    - Builds a DeepFace model once.
    - Uses L2-normalized embeddings.
    - Nearest neighbor over an in-memory + on-disk .npy DB.
    """

    # Reasonable starting thresholds (tune with your data)
    DEFAULT_THRESH = {
        "Facenet": 0.90,      # L2 (DeepFace default style)
        "Facenet512": 0.40,   # L2 over *normalized* vecs tends to be ~0.35–0.55
        "VGG-Face": 0.75,     # cosine distance (if you switch metric)
        "ArcFace": 0.85,      # cosine distance-ish; tune
    }

    def __init__(
        self,
        model_name: str = "Facenet512",
        db_dir: str = "data/embeddings",
        normalize: bool = True,
        metric: Literal["l2", "cosine"] = "l2"
    ):
        self.model_name = model_name
        self.db_dir = db_dir
        self.normalize = normalize
        self.metric = metric  # "l2" or "cosine"
        os.makedirs(db_dir, exist_ok=True)

        # Build once; skip DeepFace’s own detector since we crop
        self.model = DeepFace.build_model(model_name)
        self._memory_db: Dict[str, np.ndarray] = {}
        self.load_db()

    # ------------- embeddings -------------

    def _represent(self, face_bgr: NDArray) -> Optional[np.ndarray]:
        """Low-level call to DeepFace.represent; returns np.float32 vector."""
        if face_bgr is None or face_bgr.size == 0:
            return None
        # DeepFace accepts numpy arrays; pass detector_backend="skip"
        vec = DeepFace.represent(
            img_path=face_bgr,              # numpy array
            model=self.model,
            enforce_detection=False,
            detector_backend="skip"
        )
        if isinstance(vec, list) and len(vec) and isinstance(vec[0], dict):
            vec = np.array(vec[0]["embedding"], dtype=np.float32)
        else:
            vec = np.array(vec, dtype=np.float32)
        return vec

    def embed(self, face_bgr: NDArray) -> Optional[np.ndarray]:
        vec = self._represent(face_bgr)
        if vec is None:
            return None
        if self.normalize:
            vec = _l2_normalize(vec.astype(np.float32))
        return vec

    def embed_many(self, faces_bgr: List[NDArray]) -> List[np.ndarray]:
        out = []
        for f in faces_bgr:
            v = self.embed(f)
            if v is not None:
                out.append(v)
        return out

    # ------------- DB I/O -------------

    def save_embedding(self, name: str, face_bgr: NDArray) -> Tuple[bool, str]:
        vec = self.embed(face_bgr)
        if vec is None:
            return False, "No face embedding produced."
        np.save(os.path.join(self.db_dir, f"{name}.npy"), vec.astype(np.float32))
        self._memory_db[name] = vec
        return True, f"Saved embedding for '{name}'."

    def save_samples(self, name: str, vectors: List[np.ndarray]) -> Tuple[bool, str]:
        if not vectors:
            return False, "No vectors provided."
        stack = np.stack(vectors, axis=0).astype(np.float32)
        mean_vec = stack.mean(axis=0)
        if self.normalize:
            mean_vec = _l2_normalize(mean_vec)
        np.save(os.path.join(self.db_dir, f"{name}.npy"), mean_vec.astype(np.float32))
        self._memory_db[name] = mean_vec
        return True, f"Saved mean embedding for '{name}' ({len(vectors)} samples)."

    def load_db(self):
        self._memory_db.clear()
        for f in os.listdir(self.db_dir):
            if f.endswith(".npy"):
                name = f[:-4]
                vec = np.load(os.path.join(self.db_dir, f)).astype(np.float32)
                if self.normalize:
                    vec = _l2_normalize(vec)
                self._memory_db[name] = vec

    # ------------- metrics -------------

    def _distance(self, a: np.ndarray, b: np.ndarray) -> float:
        if self.metric == "cosine":
            # cosine distance = 1 - cosine similarity
            denom = (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)
            cos_sim = float(np.dot(a, b) / denom)
            return 1.0 - cos_sim
        # default L2
        return float(np.linalg.norm(a - b))

    # ------------- inference -------------

    def infer(self, face_bgr: NDArray, thresh: Optional[float] = None) -> Tuple[str, float]:
        """
        Return (name, distance) for a cropped face image.
        """
        vec = self.embed(face_bgr)
        if vec is None or not self._memory_db:
            return ("Unknown", 1.0)

        # pick default threshold if none provided
        if thresh is None:
            thresh = self.DEFAULT_THRESH.get(self.model_name, 0.45)

        best_name, best_dist = "Unknown", 1.0
        for name, ref in self._memory_db.items():
            dist = self._distance(vec, ref)
            if dist < best_dist:
                best_dist, best_name = dist, name
        return (best_name if best_dist <= float(thresh) else "Unknown", best_dist)

    # --------- IMAGE-LEVEL HELPERS (works with your MediaPipe detector) ---------

    @staticmethod
    def crop(bgr: NDArray, box: BBox) -> Optional[NDArray]:
        """Raw crop helper (kept for API compatibility)."""
        return _crop_raw(bgr, box)

    @staticmethod
    def draw_result(bgr: NDArray, res: RecognitionResult, color_ok=(0, 200, 0), color_unk=(0, 0, 255)) -> None:
        x1, y1, x2, y2 = res.box
        is_known = (res.name != "Unknown")
        color = color_ok if is_known else color_unk
        cv2.rectangle(bgr, (x1, y1), (x2, y2), color, 2)
        label = f"{res.name} ({res.distance:.2f})" if is_known else f"Unknown ({res.distance:.2f})"
        cv2.putText(bgr, label, (x1, max(0, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

    def recognize_in_image(
        self,
        image_source: Union[str, NDArray],
        detector,
        thresh: Optional[float] = None,
        annotate: bool = True,
        use_square_crop: bool = False,
        square_size: int = 160,
        square_pad: float = 0.25,
    ) -> Tuple[List[RecognitionResult], NDArray]:
        """
        Detect and recognize all faces in a static image.
        - detector must expose detect(bgr_img)->List[BBox]
        - If use_square_crop=True, will use face_align.crop_face for stable embeddings.
        """
        bgr = _to_bgr(image_source)
        boxes = detector.detect(bgr)
        results: List[RecognitionResult] = []

        if use_square_crop:
            # import here to avoid circular import at module top
            from face_align import crop_face
        for box in boxes:
            if use_square_crop:
                face = crop_face(bgr, box, size=square_size, pad=square_pad)
            else:
                face = self.crop(bgr, box)
            name, dist = self.infer(face, thresh=thresh)
            results.append(RecognitionResult(name=name, distance=dist, box=box))

        out = bgr.copy()
        if annotate:
            for r in results:
                self.draw_result(out, r)
        return results, out

    def enroll_from_image(
        self,
        image_source: Union[str, NDArray],
        name: str,
        detector,
        use_largest: bool = True,
        use_square_crop: bool = True,
        square_size: int = 160,
        square_pad: float = 0.25,
    ) -> Tuple[bool, str]:
        """
        Enroll from a single image. If multiple faces, uses the largest by default.
        """
        bgr = _to_bgr(image_source)
        boxes = detector.detect(bgr)
        if not boxes:
            return False, "No face detected to enroll."

        box = max(boxes, key=lambda b: (b[2]-b[0]) * (b[3]-b[1])) if use_largest else boxes[0]
        if use_square_crop:
            from face_align import crop_face
            face = crop_face(bgr, box, size=square_size, pad=square_pad)
        else:
            face = self.crop(bgr, box)
        return self.save_embedding(name, face)
