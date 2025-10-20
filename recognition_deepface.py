from deepface import DeepFace
import numpy as np
import os, cv2
from typing import Dict, Tuple, List, Union
from dataclasses import dataclass

NDArray = np.ndarray
BBox = Tuple[int, int, int, int]  # (x1, y1, x2, y2)

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

@dataclass
class RecognitionResult:
    name: str
    distance: float
    box: BBox

class RecognizerDeepFace:
    """
    DeepFace-based face recognition with helpers for static images.
    - Uses L2-normalized embeddings and simple nearest-neighbor.
    """
    def __init__(self, model_name: str = "Facenet512", db_dir: str = "data/embeddings", normalize: bool = True):
        self.model_name = model_name
        self.db_dir = db_dir
        self.normalize = normalize
        os.makedirs(db_dir, exist_ok=True)
        self.model = DeepFace.build_model(model_name)
        self._memory_db: Dict[str, np.ndarray] = {}
        self.load_db()

    def embed(self, face_bgr: NDArray) -> np.ndarray | None:
        if face_bgr is None or face_bgr.size == 0:
            return None
        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        vec = DeepFace.represent(img_path=face_rgb, model=self.model, enforce_detection=False)
        # DeepFace returns a list[dict] in many cases:
        if isinstance(vec, list) and len(vec) and isinstance(vec[0], dict):
            vec = np.array(vec[0]["embedding"], dtype=np.float32)
        else:
            vec = np.array(vec, dtype=np.float32)
        if self.normalize:
            vec = _l2_normalize(vec)
        return vec

    def save_embedding(self, name: str, face_bgr: NDArray) -> Tuple[bool, str]:
        vec = self.embed(face_bgr)
        if vec is None:
            return False, "No face embedding produced."
        np.save(os.path.join(self.db_dir, f"{name}.npy"), vec)
        self._memory_db[name] = vec
        return True, f"Saved embedding for '{name}'."

    def load_db(self):
        self._memory_db.clear()
        for f in os.listdir(self.db_dir):
            if f.endswith(".npy"):
                name = f[:-4]
                vec = np.load(os.path.join(self.db_dir, f)).astype(np.float32)
                if self.normalize:
                    vec = _l2_normalize(vec)
                self._memory_db[name] = vec

    def infer(self, face_bgr: NDArray, thresh: float = 0.45) -> Tuple[str, float]:
        """
        Return (name, distance) for a cropped face image.
        """
        vec = self.embed(face_bgr)
        if vec is None or not self._memory_db:
            return ("Unknown", 1.0)

        best_name, best_dist = "Unknown", 1.0
        for name, ref in self._memory_db.items():
            dist = float(np.linalg.norm(vec - ref))
            if dist < best_dist:
                best_dist, best_name = dist, name
        return (best_name if best_dist <= thresh else "Unknown", best_dist)

    # --------- IMAGE-LEVEL HELPERS (use with your MediaPipe detector) ---------

    @staticmethod
    def crop(bgr: NDArray, box: BBox) -> NDArray:
        x1, y1, x2, y2 = box
        return bgr[y2 if y2 < y1 else y1 : y1 if y2 < y1 else y2,
                   x2 if x2 < x1 else x1 : x1 if x2 < x1 else x2].copy()

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
        thresh: float = 0.45,
        annotate: bool = True
    ) -> Tuple[List[RecognitionResult], NDArray]:
        """
        Detect and recognize all faces in a static image.
        - image_source: file path or ndarray (BGR/RGB). Returns list of results + (optionally) annotated BGR image.
        - detector: an object with method `detect(bgr_img)->List[BBox]` (e.g., FaceDetectorMP).
        """
        bgr = _to_bgr(image_source)
        boxes = detector.detect(bgr)
        results: List[RecognitionResult] = []
        for box in boxes:
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
        use_largest: bool = True
    ) -> Tuple[bool, str]:
        """
        Enroll a face from a static image. If multiple faces, uses the largest box by default.
        """
        bgr = _to_bgr(image_source)
        boxes = detector.detect(bgr)
        if not boxes:
            return False, "No face detected to enroll."

        box = max(boxes, key=lambda b: (b[2]-b[0]) * (b[3]-b[1])) if use_largest else boxes[0]
        face = self.crop(bgr, box)
        return self.save_embedding(name, face)
