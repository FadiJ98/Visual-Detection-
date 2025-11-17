import mediapipe as mp
import numpy as np
import cv2
from typing import List, Tuple, Union, Optional

NDArray = np.ndarray
BBox = Tuple[int, int, int, int]  # (x1, y1, x2, y2)


def _clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


def _ensure_bgr(img: NDArray) -> NDArray:
    """
    Make sure the image is BGR (3 channels).
    """
    if img is None or img.size == 0:
        return img
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img  # assume already BGR


def _to_bgr(image_source: Union[str, NDArray]) -> NDArray:
    """
    Accept a file path or ndarray and return a BGR ndarray.
    """
    if isinstance(image_source, str):
        img = cv2.imread(image_source, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"Could not read image from path: {image_source}")
        return _ensure_bgr(img)
    if not isinstance(image_source, np.ndarray):
        raise TypeError("image_source must be a file path (str) or a numpy array.")
    return _ensure_bgr(image_source)


class FaceDetectorMP:
    """
    MediaPipe face detector with helpers for images.
    """

    def __init__(self, min_conf: float = 0.5, model_selection: int = 0, pad: float = 0.08):
        self.detector = mp.solutions.face_detection.FaceDetection(
            model_selection=model_selection,
            min_detection_confidence=min_conf,
        )
        self.pad = float(pad)

    def close(self):
        if hasattr(self.detector, "close"):
            self.detector.close()

    def detect(self, bgr_img: NDArray) -> List[BBox]:
        """
        Detect faces on a BGR image and return list of (x1, y1, x2, y2).
        """
        if bgr_img is None or bgr_img.size == 0:
            return []

        bgr_img = _ensure_bgr(bgr_img)
        rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        res = self.detector.process(rgb)

        boxes: List[BBox] = []
        if res.detections:
            h, w = bgr_img.shape[:2]
            for d in res.detections:
                bb = d.location_data.relative_bounding_box
                px = self.pad
                x1 = _clamp(int((bb.xmin - px) * w), 0, w - 1)
                y1 = _clamp(int((bb.ymin - px) * h), 0, h - 1)
                x2 = _clamp(int((bb.xmin + bb.width + px) * w), 0, w - 1)
                y2 = _clamp(int((bb.ymin + bb.height + px) * h), 0, h - 1)
                if x2 > x1 and y2 > y1:
                    boxes.append((x1, y1, x2, y2))
        return boxes

    def detect_with_scores(self, bgr_img: NDArray) -> List[Tuple[BBox, float]]:
        """
        Return [(bbox, score)] for debugging/tuning thresholds.
        """
        if bgr_img is None or bgr_img.size == 0:
            return []

        bgr_img = _ensure_bgr(bgr_img)
        rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        res = self.detector.process(rgb)

        out: List[Tuple[BBox, float]] = []
        if res.detections:
            h, w = bgr_img.shape[:2]
            for d in res.detections:
                bb = d.location_data.relative_bounding_box
                px = self.pad
                x1 = _clamp(int((bb.xmin - px) * w), 0, w - 1)
                y1 = _clamp(int((bb.ymin - px) * h), 0, h - 1)
                x2 = _clamp(int((bb.xmin + bb.width + px) * w), 0, w - 1)
                y2 = _clamp(int((bb.ymin + bb.height + px) * h), 0, h - 1)
                if x2 > x1 and y2 > y1:
                    score = float(d.score[0]) if d.score else 0.0
                    out.append(((x1, y1, x2, y2), score))
        return out

    def best_face(self, bgr_img: NDArray) -> Optional[BBox]:
        """
        Return the highest-score face box or None.
        """
        dets = self.detect_with_scores(bgr_img)
        if not dets:
            return None
        return max(dets, key=lambda x: x[1])[0]

    def detect_on_image(self, image_source: Union[str, NDArray]) -> List[BBox]:
        """
        Convenience: accept path or ndarray, return face boxes.
        """
        bgr = _to_bgr(image_source)
        return self.detect(bgr)

    @staticmethod
    def draw_boxes(
        bgr_img: NDArray,
        boxes: List[BBox],
        color=(0, 255, 0),
        thickness: int = 2,
    ) -> NDArray:
        out = bgr_img.copy()
        for (x1, y1, x2, y2) in boxes:
            cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
        return out
