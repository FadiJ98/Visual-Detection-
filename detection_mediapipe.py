import mediapipe as mp
import numpy as np
import cv2
from typing import List, Tuple, Union

NDArray = np.ndarray
BBox = Tuple[int, int, int, int]  # (x1, y1, x2, y2)

def _clamp(v, lo, hi):
    return max(lo, min(hi, v))

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
    # Heuristic: if last dim is 3 and looks like RGB, convert to BGR.
    # We won't try to guess; assume it's already BGR. If you know it's RGB, convert before passing.
    return image_source

class FaceDetectorMP:
    """
    MediaPipe face detector with image-friendly helpers.
    """
    def __init__(self, min_conf: float = 0.5, model_selection: int = 0, pad: float = 0.08):
        self.detector = mp.solutions.face_detection.FaceDetection(
            model_selection=model_selection,
            min_detection_confidence=min_conf
        )
        self.pad = float(pad)

    def detect(self, bgr_img: NDArray) -> List[BBox]:
        """
        Detect faces on a BGR image and return list of (x1,y1,x2,y2) boxes.
        """
        if bgr_img is None or bgr_img.size == 0:
            return []
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

    def detect_on_image(self, image_source: Union[str, NDArray]) -> List[BBox]:
        """
        Convenience: accept path or ndarray, return face boxes.
        """
        bgr = _to_bgr(image_source)
        return self.detect(bgr)

    @staticmethod
    def draw_boxes(bgr_img: NDArray, boxes: List[BBox], color=(0, 255, 255), thickness: int = 2) -> NDArray:
        """
        Return a copy of image with rectangles drawn.
        """
        out = bgr_img.copy()
        for (x1, y1, x2, y2) in boxes:
            cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
        return out
