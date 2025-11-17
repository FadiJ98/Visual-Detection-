import cv2
import numpy as np
from typing import Tuple, Optional

BBox = Tuple[int, int, int, int]  # (x1, y1, x2, y2)


def crop_face(
    img_bgr: np.ndarray,
    box: BBox,
    size: int = 160,
    pad: float = 0.25
) -> Optional[np.ndarray]:
    """
    Take a BGR image and a face bounding box, return a square
    cropped + resized face (BGR) or None if something is wrong.
    """
    if img_bgr is None or img_bgr.size == 0 or box is None:
        return None

    x1, y1, x2, y2 = map(int, box)
    h, w = img_bgr.shape[:2]

    face_w = x2 - x1
    face_h = y2 - y1
    if face_w <= 0 or face_h <= 0:
        return None

    # Center + padded square crop
    cx = x1 + face_w // 2
    cy = y1 + face_h // 2
    side = int(max(face_w, face_h) * (1.0 + pad))

    x1n = max(cx - side // 2, 0)
    y1n = max(cy - side // 2, 0)
    x2n = min(cx + side // 2, w - 1)
    y2n = min(cy + side // 2, h - 1)

    face = img_bgr[y1n:y2n, x1n:x2n]
    if face.size == 0:
        return None

    return cv2.resize(face, (size, size))
