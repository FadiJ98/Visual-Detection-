import cv2

def crop_face(img_bgr, box, size=160, pad=0.25):
    x1,y1,x2,y2 = box
    w = x2 - x1
    h = y2 - y1
    cx = x1 + w//2
    cy = y1 + h//2
    s  = int(max(w,h) * (1 + pad))
    x1n, y1n = max(cx - s//2, 0), max(cy - s//2, 0)
    x2n, y2n = min(cx + s//2, img_bgr.shape[1]-1), min(cy + s//2, img_bgr.shape[0]-1)
    face = img_bgr[y1n:y2n, x1n:x2n]
    if face.size == 0: return None
    return cv2.resize(face, (size, size))
