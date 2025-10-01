import mediapipe as mp
import numpy as np

class FaceDetectorMP:
    def __init__(self, min_conf=0.5):
        self.detector = mp.solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=min_conf)

    def detect(self, bgr_img):
        rgb = bgr_img[:,:,::-1]
        res = self.detector.process(rgb)
        boxes = []
        if res.detections:
            h, w = bgr_img.shape[:2]
            for d in res.detections:
                bb = d.location_data.relative_bounding_box
                x1 = max(int(bb.xmin * w), 0)
                y1 = max(int(bb.ymin * h), 0)
                x2 = min(int((bb.xmin + bb.width) * w), w-1)
                y2 = min(int((bb.ymin + bb.height) * h), h-1)
                boxes.append((x1,y1,x2,y2))
        return boxes
