# hug_module.py

from transformers import pipeline
from PIL import Image
import cv2
import numpy as np
import mediapipe as mp

emotion_pipeline = pipeline(
    "image-classification",
    model="dima806/facial_emotions_image_detection"
)

mp_face    = mp.solutions.face_detection
face_detector = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.2)


def detect_emotion(frame):
    """
    Detects all faces using MediaPipe and returns emotion per face.

    Returns:
        List of tuples: [((fx, fy, fw, fh), label, score), ...]
        Empty list [] if no faces found.
    """
    results   = []
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w      = frame.shape[:2]

    mp_results = face_detector.process(rgb_frame)

    if not mp_results.detections:
        return results

    for detection in mp_results.detections:
        bbox = detection.location_data.relative_bounding_box

        fx  = max(0, int(bbox.xmin * w))
        fy  = max(0, int(bbox.ymin * h))
        fw  = int(bbox.width  * w)
        fh  = int(bbox.height * h)
        fx2 = min(w, fx + fw)
        fy2 = min(h, fy + fh)

        face_crop = frame[fy:fy2, fx:fx2]
        if face_crop.size == 0:
            continue

        rgb_crop  = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_crop)

        emotion_results = emotion_pipeline(pil_image)

        label = str(emotion_results[0]["label"])   if emotion_results else "neutral"
        score = float(emotion_results[0]["score"])  if emotion_results else 0.0

        results.append(((fx, fy, fw, fh), label, score))

    return results
