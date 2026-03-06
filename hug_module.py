# hug_module.py

from transformers import pipeline
from PIL import Image
import cv2
import numpy as np
import mediapipe as mp
from y_module import model

emotion_pipeline = pipeline(
    "image-classification",
    model="dima806/facial_emotions_image_detection"
)

# MediaPipe — better for close-up, angled, multiple faces
mp_face = mp.solutions.face_detection
face_detector = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.2)


def detect_emotion(frame):
    results = []

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w = frame.shape[:2]

    mp_results = face_detector.process(rgb_frame)

    if not mp_results.detections:
        return results

    for detection in mp_results.detections:
        bbox = detection.location_data.relative_bounding_box

        # Convert relative coords to absolute pixels
        fx = max(0, int(bbox.xmin * w))
        fy = max(0, int(bbox.ymin * h))
        fw = int(bbox.width * w)
        fh = int(bbox.height * h)

        # Clamp to frame boundaries
        fx2 = min(w, fx + fw)
        fy2 = min(h, fy + fh)

        face_crop = frame[fy:fy2, fx:fx2]
        if face_crop.size == 0:
            continue

        rgb_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_crop)

        emotion_results = emotion_pipeline(pil_image)

        if emotion_results:
            label = emotion_results[0]["label"]
            score = emotion_results[0]["score"]
        else:
            label = "Unknown"
            score = 0.0

        results.append(((fx, fy, fw, fh), label, score))

    return results