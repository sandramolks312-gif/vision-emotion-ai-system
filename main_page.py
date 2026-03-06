# MAIN UI PAGE COMBINE ALL FUNCTION

import streamlit as st
import cv2
import tempfile
import joblib
from y_module import detect_objects
from hug_module import detect_emotion

# Load trained ML model
ml_model = joblib.load("fusion_model.pkl")

st.title("Smart Video Emotion + Object Detection System")

uploaded_video = st.file_uploader("Upload Video", type=["mp4", "avi"])

if uploaded_video is not None:

    # Save uploaded file temporarily
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_video.read())

    cap = cv2.VideoCapture(tfile.name)

    stframe = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Resize frame for speed
        frame = cv2.resize(frame, (640, 480))

        # YOLO detection
        person_count, phone = detect_objects(frame)

        # Emotion detection
        emotion_label, emotion_score = detect_emotion(frame)

        # Example activity level
        activity_level = 2

        # ML prediction
        features = [[person_count, phone, emotion_score, activity_level]]
        prediction = ml_model.predict(features)

        # Display text on frame
        cv2.putText(frame, f"Persons: {person_count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.putText(frame, f"Emotion: {emotion_label}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        cv2.putText(frame, f"Behavior: {prediction[0]}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        stframe.image(frame, channels="BGR")

    cap.release()