# MAIN UI PAGE - VIDEO + IMAGE SUPPORT

import streamlit as st
import cv2
import numpy as np
import tempfile
import joblib
from y_module import detect_objects
from hug_module import detect_emotion

# ─────────────────────────────────────────────
# Load model + label encoder
# ─────────────────────────────────────────────
ml_model      = joblib.load("fusion_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")

st.title("Smart Video Emotion + Object Detection System")


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def box_centre(box):
    x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
    return (x1 + x2) // 2, (y1 + y2) // 2


def face_centre(face_result):
    fx, fy, fw, fh = face_result[0]
    return fx + fw // 2, fy + fh // 2


def match_emotion_to_person(person_box, emotion_results):
    if not emotion_results:
        return "neutral", 0.0
    pcx, pcy   = box_centre(person_box)
    best_label = "neutral"
    best_score = 0.0
    best_dist  = float("inf")
    for face_result in emotion_results:
        fcx, fcy = face_centre(face_result)
        dist = ((pcx - fcx) ** 2 + (pcy - fcy) ** 2) ** 0.5
        if dist < best_dist:
            best_dist  = dist
            best_label = face_result[1]
            best_score = face_result[2]
    return best_label, best_score


def get_dominant_emotion(emotion_results):
    if not emotion_results:
        return "neutral", 0.0
    best = max(emotion_results, key=lambda item: item[2])
    return best[1], best[2]


def predict_behaviour_for_person(person_box, phone_count, emotion_results, activity_level=2):
    emotion_label, emotion_score = match_emotion_to_person(person_box, emotion_results)
    features = np.array([[
        float(1),
        float(phone_count),
        float(emotion_score),
        float(activity_level)
    ]], dtype=np.float64)
    raw_pred  = ml_model.predict(features)
    behaviour = label_encoder.inverse_transform(raw_pred)[0]
    return str(behaviour)


def predict_behaviour_overall(person_boxes, phone_count, emotion_results, activity_level=2):
    _, emotion_score = get_dominant_emotion(emotion_results)
    features = np.array([[
        float(len(person_boxes)),
        float(phone_count),
        float(emotion_score),
        float(activity_level)
    ]], dtype=np.float64)
    raw_pred  = ml_model.predict(features)
    behaviour = label_encoder.inverse_transform(raw_pred)[0]
    return str(behaviour)


def draw_frame(frame, person_boxes, emotion_results):
    """
    Draws a clean bounding box per person.
    Label is drawn INSIDE the top of the box so it never
    overlaps with adjacent boxes even when people stand close together.
    """
    for idx, pbox in enumerate(person_boxes):
        x1, y1, x2, y2, conf = pbox

        emotion_label, emotion_score = match_emotion_to_person(pbox, emotion_results)

        # ── Bounding box ──────────────────────────────────────────────
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # ── Label inside the top-left of the box ─────────────────────
        # Two lines to avoid long text running outside the box:
        #   Line 1: "P1  happy"
        #   Line 2: "(92%)"
        line1 = f"P{idx + 1}  {emotion_label}"
        line2 = f"({emotion_score:.0%})"

        font       = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.50
        thickness  = 1

        (lw1, lh1), _ = cv2.getTextSize(line1, font, font_scale, thickness)
        (lw2, lh2), _ = cv2.getTextSize(line2, font, font_scale, thickness)

        pad     = 4
        strip_h = lh1 + lh2 + pad * 3   # height of background strip
        strip_w = max(lw1, lw2) + pad * 2

        # Clamp strip inside frame
        strip_x2 = min(frame.shape[1], x1 + strip_w)
        strip_y2 = min(frame.shape[0], y1 + strip_h)

        # Dark green background strip inside top of box
        cv2.rectangle(frame, (x1, y1), (strip_x2, strip_y2), (0, 180, 0), -1)

        # White text on green background
        cv2.putText(frame, line1,
                    (x1 + pad, y1 + pad + lh1),
                    font, font_scale, (255, 255, 255), thickness)
        cv2.putText(frame, line2,
                    (x1 + pad, y1 + pad * 2 + lh1 + lh2),
                    font, font_scale, (255, 255, 255), thickness)

    return frame


# ─────────────────────────────────────────────────────────────────────────────
# MODE SELECTOR
# ─────────────────────────────────────────────────────────────────────────────
mode = st.radio("Select Input Type", ["Video", "Image"], horizontal=True)

# ─────────────────────────────────────────────────────────────────────────────
# VIDEO MODE
# ─────────────────────────────────────────────────────────────────────────────
if mode == "Video":
    uploaded_video = st.file_uploader("Upload Video", type=["mp4", "avi"])

    if uploaded_video is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_video.read())

        cap     = cv2.VideoCapture(tfile.name)
        stframe = st.empty()

        m1, m2, m3, m4 = st.columns(4)
        p_box  = m1.empty()
        ph_box = m2.empty()
        em_box = m3.empty()
        bh_box = m4.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (640, 480))

            person_boxes, phone_count = detect_objects(frame)
            emotion_results           = detect_emotion(frame)
            overall_behaviour         = predict_behaviour_overall(person_boxes, phone_count, emotion_results)
            dominant_label, _         = get_dominant_emotion(emotion_results)

            frame = draw_frame(frame, person_boxes, emotion_results)
            stframe.image(frame, channels="BGR")

            p_box.metric("Persons",   len(person_boxes))
            ph_box.metric("Phones",   phone_count)
            em_box.metric("Emotion",  dominant_label)
            bh_box.metric("Behavior", overall_behaviour)

        cap.release()

# ─────────────────────────────────────────────────────────────────────────────
# IMAGE MODE
# ─────────────────────────────────────────────────────────────────────────────
elif mode == "Image":
    uploaded_image = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])

    if uploaded_image is not None:
        file_bytes = np.frombuffer(uploaded_image.read(), np.uint8)
        frame      = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        frame      = cv2.resize(frame, (640, 480))

        person_boxes, phone_count = detect_objects(frame)
        emotion_results           = detect_emotion(frame)

        frame = draw_frame(frame, person_boxes, emotion_results)
        st.image(frame, channels="BGR", caption="Detection Result")

        # ── Per-person result summary under the image ─────────────────
        if person_boxes:
            st.subheader("Result Summary")
            for idx, pbox in enumerate(person_boxes):
                emotion_label, emotion_score = match_emotion_to_person(pbox, emotion_results)
                behaviour = predict_behaviour_for_person(pbox, phone_count, emotion_results)

                col1, col2, col3 = st.columns(3)
                col1.metric(f"Person {idx + 1}", "")
                col2.metric("Emotion",  f"{emotion_label} ({emotion_score:.0%})")
                col3.metric("Behavior", behaviour)
                st.divider()
        else:
            st.info("No persons detected.")