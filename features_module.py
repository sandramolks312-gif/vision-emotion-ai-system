# FEATURE EXTRACTION MODULE

import pandas as pd

# Consistent emotion mapping
EMOTION_MAP = {
    "happy": 0,
    "sad": 1,
    "angry": 2,
    "fear": 3,
    "surprise": 4,
    "disgust": 5,
    "neutral": 6
}

def encode_emotion(emotion_label):
    return EMOTION_MAP.get(emotion_label.lower(), 6)  # default to neutral if unknown

def save_features(person_count, phone, emotion_label, emotion_score, activity_level, label):
    emotion_encoded = encode_emotion(emotion_label)
    data = {
        "person_count": [person_count],
        "phone": [phone],
        "emotion_encoded": [emotion_encoded],  # NEW — which emotion
        "emotion_score": [emotion_score],       # existing — how confident
        "activity_level": [activity_level],
        "label": [label]
    }
    df = pd.DataFrame(data)
    try:
        df.to_csv(r"C:\PRACTICE DS PROJECT\AI PROJECTS\MAIN PROJECT\AllDetails\MAIN.csv", mode="a", header=False, index=False)
    except:
        df.to_csv(r"C:\PRACTICE DS PROJECT\AI PROJECTS\MAIN PROJECT\AllDetails\MAIN.csv", index=False)