import streamlit as st
import cv2
import librosa
import torch
import numpy as np
import sounddevice as sd
from transformers import pipeline
from deepface import DeepFace
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Load models
sentiment_pipeline = pipeline("sentiment-analysis")
audio_model = load_model("audio_emotion_model.h5")  # Pretrained audio emotion model

# Emotion labels
emotions = ["angry", "happy", "neutral", "sad", "surprised"]

def analyze_text(text):
    result = sentiment_pipeline(text)[0]
    return result['label'], result['score']

def analyze_audio_live(duration=5, sr=22050):
    st.write("Recording...")
    audio_data = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype='float32')
    sd.wait()
    mfccs = librosa.feature.mfcc(y=audio_data.flatten(), sr=sr, n_mfcc=40)
    mfccs = np.mean(mfccs.T, axis=0)
    mfccs = np.expand_dims(mfccs, axis=0)
    prediction = audio_model.predict(mfccs)
    return emotions[np.argmax(prediction)], np.max(prediction)

def analyze_video_live():
    cap = cv2.VideoCapture(0)
    st.write("Analyzing live video...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        face_analysis = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        if face_analysis:
            emotion = face_analysis[0]['dominant_emotion']
            st.write(f"Detected Emotion: {emotion}")
    cap.release()

# Streamlit UI
st.title("Multi-Modal Emotion AI")
input_type = st.radio("Select Input Type", ["Text", "Live Audio", "Live Video"])

if input_type == "Text":
    text_input = st.text_area("Enter text for emotion analysis")
    if st.button("Analyze"):
        emotion, confidence = analyze_text(text_input)
        st.write(f"Detected Emotion: {emotion} (Confidence: {confidence:.2f})")

elif input_type == "Live Audio":
    if st.button("Record and Analyze"):
        emotion, confidence = analyze_audio_live()
        st.write(f"Detected Emotion: {emotion} (Confidence: {confidence:.2f})")

elif input_type == "Live Video":
    if st.button("Start Live Video Analysis"):
        analyze_video_live()
