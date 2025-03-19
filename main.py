import streamlit as st
import cv2
import librosa
import numpy as np
import sounddevice as sd
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

# Load pre-trained models
TEXT_MODEL_PATH = "models/text_emotion_model.h5"
AUDIO_MODEL_PATH = "models/audio_emotion_model.h5"
FACE_MODEL_PATH = "models/image_emotion_model.h5"

text_model = load_model(TEXT_MODEL_PATH)
audio_model = load_model(AUDIO_MODEL_PATH)
face_model = load_model(FACE_MODEL_PATH)

# Load Tokenizer for Text Processing
TEXT_MAXLEN = 100
TEXT_VOCAB_SIZE = 20000
tokenizer = Tokenizer(num_words=TEXT_VOCAB_SIZE, oov_token="<OOV>")

# Define emotion labels
emotion_labels = ["Anger", "Happy", "Neutral", "Sad", "Surprise", "Contempt", "Disgust", "Fear"]

# Function to analyze text
def analyze_text(text):
    if not text or not isinstance(text, str):  # Handle empty or invalid input
        return "Unknown", 0.0

    # Ensure tokenizer has been trained
    if not tokenizer.word_index:  
        sample_texts = ["I am happy", "I am sad", "I am angry", "I feel neutral"]
        tokenizer.fit_on_texts(sample_texts)  # Fit on sample texts

    sequences = tokenizer.texts_to_sequences([text])

    if not sequences or sequences == [[]]:  # Handle empty sequences
        return "Unknown", 0.0

    text_input = pad_sequences(sequences, maxlen=TEXT_MAXLEN, dtype="int32", padding="post", truncating="post")

    prediction = text_model.predict(text_input)
    emotion = emotion_labels[np.argmax(prediction)]
    confidence = np.max(prediction)
    return emotion, confidence


# Function to analyze live audio
def analyze_audio_live(duration=5, sr=16000):
    st.write("🎤 Recording...")
    audio_data = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype='float32')
    sd.wait()
    
    # Extract features using librosa
    mfccs = librosa.feature.mfcc(y=audio_data.flatten(), sr=sr, n_mfcc=13)
    mfccs = np.resize(mfccs, (50, 13))
    
    prediction = audio_model.predict(np.expand_dims(mfccs, axis=0))
    emotion = emotion_labels[np.argmax(prediction)]
    confidence = np.max(prediction)
    return emotion, confidence

# Function to analyze live video using OpenCV
def analyze_video_live():
    cap = cv2.VideoCapture(0)
    st.write("📷 Analyzing live video... Press 'q' to exit.")

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            face_roi = gray[y:y+h, x:x+w]
            face_roi = cv2.resize(face_roi, (48, 48)) / 255.0
            face_roi = np.expand_dims(face_roi, axis=-1)
            face_roi = np.expand_dims(face_roi, axis=0)
            
            prediction = face_model.predict(face_roi)
            emotion = emotion_labels[np.argmax(prediction)]
            confidence = np.max(prediction)

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"{emotion} ({confidence:.2f})", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("Real-Time Emotion Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

# Streamlit UI
st.title("Multi-Modal Emotion AI")
input_type = st.radio("Select Input Type", ["Text", "Live Audio", "Live Video"])

if input_type == "Text":
    text_input = st.text_area("Enter text for emotion analysis")
    if st.button("Analyze"):
        emotion, confidence = analyze_text(text_input)
        st.write(f"📝 Detected Emotion: **{emotion}** (Confidence: {confidence:.2f})")

elif input_type == "Live Audio":
    if st.button("Record and Analyze"):
        emotion, confidence = analyze_audio_live()
        st.write(f"🎤 Detected Emotion: **{emotion}** (Confidence: {confidence:.2f})")

elif input_type == "Live Video":
    if st.button("Start Live Video Analysis"):
        analyze_video_live()
