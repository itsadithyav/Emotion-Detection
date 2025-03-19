import streamlit as st
import cv2
import librosa
import numpy as np
import sounddevice as sd
import tensorflow as tf
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

# Load trained model
MODEL_PATH = "models/emotion_model.keras"
model = load_model(MODEL_PATH)  # Loads in the new Keras format

# Load Tokenizer for Text Processing
TEXT_MAXLEN = 100
TEXT_VOCAB_SIZE = 20000
tokenizer = Tokenizer(num_words=TEXT_VOCAB_SIZE, oov_token="<OOV>")

# Define emotion labels
emotion_labels = ["Angry", "Happy", "Neutral", "Sad", "Surprised"]

# Function to analyze text
def analyze_text(text):
    if not text or not isinstance(text, str):  # Handle empty or invalid input
        return "Unknown", 0.0

    sequences = tokenizer.texts_to_sequences([text])

    if not sequences or sequences == [[]]:  # Handle empty sequences
        return "Unknown", 0.0

    text_input = pad_sequences(sequences, maxlen=TEXT_MAXLEN, dtype="int32", padding="post", truncating="post")
    
    # Ensure input shape matches model's expected shape
    text_input = np.array(text_input).reshape(1, TEXT_MAXLEN)

    # Create dummy inputs for face & audio
    dummy_face = np.zeros((1, 48, 48, 1))
    dummy_audio = np.zeros((1, 50, 13))

    prediction = model.predict([dummy_face, text_input, dummy_audio])
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
    mfccs = np.resize(mfccs, (50, 13))  # Reshape to match model input
    
    # Dummy inputs for text & face
    dummy_face = np.zeros((1, 48, 48, 1))
    dummy_text = np.zeros((1, TEXT_MAXLEN))
    
    prediction = model.predict([dummy_face, dummy_text, np.expand_dims(mfccs, axis=0)])
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
            face_roi = cv2.resize(face_roi, (48, 48)) / 255.0  # Normalize
            face_roi = np.expand_dims(face_roi, axis=-1)
            face_roi = np.expand_dims(face_roi, axis=0)
            
            # Dummy inputs for text & audio
            dummy_text = np.zeros((1, TEXT_MAXLEN))
            dummy_audio = np.zeros((1, 50, 13))
            
            # Predict emotion
            prediction = model.predict([face_roi, dummy_text, dummy_audio])
            emotion = emotion_labels[np.argmax(prediction)]
            confidence = np.max(prediction)

            # Display result on video
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
