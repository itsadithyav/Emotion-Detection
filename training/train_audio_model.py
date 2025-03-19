import numpy as np
import tensorflow as tf
import os
import pandas as pd
import librosa
import cv2
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Conv1D, Conv2D, MaxPooling2D, Flatten, Dense, LSTM, Embedding, Dropout, Input

# Paths
MODEL_DIR = "models"
DATA_DIR = "data"
AUDIO_MODEL_PATH = os.path.join(MODEL_DIR, "audio_emotion_model.h5")
AUDIO_MAPPING_CSV = os.path.join(DATA_DIR, "audio", "file_mapping.csv")

# Parameters
AUDIO_SHAPE = (50, 13)
EMOTION_CATEGORIES = ["Anger", "Happy", "Neutral", "Sad", "Surprise", "Contempt", "Disgust", "Fear"]
NUM_CLASSES = len(EMOTION_CATEGORIES)
BATCH_SIZE = 32
EPOCHS = 10

# Ensure model directory exists
os.makedirs(MODEL_DIR, exist_ok=True)

# Load audio dataset
def load_audio(mapping_csv):
    print("🔉 Loading audio data using file mapping...")
    df = pd.read_csv(mapping_csv)
    
    if "NewPath" not in df.columns or "Emotions" not in df.columns:
        raise ValueError("❌ CSV file missing required columns: 'NewPath', 'Emotions'")

    audio_data, labels = [], []
    skipped_count = 0

    for _, row in df.iterrows():
        file_path, emotion_label = row["NewPath"], row["Emotions"].capitalize()

        if emotion_label not in EMOTION_CATEGORIES or not os.path.exists(file_path):
            print(f"⚠️ Skipping {file_path} (Invalid emotion: {emotion_label} or file missing)")
            skipped_count += 1
            continue

        try:
            audio, sr = librosa.load(file_path, sr=44100)
            mel_spec = librosa.power_to_db(librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=AUDIO_SHAPE[1]), ref=np.max)

            # Ensure correct shape (50, 13)
            if mel_spec.shape[1] < AUDIO_SHAPE[1]:  
                pad_width = AUDIO_SHAPE[1] - mel_spec.shape[1]
                mel_spec = np.pad(mel_spec, ((0, 0), (0, pad_width)), mode='constant')
            elif mel_spec.shape[1] > AUDIO_SHAPE[1]:  
                mel_spec = mel_spec[:, :AUDIO_SHAPE[1]]

            if mel_spec.shape[0] < AUDIO_SHAPE[0]:  
                pad_height = AUDIO_SHAPE[0] - mel_spec.shape[0]
                mel_spec = np.pad(mel_spec, ((0, pad_height), (0, 0)), mode='constant')
            elif mel_spec.shape[0] > AUDIO_SHAPE[0]:  
                mel_spec = mel_spec[:AUDIO_SHAPE[0], :]

            audio_data.append(mel_spec)
            labels.append(EMOTION_CATEGORIES.index(emotion_label))

        except Exception as e:
            print(f"❌ Error processing {file_path}: {e}")
            skipped_count += 1

    print(f"✅ Successfully loaded {len(audio_data)} audio samples (Skipped {skipped_count} files).")
    return np.array(audio_data, dtype="float32"), to_categorical(labels, NUM_CLASSES)

# Load datasets
X_audio, y_audio = load_audio(AUDIO_MAPPING_CSV)

# Train-test split
X_train_aud, X_test_aud, y_train_aud, y_test_aud = train_test_split(X_audio, y_audio, test_size=0.2, random_state=42)

print("✅ Data loading complete!")

# Define models
def create_audio_model():
    model = tf.keras.Sequential([
        Conv1D(32, 3, activation='relu', input_shape=AUDIO_SHAPE),
        LSTM(64),
        Dense(NUM_CLASSES, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Train models
if os.path.exists(AUDIO_MODEL_PATH):
    print("🔄 Loading existing audio model...")
    audio_model = load_model(AUDIO_MODEL_PATH)
else:
    print("🆕 Creating new audio model...")
    audio_model = create_audio_model()

audio_model.fit(X_train_aud, y_train_aud, validation_data=(X_test_aud, y_test_aud), epochs=EPOCHS, batch_size=BATCH_SIZE)
audio_model.save(AUDIO_MODEL_PATH)

print("✅ Training complete! Models saved.")

# Evaluate model performance
loss, accuracy = audio_model.evaluate(X_test_aud, y_test_aud, verbose=1)
print(f"✅ Model evaluation complete! Test Accuracy of Audio Model: {accuracy:.4f}")

