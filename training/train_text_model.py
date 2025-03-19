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
TEXT_MODEL_PATH = os.path.join(MODEL_DIR, "text_emotion_model.h5")
TEXT_DATA_PATH = os.path.join(DATA_DIR, "text", "combined_emotion.csv")

# Parameters
TEXT_MAXLEN = 100
TEXT_VOCAB_SIZE = 20000
EMOTION_CATEGORIES = ["Anger", "Happy", "Neutral", "Sad", "Surprise", "Contempt", "Disgust", "Fear"]
NUM_CLASSES = len(EMOTION_CATEGORIES)
BATCH_SIZE = 32
EPOCHS = 10

# Ensure model directory exists
os.makedirs(MODEL_DIR, exist_ok=True)

# Load text dataset
def load_text(text_file):
    print(f"\U0001F4C2 Loading text data from {text_file}...")
    df = pd.read_csv(text_file)
    
    missing_sentences = df['sentence'].isnull().sum()
    missing_emotions = df['emotion'].isnull().sum()
    
    if missing_sentences > 0:
        print(f"⚠️ Warning: {missing_sentences} missing sentences found and skipped.")
    if missing_emotions > 0:
        print(f"⚠️ Warning: {missing_emotions} missing emotions found and skipped.")
    
    df = df.dropna(subset=['sentence', 'emotion'])
    
    label_encoder = LabelEncoder()
    labels = to_categorical(label_encoder.fit_transform(df["emotion"]), NUM_CLASSES)
    
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=TEXT_VOCAB_SIZE, oov_token="<OOV>")
    tokenizer.fit_on_texts(df["sentence"].astype(str))
    text_data = tf.keras.preprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences(df["sentence"].astype(str)), maxlen=TEXT_MAXLEN)
    
    print(f"✅ Loaded {len(text_data)} text samples.")
    return text_data, labels


# Load datasets
X_text, y_text = load_text(TEXT_DATA_PATH)

# Train-test split
X_train_txt, X_test_txt, y_train_txt, y_test_txt = train_test_split(X_text, y_text, test_size=0.2, random_state=42)

print("✅ Data loading complete!")

# Define models
def create_text_model():
    model = tf.keras.Sequential([
        Embedding(TEXT_VOCAB_SIZE, 128, input_length=TEXT_MAXLEN),
        LSTM(32),
        Dense(NUM_CLASSES, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Train text model
if os.path.exists(TEXT_MODEL_PATH):
    print("🔄 Loading existing text model...")
    text_model = load_model(TEXT_MODEL_PATH)
else:
    print("🆕 Creating new text model...")
    text_model = create_text_model()

text_model.fit(X_train_txt, y_train_txt, validation_data=(X_test_txt, y_test_txt), epochs=EPOCHS, batch_size=BATCH_SIZE)
text_model.save(TEXT_MODEL_PATH)

print("✅ Training complete! Models saved.")

# Evaluate model performance
loss, accuracy = text_model.evaluate(X_test_txt, y_test_txt, verbose=1)
print(f"✅ Model evaluation complete! Test Accuracy of Text Model: {accuracy:.4f}")
