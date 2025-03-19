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
IMAGE_MODEL_PATH = os.path.join(MODEL_DIR, "image_emotion_model.h5")
TEXT_MODEL_PATH = os.path.join(MODEL_DIR, "text_emotion_model.h5")
AUDIO_MODEL_PATH = os.path.join(MODEL_DIR, "audio_emotion_model.h5")
IMAGE_DATASET_PATH = os.path.join(DATA_DIR, "image")
TEXT_TRAIN_PATH = os.path.join(DATA_DIR, "text", "train.csv")
TEXT_TEST_PATH = os.path.join(DATA_DIR, "text", "test.csv")
AUDIO_MAPPING_CSV = os.path.join(DATA_DIR, "audio", "file_mapping.csv")

# Parameters
IMAGE_SHAPE = (48, 48, 1)
TEXT_MAXLEN = 100
TEXT_VOCAB_SIZE = 20000
AUDIO_SHAPE = (50, 13)
EMOTION_CATEGORIES = ["Anger", "Happy", "Neutral", "Sad", "Surprise", "Contempt", "Disgust", "Fear"]
NUM_CLASSES = len(EMOTION_CATEGORIES)
BATCH_SIZE = 32
EPOCHS = 10

# Ensure model directory exists
os.makedirs(MODEL_DIR, exist_ok=True)

# Load and preprocess images
def load_images(image_folder):
    print("📂 Loading images from dataset...")
    image_data, labels = [], []
    
    for folder in sorted(os.listdir(image_folder)):
        folder_path = os.path.join(image_folder, folder)
        if not os.path.isdir(folder_path):
            continue
        
        print(f"📂 Processing folder: {folder}")
        
        for img_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_name)
            
            emotion_label = next((e for e in EMOTION_CATEGORIES if e.lower() in img_name.lower()), None)
            if emotion_label is None:
                print(f"⚠️ Skipping {img_name} (No recognizable emotion in filename)")
                continue
            
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"❌ Failed to load {img_path}")
                continue
            
            img = cv2.resize(img, IMAGE_SHAPE[:2]) / 255.0
            img = np.expand_dims(img, axis=-1)
            
            image_data.append(img)
            labels.append(EMOTION_CATEGORIES.index(emotion_label))
    
    print(f"✅ Loaded {len(image_data)} images.")
    return np.array(image_data, dtype="float32"), to_categorical(labels, NUM_CLASSES)

# Load text dataset
def load_text(text_file):
    print(f"📂 Loading text data from {text_file}...")
    df = pd.read_csv(text_file)
    label_encoder = LabelEncoder()
    labels = to_categorical(label_encoder.fit_transform(df["emotion"]), NUM_CLASSES)
    
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=TEXT_VOCAB_SIZE, oov_token="<OOV>")
    tokenizer.fit_on_texts(df["text"].astype(str))
    text_data = tf.keras.preprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences(df["text"].astype(str)), maxlen=TEXT_MAXLEN)
    
    print(f"✅ Loaded {len(text_data)} text samples.")
    return text_data, labels

# Load audio dataset
def load_audio(mapping_csv):
    print("📂 Loading audio data using file mapping...")
    df = pd.read_csv(mapping_csv)
    if "NewPath" not in df.columns or "Emotions" not in df.columns:
        raise ValueError("❌ CSV file missing required columns: 'NewPath', 'Emotions'")
    
    audio_data, labels = [], []
    for _, row in df.iterrows():
        file_path, emotion_label = row["NewPath"], row["Emotions"].capitalize()
        
        if emotion_label not in EMOTION_CATEGORIES or not os.path.exists(file_path):
            print(f"⚠️ Skipping {file_path} (Invalid emotion or file missing)")
            continue
        
        try:
            audio, sr = librosa.load(file_path, sr=44100)
            mel_spec = librosa.power_to_db(librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=13), ref=np.max)
            mel_spec = mel_spec[:AUDIO_SHAPE[0], :AUDIO_SHAPE[1]]
            
            audio_data.append(mel_spec)
            labels.append(EMOTION_CATEGORIES.index(emotion_label))
        except Exception as e:
            print(f"❌ Error processing {file_path}: {e}")
    
    print(f"✅ Successfully loaded {len(audio_data)} audio samples.")
    return np.array(audio_data, dtype="float32"), to_categorical(labels, NUM_CLASSES)

# Load datasets
X_images, y_images = load_images(IMAGE_DATASET_PATH)
X_train_txt, y_train_txt = load_text(TEXT_TRAIN_PATH)
X_test_txt, y_test_txt = load_text(TEXT_TEST_PATH)
X_audio, y_audio = load_audio(AUDIO_MAPPING_CSV)

# Display dataset shapes
print(f"🔉 Audio data shape: {X_audio.shape}, Labels shape: {y_audio.shape}")
print(f"🗒️ Text data shape: {X_train_txt.shape}, Labels shape: {y_train_txt.shape}")
print(f"📷 Image data shape: {X_images.shape}, Labels shape: {y_images.shape}")

# Train-test split
X_train_img, X_test_img, y_train_img, y_test_img = train_test_split(X_images, y_images, test_size=0.2, random_state=42)
X_train_aud, X_test_aud, y_train_aud, y_test_aud = train_test_split(X_audio, y_audio, test_size=0.2, random_state=42)

# Define models
def create_image_model():
    model = tf.keras.Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=IMAGE_SHAPE),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(NUM_CLASSES, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def create_text_model():
    model = tf.keras.Sequential([
        Embedding(TEXT_VOCAB_SIZE, 128, input_length=TEXT_MAXLEN),
        LSTM(32),
        Dense(NUM_CLASSES, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def create_audio_model():
    model = tf.keras.Sequential([
        Conv1D(32, 3, activation='relu', input_shape=AUDIO_SHAPE),
        LSTM(64),
        Dense(NUM_CLASSES, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Train models
if not os.path.exists(IMAGE_MODEL_PATH):
    image_model = create_image_model()
    image_model.fit(X_train_img, y_train_img, validation_data=(X_test_img, y_test_img), epochs=EPOCHS, batch_size=BATCH_SIZE)
    image_model.save(IMAGE_MODEL_PATH)

if not os.path.exists(TEXT_MODEL_PATH):
    text_model = create_text_model()
    text_model.fit(X_train_txt, y_train_txt, validation_data=(X_test_txt, y_test_txt), epochs=EPOCHS, batch_size=BATCH_SIZE)
    text_model.save(TEXT_MODEL_PATH)

if not os.path.exists(AUDIO_MODEL_PATH):
    audio_model = create_audio_model()
    audio_model.fit(X_train_aud, y_train_aud, validation_data=(X_test_aud, y_test_aud), epochs=EPOCHS, batch_size=BATCH_SIZE)
    audio_model.save(AUDIO_MODEL_PATH)

print("✅ Training complete! Models saved.")
