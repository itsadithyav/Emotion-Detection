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
IMAGE_DATASET_PATH = os.path.join(DATA_DIR, "image")

# Parameters
IMAGE_SHAPE = (48, 48, 1)
EMOTION_CATEGORIES = ["Anger", "Happy", "Neutral", "Sad", "Surprise", "Contempt", "Disgust", "Fear"]
NUM_CLASSES = len(EMOTION_CATEGORIES)
BATCH_SIZE = 32
EPOCHS = 10

# Ensure model directory exists
os.makedirs(MODEL_DIR, exist_ok=True)

# Load and preprocess images
def load_images(image_folder):
    print("\U0001F4C2 Loading images from dataset...")
    image_data, labels = [], []
    
    for emotion in EMOTION_CATEGORIES:
        emotion_folder = os.path.join(image_folder, emotion)  # Directly use folder name
        if not os.path.isdir(emotion_folder):
            print(f"⚠️ Folder {emotion_folder} not found, skipping.")
            continue

        print(f"\U0001F4C2 Processing folder: {emotion}")

        for img_name in os.listdir(emotion_folder):
            img_path = os.path.join(emotion_folder, img_name)
            
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"❌ Failed to load {img_path}")
                continue
            
            img = cv2.resize(img, IMAGE_SHAPE[:2]) / 255.0
            img = np.expand_dims(img, axis=-1)
            
            image_data.append(img)
            labels.append(EMOTION_CATEGORIES.index(emotion))  # Use folder name as label
    
    print(f"✅ Loaded {len(image_data)} images.")
    return np.array(image_data, dtype="float32"), to_categorical(labels, NUM_CLASSES)

# Load train dataset
X_train_img, y_train_img = load_images(os.path.join(IMAGE_DATASET_PATH, "train"))

# Load test dataset
X_test_img, y_test_img = load_images(os.path.join(IMAGE_DATASET_PATH, "test"))

print(f"✅ Train samples: {len(X_train_img)}, Test samples: {len(X_test_img)}")
print("✅ Data loading complete!")

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

# Train models
if os.path.exists(IMAGE_MODEL_PATH):
    print("🔄 Loading existing image model...")
    image_model = load_model(IMAGE_MODEL_PATH)
else:
    print("🆕 Creating new image model...")
    image_model = create_image_model()

image_model.fit(X_train_img, y_train_img, validation_data=(X_test_img, y_test_img), epochs=EPOCHS, batch_size=BATCH_SIZE)
image_model.save(IMAGE_MODEL_PATH)
    
print("✅ Training complete! Models saved.")

# Evaluate model performance
loss, accuracy = image_model.evaluate(X_test_img, y_test_img, verbose=1)
print(f"✅ Model evaluation complete! Test Accuracy of Image Model: {accuracy:.4f}")

