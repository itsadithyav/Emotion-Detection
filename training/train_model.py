import numpy as np
import tensorflow as tf
import os
import pandas as pd
import librosa
import cv2
from sklearn.preprocessing import LabelEncoder
import re
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Conv1D, Conv2D, MaxPooling2D, Flatten, Dense, LSTM, Embedding, Bidirectional, Dropout, Input, Concatenate

# Paths
MODEL_PATH = "Emotion-Detection-with-GUI-for-face-voice-and-text/models/emotion_model.h5"

# Dataset paths
IMAGE_TRAIN_PATH = "Emotion-Detection-with-GUI-for-face-voice-and-text/data/image/train"
TEXT_DATASET_PATH = "Emotion-Detection-with-GUI-for-face-voice-and-text/data/text/tweet_emotions.csv"
AUDIO_DATASET_PATH = "Emotion-Detection-with-GUI-for-face-voice-and-text/data/audio/audio_speech_actors_01-24"

# Parameters
IMAGE_SHAPE = (48, 48, 1)
TEXT_MAXLEN = 100
TEXT_VOCAB_SIZE = 20000
AUDIO_SHAPE = (50, 13)
NUM_CLASSES = 7
BATCH_SIZE = 32
EPOCHS = 10


# 🔹 Load and preprocess images
def load_images(image_path):
    emotion_labels = sorted(os.listdir(image_path))
    label_map = {emotion: i for i, emotion in enumerate(emotion_labels)}
    image_data, labels = [], []

    for emotion, label in label_map.items():
        emotion_folder = os.path.join(image_path, emotion)
        if not os.path.isdir(emotion_folder): 
            continue

        for img_name in os.listdir(emotion_folder):
            img_path = os.path.join(emotion_folder, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, IMAGE_SHAPE[:2]) / 255.0
            img = np.expand_dims(img, axis=-1)
            image_data.append(img)
            labels.append(label)

    image_data = np.array(image_data, dtype="float32")
    labels = to_categorical(labels, num_classes=NUM_CLASSES)
    return image_data, labels


# 🔹 Load and preprocess text
def load_text(text_path):
    df = pd.read_csv(text_path)

    # Extract text and labels
    texts = df["content"].astype(str).values
    labels = df["sentiment"].astype(str).values

    # Ensure both have the same length
    min_size = min(len(texts), len(labels))
    texts, labels = texts[:min_size], labels[:min_size]

    # Encode labels
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)
    labels = to_categorical(labels, num_classes=len(label_encoder.classes_))

    # Tokenize and pad text sequences
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=TEXT_VOCAB_SIZE, oov_token="<OOV>")
    tokenizer.fit_on_texts(texts)
    text_sequences = tokenizer.texts_to_sequences(texts)
    text_data = tf.keras.preprocessing.sequence.pad_sequences(text_sequences, maxlen=TEXT_MAXLEN)

    # Ensure final sizes match
    min_size = min(len(text_data), len(labels))
    text_data, labels = text_data[:min_size], labels[:min_size]

    print(f"✅ Final X_text size: {len(text_data)}")
    print(f"✅ Final y_text size: {len(labels)}")

    return text_data, labels


# 🔹 Load and preprocess audio
def load_audio(audio_path):
    records = []
    for dirname, _, filenames in os.walk(audio_path):
        actor_match = re.search(r"Actor_(\d+)", os.path.basename(dirname))
        if not actor_match:
            continue
        for filename in filenames:
            if filename.endswith(".wav"):
                records.append([filename, os.path.join(dirname, filename)])

    data = pd.DataFrame(records, columns=['filename', 'path'])
    if data.empty:
        print("❌ No audio files found! Check dataset path.")
        return np.array([]), np.array([])

    print(f"✅ Found {len(data)} audio files.")

    audio_data, labels = [], []
    for _, row in data.iterrows():
        file_path = row['path']
        try:
            audio, sr = librosa.load(file_path, sr=16000)
            mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=13)
            mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
            mel_spec = np.resize(mel_spec, AUDIO_SHAPE)
            audio_data.append(mel_spec)
            labels.append(0)  # Dummy label (FIX THIS)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    labels = to_categorical(labels, num_classes=NUM_CLASSES)
    return np.array(audio_data), labels


# 🔹 Load datasets
X_images, y_images = load_images(IMAGE_TRAIN_PATH)
X_text, y_text = load_text(TEXT_DATASET_PATH)
X_audio, y_audio = load_audio(AUDIO_DATASET_PATH)

# Ensure all datasets have the same size
min_size = min(len(X_images), len(X_text), len(X_audio), len(y_images), len(y_text), len(y_audio))

# Trim datasets to match the smallest size
X_images, y_images = X_images[:min_size], y_images[:min_size]
X_text, y_text = X_text[:min_size], y_text[:min_size]
X_audio, y_audio = X_audio[:min_size], y_audio[:min_size]

# 🔹 Train-test split
X_train_images, X_test_images, y_train_images, y_test_images = train_test_split(X_images, y_images, test_size=0.2, random_state=42)
X_train_text, X_test_text, y_train_text, y_test_text = train_test_split(X_text, y_text, test_size=0.2, random_state=42)
X_train_audio, X_test_audio, y_train_audio, y_test_audio = train_test_split(X_audio, y_audio, test_size=0.2, random_state=42)

# 🔹 Load or create model
if os.path.exists(MODEL_PATH):
    print("✅ Loading existing model...")
    model = load_model(MODEL_PATH)
else:
    print("🚀 Creating a new model...")
    image_input = Input(shape=IMAGE_SHAPE)
    text_input = Input(shape=(TEXT_MAXLEN,))
    audio_input = Input(shape=AUDIO_SHAPE)

    x = Conv2D(32, (3, 3), activation='relu')(image_input)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)
    image_output = Dense(NUM_CLASSES, activation='softmax')(x)

    y = Embedding(TEXT_VOCAB_SIZE, 128, input_length=TEXT_MAXLEN)(text_input)
    y = LSTM(32)(y)
    text_output = Dense(NUM_CLASSES, activation='softmax')(y)

    z = Conv1D(filters=32, kernel_size=3, activation='relu')(audio_input)
    z = LSTM(64)(z)
    audio_output = Dense(NUM_CLASSES, activation='softmax')(z)

    combined = Concatenate()([image_output, text_output, audio_output])
    final_output = Dense(NUM_CLASSES, activation='softmax')(combined)

    model = tf.keras.Model(inputs=[image_input, text_input, audio_input], outputs=final_output)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 🔹 Train model
model.fit([X_train_images, X_train_text, X_train_audio], y_train_images, validation_data=([X_test_images, X_test_text, X_test_audio], y_test_images), epochs=EPOCHS, batch_size=BATCH_SIZE)

# 🔹 Save model
model.save(MODEL_PATH)
print("✅ Model training complete and saved!")
