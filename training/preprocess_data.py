import numpy as np
import tensorflow as tf
import pandas as pd
import librosa
import cv2
import os
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import re

# Define dataset paths
IMAGE_TRAIN_PATH = "AI Powered Task Optimizer/data/image/train"
IMAGE_TEST_PATH = "AI Powered Task Optimizer/data/image/test"
TEXT_DATASET_PATH = "AI Powered Task Optimizer/data/text/tweet_emotions.csv"
AUDIO_DATASET_PATH = "AI Powered Task Optimizer/data/audio/audio_speech_actors_01-24"

# Define parameters
IMAGE_SIZE = (48, 48)  # Target image size
NUM_CLASSES = 7  # 7 emotion categories
IMAGE_CHANNELS = 1  # Grayscale images
TEXT_MAXLEN = 100
TEXT_VOCAB_SIZE = 20000
AUDIO_SHAPE = (50, 13)

# Load and preprocess images
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

            img = cv2.resize(img, IMAGE_SIZE)
            img = np.expand_dims(img, axis=-1)
            img = img / 255.0
            
            image_data.append(img)
            labels.append(label)

    image_data = np.array(image_data, dtype="float32")
    labels = to_categorical(labels, num_classes=NUM_CLASSES)
    return image_data, labels

# Load and preprocess text
def load_text(text_path):
    df = pd.read_csv(text_path)
    texts, labels = df["content"].values, df["sentiment"].values
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)
    labels = tf.keras.utils.to_categorical(labels, num_classes=len(label_encoder.classes_))
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=TEXT_VOCAB_SIZE, oov_token="<OOV>")
    tokenizer.fit_on_texts(texts)
    text_sequences = tokenizer.texts_to_sequences(texts)
    text_data = tf.keras.preprocessing.sequence.pad_sequences(text_sequences, maxlen=TEXT_MAXLEN)
    return text_data, labels

# Load and preprocess audio
def prepare_audio_dataframe(audio_path):
    records = []

    for dirname, _, filenames in os.walk(audio_path):
        # Extract actor folder name (Actor_XX)
        actor_match = re.search(r"Actor_(\d+)", os.path.basename(dirname))
        
        if not actor_match:
            print(f"❌ No match for {os.path.basename(dirname)}")
            continue  # Skip non-actor folders
        
        for filename in filenames:
            if filename.endswith(".wav"):
                records.append([filename, os.path.join(dirname, filename), actor_match.group(0)])

    data = pd.DataFrame(records, columns=['filename', 'path', 'actor'])
    
    if data.empty:
        print("❌ No audio files found! Check dataset path.")
        return data  # Return empty DataFrame to prevent crashes

    print(f"✅ Found {len(data)} audio files from {data['actor'].nunique()} actors.")
    return data



def extract_emotion_label(file_path):
    emotion_map = {"01": 0, "02": 1, "03": 2, "04": 3, "05": 4, "06": 5, "07": 6}

    filename = os.path.basename(file_path)  # Extract filename

    # Match emotion codes from filenames (e.g., "03-01-06-02-02-01-12.wav")
    match = re.search(r"-(\d{2})-\d{2}\.wav$", filename)

    if match:
        emotion_code = match.group(1)  # Extracts "01", "02", etc.
        return emotion_map.get(emotion_code, -1)

    print(f"❌ No match for emotion in {filename}")
    return -1  #


def load_audio(audio_path):
    data = prepare_audio_dataframe(audio_path)
    
    if data.empty:
        print("No valid audio data loaded!")
        return np.array([]), np.array([])
    
    audio_data, labels = [], []
    
    for _, row in data.iterrows():
        file_path = row['path']
        emotion_label = extract_emotion_label(row['filename'])
        
        if emotion_label == -1:
            continue

        try:
            audio, sr = librosa.load(file_path, sr=16000)
            mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=13)
            mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
            mel_spec = np.resize(mel_spec, AUDIO_SHAPE)
            audio_data.append(mel_spec)
            labels.append(emotion_label)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    if not audio_data:
        print("No valid audio samples processed!")
        return np.array([]), np.array([])
    
    labels = to_categorical(labels, num_classes=NUM_CLASSES)
    return np.array(audio_data), labels
