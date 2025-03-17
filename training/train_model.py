import numpy as np
import tensorflow as tf
import os
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv1D, Conv2D, MaxPooling2D, Flatten, Dense, LSTM, Embedding, Bidirectional, TimeDistributed, Dropout, Input, Concatenate

# Paths
MODEL_PATH = "AI Powered Task Optimizer/models/emotion_model.h5"

# Dataset paths
IMAGE_TRAIN_PATH = "AI Powered Task Optimizer/data/image/train"
IMAGE_TEST_PATH = "AI Powered Task Optimizer/data/image/test"
TEXT_DATASET_PATH = "AI Powered Task Optimizer/data/text/tweet_emotions.csv"
AUDIO_DATASET_PATH = "AI Powered Task Optimizer/data/audio/audio_speech_actors_01-24"

# Parameters
IMAGE_SHAPE = (48, 48, 1)  # Grayscale images
TEXT_MAXLEN = 100
TEXT_VOCAB_SIZE = 20000
AUDIO_SHAPE = (50, 13)  # Corrected audio input shape
NUM_CLASSES = 7  # Ensure consistency in class output
BATCH_SIZE = 32
EPOCHS = 10

# Load preprocessed data
from preprocess_data import load_images, load_text, load_audio

X_train_images, y_train_images = load_images(IMAGE_TRAIN_PATH)
X_test_images, y_test_images = load_images(IMAGE_TEST_PATH)
X_text, y_text = load_text(TEXT_DATASET_PATH)
X_audio, y_audio = load_audio(AUDIO_DATASET_PATH)

# Train-test split
X_train_text, X_test_text, y_train_text, y_test_text = train_test_split(X_text, y_text, test_size=0.2, random_state=42)
X_train_audio, X_test_audio, y_train_audio, y_test_audio = train_test_split(X_audio, y_audio, test_size=0.2, random_state=42)

# Ensure labels are one-hot encoded
y_train_images = to_categorical(y_train_images, NUM_CLASSES)
y_test_images = to_categorical(y_test_images, NUM_CLASSES)
y_train_text = to_categorical(y_train_text, NUM_CLASSES)
y_test_text = to_categorical(y_test_text, NUM_CLASSES)
y_train_audio = to_categorical(y_train_audio, NUM_CLASSES)
y_test_audio = to_categorical(y_test_audio, NUM_CLASSES)

X_train = [X_train_images, X_train_text, X_train_audio]
y_train = y_train_images  # Unified label for all
X_test = [X_test_images, X_test_text, X_test_audio]
y_test = y_test_images

# 🔹 Load existing model or create a new one
if os.path.exists(MODEL_PATH):
    print("✅ Model found! Loading existing model...")
    model = load_model(MODEL_PATH)
else:
    print("🚀 No existing model found. Creating a new one...")

    # CNN for Images
    image_input = Input(shape=IMAGE_SHAPE)
    x = Conv2D(32, (3, 3), activation='relu')(image_input)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    image_output = Dense(NUM_CLASSES, activation='softmax')(x)
    image_model = tf.keras.Model(inputs=image_input, outputs=image_output)

    # LSTM for Text
    text_input = Input(shape=(TEXT_MAXLEN,))
    y = Embedding(TEXT_VOCAB_SIZE, 128, input_length=TEXT_MAXLEN)(text_input)
    y = Bidirectional(LSTM(64, return_sequences=True))(y)
    y = Dropout(0.3)(y)
    y = LSTM(32)(y)
    y = Dense(128, activation='relu')(y)
    text_output = Dense(NUM_CLASSES, activation='softmax')(y)
    text_model = tf.keras.Model(inputs=text_input, outputs=text_output)

    # CNN+LSTM for Audio
    audio_input = Input(shape=AUDIO_SHAPE)
    z = Conv1D(filters=32, kernel_size=3, activation='relu')(audio_input)
    z = LSTM(64, return_sequences=True)(z)
    z = Flatten()(z)
    z = Dense(128, activation='relu')(z)
    audio_output = Dense(NUM_CLASSES, activation='softmax')(z)
    audio_model = tf.keras.Model(inputs=audio_input, outputs=audio_output)

    # Merge models
    combined = Concatenate()([image_model.output, text_model.output, audio_model.output])
    final_output = Dense(128, activation='relu')(combined)
    final_output = Dense(NUM_CLASSES, activation='softmax')(final_output)

    model = tf.keras.Model(inputs=[image_input, text_input, audio_input], outputs=final_output)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 🔹 Train the model
model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE
)

# 🔹 Save model
model.save(MODEL_PATH)
print("✅ Model training complete and saved!")