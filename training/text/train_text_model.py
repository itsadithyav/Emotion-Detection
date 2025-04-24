import numpy as np
import tensorflow as tf
import os
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import LSTM, Embedding, Dense, Dropout, Bidirectional, Input, Attention, BatchNormalization, GlobalAveragePooling1D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
from tensorflow.keras.layers import LSTM, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.regularizers import l2
from tensorflow.keras.metrics import AUC, Precision, Recall

# Paths
MODEL_DIR = "models"
DATA_DIR = "data/text"
TEXT_MODEL_PATH = os.path.join(MODEL_DIR, "text_emotion_model.h5")
TOKENIZER_PATH = os.path.join(MODEL_DIR, "tokenizer.pkl")

# Parameters
TEXT_MAXLEN = 100
TEXT_VOCAB_SIZE = 20000
BATCH_SIZE = 32
EPOCHS = 100

# Ensure model directory exists
os.makedirs(MODEL_DIR, exist_ok=True)

# Load emotions list
with open(os.path.join(DATA_DIR, "emotions.txt"), "r") as file:
    emotion_list = file.read().splitlines()

# Filtered emotions for classification
selected_emotions = ['disgust', 'fear', 'joy', 'sadness', 'anger', 'neutral']

# Load dataset
df_train = pd.read_csv(os.path.join(DATA_DIR, "train.tsv"), sep='\t', header=None, names=['Text', 'Class', 'ID'])
df_val = pd.read_csv(os.path.join(DATA_DIR, "dev.tsv"), sep='\t', header=None, names=['Text', 'Class', 'ID'])
df_test = pd.read_csv(os.path.join(DATA_DIR, "test.tsv"), sep='\t', header=None, names=['Text', 'Class', 'ID'])

# Convert index to emotions
def idx2class_filtered(idx_list):
    return [emotion_list[int(i)] for i in idx_list.split(',') if emotion_list[int(i)] in selected_emotions]

df_train['Emotions'] = df_train['Class'].apply(idx2class_filtered)
df_val['Emotions'] = df_val['Class'].apply(idx2class_filtered)
df_test['Emotions'] = df_test['Class'].apply(idx2class_filtered)

# Multi-label binarization (One-hot encoding)
mlb = MultiLabelBinarizer(classes=selected_emotions)

y_train = mlb.fit_transform(df_train['Emotions'])
y_val = mlb.transform(df_val['Emotions'])
y_test = mlb.transform(df_test['Emotions'])

# Tokenization
if os.path.exists(TOKENIZER_PATH):  # Load tokenizer if exists
    with open(TOKENIZER_PATH, 'rb') as f:
        tokenizer = pickle.load(f)
else:  # Create new tokenizer
    tokenizer = Tokenizer(num_words=TEXT_VOCAB_SIZE, oov_token="<OOV>")
    tokenizer.fit_on_texts(df_train["Text"].astype(str))
    with open(TOKENIZER_PATH, 'wb') as f:
        pickle.dump(tokenizer, f)  # Save tokenizer for future use

X_train = pad_sequences(tokenizer.texts_to_sequences(df_train["Text"].astype(str)), maxlen=TEXT_MAXLEN)
X_val = pad_sequences(tokenizer.texts_to_sequences(df_val["Text"].astype(str)), maxlen=TEXT_MAXLEN)
X_test = pad_sequences(tokenizer.texts_to_sequences(df_test["Text"].astype(str)), maxlen=TEXT_MAXLEN)

print(f"✅ Loaded {len(X_train)} training samples, {len(X_val)} validation samples, {len(X_test)} test samples.")

# Define number of classes
NUM_CLASSES = len(selected_emotions)
print(f"✅ Number of emotion classes: {NUM_CLASSES}")

# Define improved model
def create_text_model():
    inputs = Input(shape=(TEXT_MAXLEN,))
    
    # Enhanced Embedding with regularization
    embedding = Embedding(
        TEXT_VOCAB_SIZE, 
        256, 
        input_length=TEXT_MAXLEN,
        embeddings_regularizer=l2(1e-5),
        trainable=True
    )(inputs)
    
    # Dropout after embedding
    embedding = Dropout(0.2)(embedding)
    
    # Deeper LSTM layers with residual connections
    lstm_out1 = Bidirectional(LSTM(256, return_sequences=True))(embedding)
    lstm_out1 = BatchNormalization()(lstm_out1)
    lstm_out1 = Dropout(0.3)(lstm_out1)
    
    lstm_out2 = Bidirectional(LSTM(128, return_sequences=True))(lstm_out1)
    lstm_out2 = BatchNormalization()(lstm_out2)
    lstm_out2 = Dropout(0.3)(lstm_out2)
    
    # Multi-head attention
    attention1 = Attention()([lstm_out2, lstm_out2])
    attention2 = Attention()([lstm_out1, lstm_out1])
    
    # Combine attention outputs
    concat_attention = tf.keras.layers.Concatenate()([
        GlobalAveragePooling1D()(attention1),
        GlobalAveragePooling1D()(attention2)
    ])
    
    # Enhanced Dense layers with residual connections
    dense1 = Dense(256, activation="relu", kernel_regularizer=l2(1e-4))(concat_attention)
    dense1 = BatchNormalization()(dense1)
    dense1 = Dropout(0.5)(dense1)
    
    dense2 = Dense(128, activation="relu", kernel_regularizer=l2(1e-4))(dense1)
    dense2 = BatchNormalization()(dense2)
    dense2 = Dropout(0.4)(dense2)
    
    # Output Layer with adjusted activation
    outputs = Dense(NUM_CLASSES, activation="sigmoid")(dense2)
    
    # Create and compile model with custom metrics
    model = Model(inputs, outputs)
    
    # Learning rate schedule
    initial_learning_rate = 0.001
    decay_steps = 1000
    decay_rate = 0.9
    learning_rate_schedule = ExponentialDecay(
        initial_learning_rate, decay_steps, decay_rate
    )
    
    # Optimizer with gradient clipping
    optimizer = Adam(
        learning_rate=learning_rate_schedule,
        clipnorm=1.0
    )
    
    # Add more metrics for better monitoring
    metrics = [
        'accuracy',
        AUC(name='auc'),
        Precision(name='precision'),
        Recall(name='recall')
    ]
    
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=metrics
    )
    
    return model

# Load or create model
if os.path.exists(TEXT_MODEL_PATH):
    print("🔄 Loading existing text model...")
    text_model = load_model(TEXT_MODEL_PATH)
else:
    print("🆕 Creating new text model...")
    text_model = create_text_model()

# Enhanced callbacks
callbacks = [
    EarlyStopping(
        monitor="val_auc",
        patience=15,
        restore_best_weights=True,
        verbose=1
    ),
    ModelCheckpoint(
        TEXT_MODEL_PATH,
        monitor="val_auc",
        save_best_only=True,
        verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )
]

# Convert dataset to TensorFlow dataset format for performance boost
train_data = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
val_data = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
test_data = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# Train model with class weights
history = text_model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    callbacks=callbacks,
    class_weight={i: 1.0 for i in range(NUM_CLASSES)}  # Adjust if classes are imbalanced
)

# Evaluate model with all metrics
metrics = text_model.evaluate(test_data, verbose=1)
print("\n✅ Model Evaluation Results:")
for metric_name, value in zip(text_model.metrics_names, metrics):
    print(f"{metric_name}: {value:.4f}")