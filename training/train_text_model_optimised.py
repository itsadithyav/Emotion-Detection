import os
# Set performance-oriented environment variables
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
os.environ['TF_GPU_THREAD_COUNT'] = '1'
os.environ['TF_USE_CUDNN_BATCHNORM_SPATIAL_PERSISTENT'] = '1'
os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'

import tensorflow as tf

# Optimize GPU configuration
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
        
        # Use mixed precision policy
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
        
        # Enable XLA optimization
        tf.config.optimizer.set_jit(True)
        
        print("✅ GPU optimized successfully")
    except RuntimeError as e:
        print(f"❌ GPU configuration error: {e}")

# Verify GPU setup
logical_devices = tf.config.list_logical_devices('GPU')
if logical_devices:
    print(f"✅ Using GPU: {logical_devices[0].name}")
else:
    print("❌ No GPU available")

import numpy as np
import tensorflow as tf
import os
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import LSTM, Embedding, Dense, Dropout, Bidirectional, Input, Attention, BatchNormalization, GlobalAveragePooling1D, Lambda
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
from tensorflow.keras.layers import LSTM, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay, LearningRateSchedule
from tensorflow.keras.regularizers import l2
from tensorflow.keras.metrics import AUC, Precision, Recall

# Register WarmUp scheduler with Keras
@tf.keras.utils.register_keras_serializable()
class WarmUp(LearningRateSchedule):
    def __init__(self, initial_learning_rate, decay_schedule_fn, warmup_steps):
        super(WarmUp, self).__init__()
        self.initial_learning_rate = initial_learning_rate
        self.decay_schedule_fn = decay_schedule_fn
        self.warmup_steps = warmup_steps
        
    def __call__(self, step):
        global_step = tf.cast(step, tf.float32)
        warmup_percent = global_step / self.warmup_steps
        warmup_learning_rate = self.initial_learning_rate * warmup_percent
        decay_learning_rate = self.decay_schedule_fn(step)
        return tf.cond(
            global_step < self.warmup_steps,
            lambda: warmup_learning_rate,
            lambda: decay_learning_rate
        )
    
    def get_config(self):
        return {
            "initial_learning_rate": self.initial_learning_rate,
            "decay_schedule_fn": self.decay_schedule_fn,
            "warmup_steps": self.warmup_steps,
        }

@tf.keras.utils.register_keras_serializable()
class CustomModelCheckpoint(tf.keras.callbacks.ModelCheckpoint):
    def __init__(self, filepath, monitor='val_loss', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode='auto', save_freq='epoch'):
        super().__init__(
            filepath=filepath, monitor=monitor, verbose=verbose,
            save_best_only=save_best_only, save_weights_only=save_weights_only,
            mode=mode, save_freq=save_freq)
        self.model = None

    def set_model(self, model):
        self.model = model

    def get_config(self):
        config = super().get_config()
        return config

# Paths
MODEL_DIR = "models"
DATA_DIR = "data/text"
TEXT_MODEL_PATH = os.path.join(MODEL_DIR, "text_emotion_model.h5")
TOKENIZER_PATH = os.path.join(MODEL_DIR, "tokenizer.pkl")

# Parameters
TEXT_MAXLEN = 100
TEXT_VOCAB_SIZE = 20000
BATCH_SIZE = 64  # Increased for better GPU utilization
EPOCHS = 100
ACCUMULATION_STEPS = 1  # Removed gradient accumulation since it's not needed

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
    
    # Enhanced Embedding with larger dimensions
    embedding = Embedding(
        TEXT_VOCAB_SIZE, 
        384,  # Increased embedding size
        input_length=TEXT_MAXLEN,
        embeddings_regularizer=l2(1e-6),
        trainable=True
    )(inputs)
    
    # Improved architecture with transformer-style blocks
    x = embedding
    for i in range(2):  # Two transformer blocks
        # Multi-head attention
        attention_output = tf.keras.layers.MultiHeadAttention(
            num_heads=8, 
            key_dim=48
        )(x, x)
        x = tf.keras.layers.Add()([x, attention_output])
        x = BatchNormalization()(x)
        
        # Feed-forward network
        ffn = Dense(768, activation="gelu")(x)
        ffn = Dropout(0.2)(ffn)
        ffn = Dense(384)(ffn)
        x = tf.keras.layers.Add()([x, ffn])
        x = BatchNormalization()(x)
    
    # Global pooling with attention
    attention_weights = Dense(1, use_bias=False)(x)
    attention_weights = tf.keras.layers.Softmax(axis=1)(attention_weights)
    weighted_output = tf.keras.layers.Multiply()([x, attention_weights])
    
    # Replace Sum layer with Lambda layer
    context_vector = Lambda(lambda x: tf.reduce_sum(x, axis=1))(weighted_output)
    
    # Final classification layers
    x = Dense(256, activation="gelu")(context_vector)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    outputs = Dense(NUM_CLASSES, activation="sigmoid")(x)
    
    model = Model(inputs, outputs)
    
    # Cosine decay with warmup
    total_steps = (len(X_train) // (BATCH_SIZE * ACCUMULATION_STEPS)) * EPOCHS
    warmup_steps = total_steps // 10
    
    learning_rate = tf.keras.optimizers.schedules.CosineDecayRestarts(
        initial_learning_rate=0.001,
        first_decay_steps=total_steps - warmup_steps,
        t_mul=2.0,
        m_mul=0.9,
        alpha=0.1
    )
    
    # Warmup schedule
    learning_rate = WarmUp(
        initial_learning_rate=0.0001,
        decay_schedule_fn=learning_rate,
        warmup_steps=warmup_steps
    )
    
    # Replace AdamW with Adam configuration
    optimizer = tf.keras.mixed_precision.LossScaleOptimizer(
        Adam(
            learning_rate=learning_rate,
            clipnorm=1.0,
            epsilon=1e-7,
            beta_1=0.9,
            beta_2=0.999
        )
    )
    
    # Add L2 regularization in the model compile step
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
        metrics=['accuracy', AUC(name='auc', multi_label=True)],
        jit_compile=True  # Enable XLA
    )
    
    return model

# Load or create model with custom objects
custom_objects = {
    'WarmUp': WarmUp,
    'CustomModelCheckpoint': CustomModelCheckpoint
}

if os.path.exists(TEXT_MODEL_PATH):
    print("🔄 Loading existing text model...")
    text_model = load_model(TEXT_MODEL_PATH, custom_objects=custom_objects)
else:
    print("🆕 Creating new text model...")
    text_model = create_text_model()

# Update the LearningRateLogger class
class LearningRateLogger(tf.keras.callbacks.Callback):
    def __init__(self):
        super(LearningRateLogger, self).__init__()
        self._current_epoch = 0

    def on_epoch_begin(self, epoch, logs=None):
        self._current_epoch = epoch

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        optimizer = self.model.optimizer
        
        if isinstance(optimizer.learning_rate, tf.keras.optimizers.schedules.LearningRateSchedule):
            # Get the current learning rate from the schedule
            current_lr = optimizer.learning_rate(self._current_epoch)
            logs['learning_rate'] = float(current_lr.numpy())
        else:
            try:
                current_lr = optimizer.learning_rate
                if hasattr(current_lr, 'numpy'):
                    logs['learning_rate'] = float(current_lr.numpy())
                else:
                    logs['learning_rate'] = float(current_lr)
            except:
                # Fallback if we can't get the learning rate
                logs['learning_rate'] = 0.0

# Create callbacks with model reference
def create_callbacks(model):
    return [
        EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
            mode='min',
            verbose=1
        ),
        CustomModelCheckpoint(
            filepath=TEXT_MODEL_PATH,
            monitor="val_loss",
            mode='min',
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.2,
            patience=5,
            min_lr=1e-7,
            mode='min',
            verbose=1
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir='./logs',
            histogram_freq=1,
            update_freq='epoch'
        ),
        tf.keras.callbacks.CSVLogger(
            'training_log.csv',
            separator=',',
            append=False
        ),
        LearningRateLogger()
    ]

# Update create_dataset function
def create_dataset(x, y, batch_size, is_training=False):
    AUTO = tf.data.AUTOTUNE
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    
    if is_training:
        dataset = dataset.shuffle(50000, reshuffle_each_iteration=True)
        dataset = dataset.cache()
    
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(AUTO)
    
    # Enable parallel processing
    options = tf.data.Options()
    options.experimental_optimization.map_and_batch_fusion = True
    options.experimental_optimization.parallel_batch = True
    options.experimental_deterministic = False
    dataset = dataset.with_options(options)
    
    return dataset

# Create optimized datasets
train_data = create_dataset(X_train, y_train, BATCH_SIZE, is_training=True)
val_data = create_dataset(X_val, y_val, BATCH_SIZE)
test_data = create_dataset(X_test, y_test, BATCH_SIZE)

# Print dataset sizes before training
print(f"\nDataset sizes:")
print(f"Training samples: {len(X_train)}")
print(f"Training batches: {len(list(train_data))}")
print(f"Validation samples: {len(X_val)}")
print(f"Test samples: {len(X_test)}")

# Update training step function - remove jit_compile
@tf.function(jit_compile=True, reduce_retracing=True)
def train_step(x, y):
    with tf.device('/GPU:0'):
        with tf.GradientTape() as tape:
            predictions = text_model(x, training=True)
            per_example_loss = tf.keras.losses.binary_crossentropy(y, predictions)
            loss = tf.reduce_mean(per_example_loss)
            scaled_loss = text_model.optimizer.get_scaled_loss(loss)
        
        gradients = tape.gradient(scaled_loss, text_model.trainable_variables)
        gradients = text_model.optimizer.get_unscaled_gradients(gradients)
        text_model.optimizer.apply_gradients(zip(gradients, text_model.trainable_variables))
        return loss

# Main training loop
callbacks = create_callbacks(text_model)
for callback in callbacks:
    callback.set_model(text_model)
    callback.on_train_begin()

summary_writer = tf.summary.create_file_writer('./logs')

for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    
    # Training
    total_loss = 0.0
    num_batches = 0
    for x_batch, y_batch in train_data:
        loss = train_step(x_batch, y_batch)
        total_loss += loss
        num_batches += 1
    
    avg_loss = total_loss / num_batches
    print(f"Training loss: {avg_loss:.4f}")
    
    # Validation
    val_metrics = text_model.evaluate(val_data, verbose=1)
    
    # Log metrics manually to TensorBoard
    with summary_writer.as_default():
        tf.summary.scalar('epoch_loss', avg_loss, step=epoch)
        tf.summary.scalar('epoch_val_loss', val_metrics[0], step=epoch)
        tf.summary.scalar('epoch_val_accuracy', val_metrics[1], step=epoch)
        tf.summary.scalar('epoch_val_auc', val_metrics[2], step=epoch)
    
    # Update callbacks with proper logs
    logs = {
        'loss': float(avg_loss),
        'val_loss': float(val_metrics[0]),
        'val_accuracy': float(val_metrics[1]),
        'val_auc': float(val_metrics[2])
    }
    
    # Update callbacks except TensorBoard (we handle it manually)
    for callback in callbacks:
        if not isinstance(callback, tf.keras.callbacks.TensorBoard):
            callback.on_epoch_end(epoch, logs=logs)

# Cleanup
summary_writer.close()
for callback in callbacks:
    callback.on_train_end()

# Evaluate model with all metrics
metrics = text_model.evaluate(test_data, verbose=1)
print("\n✅ Model Evaluation Results:")
for metric_name, value in zip(text_model.metrics_names, metrics):
    print(f"{metric_name}: {value:.4f}")