import os
import subprocess
import sys

# Paths
MODEL_DIR = "models"
TRAIN_AUDIO_SCRIPT = "training/train_audio_model.py"
TRAIN_TEXT_SCRIPT = "training/train_text_model.py"
TRAIN_IMAGE_SCRIPT = "training/train_image_model.py"

# Ensure model directory exists
os.makedirs(MODEL_DIR, exist_ok=True)

# Train models by calling external scripts
def train_model(script_name):
    if os.path.exists(script_name):
        print(f"🚀 Running {script_name}...")
        subprocess.run(["python", script_name], check=True)
        print(f"✅ {script_name} completed!")
    else:
        print(f"❌ {script_name} not found!")

# Execute training scripts
train_model(TRAIN_AUDIO_SCRIPT)
train_model(TRAIN_TEXT_SCRIPT)
train_model(TRAIN_IMAGE_SCRIPT)

print("✅ All training scripts executed!")
