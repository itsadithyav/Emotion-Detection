import os
import re
import pandas as pd


AUDIO_DATASET_PATH = "AI Powered Task Optimizer/data/audio/audio_speech_actors_01-24"


audio_files = os.listdir(AUDIO_DATASET_PATH)
print(f"Files in {AUDIO_DATASET_PATH}: {audio_files}")


def prepare_audio_dataframe(audio_path):
    records = []
    
    # Loop through each actor directory
    for actor_folder in os.listdir(audio_path):
        actor_path = os.path.join(audio_path, actor_folder)
        
        # Skip if it's not a directory
        if not os.path.isdir(actor_path):
            continue

        # Loop through each file inside the actor's folder
        for filename in os.listdir(actor_path):
            if filename.endswith(".wav"):
                records.append([filename, os.path.join(actor_path, filename)])

    if not records:
        print("❌ No audio files found! Check the dataset path.")

    data = pd.DataFrame(records, columns=['filename', 'path'])
    print(f"✅ Found {len(data)} audio files.")

    return data


def extract_emotion_label(filename):
    # RAVDESS Format: "03-01-05-02-02-02-12.wav"
    parts = filename.split("-")
    
    if len(parts) < 2:
        print(f"❌ Filename format not recognized: {filename}")
        return -1  # Skip if the format is incorrect

    emotion_code = parts[2]  # Emotion is the 3rd part (index 2)
    
    emotion_map = {
        "01": 0, "02": 1, "03": 2, "04": 3, 
        "05": 4, "06": 5, "07": 6, "08": 6  # Adjusted if needed
    }

    return emotion_map.get(emotion_code, -1)


data = prepare_audio_dataframe(AUDIO_DATASET_PATH)
print(data.head())  # Check if paths are correct


for filename in data['filename'].head(10):  # Test extraction on a few files
    print(f"{filename} → Emotion: {extract_emotion_label(filename)}")
    

#X_audio, y_audio = load_audio(AUDIO_DATASET_PATH)
#print(f"X_audio shape: {X_audio.shape}, y_audio shape: {y_audio.shape}")

