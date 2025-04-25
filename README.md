# 🎭 EmotiSense AI: Multimodal Emotion Detection Platform

EmotiSense AI is a comprehensive emotion detection platform that analyzes emotions through facial expressions, text, and voice input. Built with Python and Streamlit, it provides real-time emotion analysis with an intuitive user interface.

## ✨ Key Features

### 🎥 Real-Time Facial Emotion Detection
- Real-time facial expression analysis
- Emotion confidence scoring
- Smooth emotion transitions
- Support for 7 basic emotions: happiness, sadness, anger, neutral, fear, surprise, and disgust

### 📝 Text Emotion Analysis
- Natural language emotion detection
- Support for multiple languages
- Confidence scoring for each emotion
- Instant analysis of text input

### 🎤 Voice Emotion Analysis
- Speech-to-text conversion
- Offline speech recognition support
- Multiple recognition engines (Whisper, Google, Sphinx)
- Real-time voice processing

### 📋 Task Management System
- Emotion-based task recommendations
- Category-based task organization
- Task completion tracking
- Predefined task templates

## 🚀 Installation & Setup

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/yourusername/EmotiSense-AI.git
   cd EmotiSense-AI
   ```

2. **Create Virtual Environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate     # Windows
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch Application:**
   ```bash
   streamlit run Emotion_Detection.py
   ```

## 🔧 Technical Architecture

### Core Components
- **EmotionSmoother**: Provides smooth emotion transitions
- **AudioProcessor**: Handles voice input and processing
- **TaskManager**: Manages task recommendations and tracking
- **ModelManager**: Handles AI model initialization and caching

### Configuration
```python
class Config:
    FRAME_RATE: int = 10
    FACE_DETECTION_SIZE: Tuple[int, int] = (160, 160)
    WEBCAM_DISPLAY_SIZE: Tuple[int, int] = (640, 480)
    SAMPLE_RATE: int = 16000
    DETECTOR_BACKEND: str = "opencv"
    OFFLINE_MODE: bool = False
```

## 💡 Usage Guidelines

### Camera Mode
1. Click "Start Camera" to begin facial analysis
2. Position yourself in good lighting
3. View real-time emotion analysis
4. Check recommended tasks based on emotions

### Text Mode
1. Enter text in the input area
2. Click "Analyze Text"
3. View emotion distribution
4. Get personalized task recommendations

### Voice Mode
1. Click "Start Recording"
2. Speak clearly into your microphone
3. Click "Stop Recording"
4. View transcription and emotion analysis

## 🎯 Task Management

### Categories
- Work
- Personal
- Health
- Learning
- Other

### Features
- Add custom tasks
- Mark tasks as complete
- View emotion-based recommendations
- Track task progress

## 🔒 Privacy & Security

- Local processing of all data
- No cloud dependencies
- Optional offline mode
- Secure data handling
- No data storage beyond session

## 🔧 Advanced Configuration

### Performance Tuning
```python
FRAME_RATE = 10        # Adjust for smoother/faster processing
BUFFER_SIZE = 5        # Emotion smoothing window
DETECTOR_BACKEND = "opencv"  # Change detection backend
```

### Model Configuration
```python
TEXT_MODEL = "SamLowe/roberta-base-go_emotions"
MODEL_CACHE_TTL = 30  # Days to keep cached models
OFFLINE_MODE = False  # Force offline processing
```

## 📊 Output Format

### Emotion Analysis Result
```json
{
  "dominant_emotion": "happy",
  "confidence": 0.85,
  "emotion_scores": {
    "happy": 0.85,
    "neutral": 0.10,
    "sad": 0.03,
    "angry": 0.02
  }
}
```

## 🤝 Contributing

Contributions are welcome! Please check our contribution guidelines.

## 📝 License

MIT License - Feel free to use for personal and commercial projects.

## ⚠️ Known Limitations

- CPU-only processing for wider compatibility
- Initial model loading time
- Requires good lighting for facial detection
- Network connection needed for first-time model download
