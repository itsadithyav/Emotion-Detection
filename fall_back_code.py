import streamlit as st
import cv2
import time
import asyncio
import sys
from typing import Dict, Optional, List, Any, Tuple, Set
import plotly.graph_objects as go
from transformers import pipeline
from deepface import DeepFace
import shutil, os
import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wavfile
import speech_recognition as sr
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import logging
from functools import lru_cache
import tempfile
import json

# Add page configuration at the top
st.set_page_config(
    page_title="🎭 EmotiSense AI",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="collapsed"
)

@dataclass
class Config:
    FRAME_RATE: int = 10
    FACE_DETECTION_SIZE: Tuple[int, int] = (160, 160)
    WEBCAM_DISPLAY_SIZE: Tuple[int, int] = (640, 480)
    SAMPLE_RATE: int = 16000
    MIN_AUDIO_LENGTH: float = 0.5
    DETECTOR_BACKEND: str = "opencv"
    CONFIDENCE_THRESHOLD: float = 0.4
    TEMP_DIR: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp")
    MODELS_DIR: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    TEXT_MODEL_CACHE: str = os.path.join(MODELS_DIR, "text_model_cache")
    FACE_MODEL_CACHE: str = os.path.join(MODELS_DIR, "face_model_cache")
    VOICE_MODEL_CACHE: str = os.path.join(MODELS_DIR, "voice_model_cache")
    OFFLINE_MODE: bool = False  # Set to True to force offline mode
    MODEL_CACHE_TTL: int = 30  # Days to keep cached models
    TEXT_MODEL: str = "SamLowe/roberta-base-go_emotions"
    EMOTION_BUFFER_SIZE: int = 5
    THEME_COLOR: str = "#6750A4"
    THEME_SECONDARY: str = "#B4A7D6"
    ACCENT_COLOR: str = "#D0BCFF"

# Global configuration
config = Config()

# Create temp directory if it doesn't exist
os.makedirs(config.TEMP_DIR, exist_ok=True)

# Force CPU-only mode
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Configure logging
logging.basicConfig(
    handlers=[logging.NullHandler()],
    level=logging.WARNING
)
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())
logger.propagate = False

# Initialize thread pool
thread_pool = ThreadPoolExecutor(max_workers=2)

class TaskManager:
    def __init__(self):
        self.tasks_file = os.path.join(config.TEMP_DIR, "tasks.json")
        self._load_tasks()
        
        self.task_recommendations = {
            "happy": ["creative tasks", "social activities", "learning new skills"],
            "sad": ["relaxing activities", "mindfulness exercises", "light physical activities"],
            "angry": ["calming activities", "organization tasks", "problem-solving"],
            "fear": ["planning activities", "simple tasks", "team activities"],
            "neutral": ["routine tasks", "administrative work", "analysis tasks"],
            "surprise": ["exploration tasks", "brainstorming", "innovative projects"],
            "disgust": ["cleansing activities", "reorganization tasks", "improvement projects"]
        }
    
    def _load_tasks(self) -> None:
        if os.path.exists(self.tasks_file):
            try:
                with open(self.tasks_file, 'r') as f:
                    self.tasks = json.load(f)
            except:
                self.tasks = {"pending": [], "completed": []}
        else:
            self.tasks = {"pending": [], "completed": []}
    
    def _save_tasks(self) -> None:
        with open(self.tasks_file, 'w') as f:
            json.dump(self.tasks, f)
    
    def add_task(self, task: str, category: str = "") -> None:
        self.tasks["pending"].append({
            "task": task,
            "category": category,
            "created_at": time.time()
        })
        self._save_tasks()
    
    def complete_task(self, task_index: int) -> None:
        if 0 <= task_index < len(self.tasks["pending"]):
            task = self.tasks["pending"].pop(task_index)
            task["completed_at"] = time.time()
            self.tasks["completed"].append(task)
            self._save_tasks()
    
    def get_recommendations(self, emotion: str) -> List[str]:
        emotion = emotion.lower()
        if emotion in self.task_recommendations:
            return self.task_recommendations[emotion]
        return self.task_recommendations["neutral"]

class EmotionSmoother:
    def __init__(self, buffer_size: int):
        self.buffer_size = buffer_size
        self.buffer: List[Dict[str, float]] = [{} for _ in range(buffer_size)]
        self.index = 0

    def update(self, new_scores: Dict[str, float]) -> Dict[str, float]:
        self.buffer[self.index] = new_scores
        self.index = (self.index + 1) % self.buffer_size
        
        # Calculate smoothed scores
        smoothed = {}
        for emotion in new_scores.keys():
            scores = [b[emotion] for b in self.buffer]
            smoothed[emotion] = sum(scores) / len(scores)
        return smoothed

class FPSLimiter:
    def __init__(self, fps: int):  # Changed parameter name from target_fps to fps
        self.interval = 1.0 / fps
        self.last_time = time.time()
    
    def should_process_frame(self) -> bool:
        current_time = time.time()
        if current_time - self.last_time >= self.interval:
            self.last_time = current_time
            return True
        return False

class AudioRecorder:
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.audio_data: List[float] = []
        self.stream: Optional[sd.InputStream] = None
        self.recording: bool = False

    def callback(self, indata: np.ndarray, frames: int, time: Any, status: Any) -> None:
        if status:
            st.warning(f'Audio callback status: {status}')
        if self.recording:
            self.audio_data.extend(indata.copy().flatten())

    def start(self):
        try:
            self.audio_data = []
            self.recording = True
            self.stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                callback=self.callback,
                dtype=np.float32
            )
            self.stream.start()
        except Exception as e:
            st.error(f"Error starting audio recording: {str(e)}")

    def stop(self):
        if self.stream:
            try:
                self.stream.stop()
                self.stream.close()
                self.recording = False
                return np.array(self.audio_data, dtype=np.float32)
            except Exception as e:
                st.error(f"Error stopping audio recording: {str(e)}")
                return None
        return None

class AudioProcessor:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.recognizer.energy_threshold = 300
        self.recognizer.dynamic_energy_threshold = True
        self.recognizer.dynamic_energy_adjustment_damping = 0.15
        self.recognizer.dynamic_energy_ratio = 1.5
        self.recognizer.pause_threshold = 0.8
        self.recognizer.phrase_threshold = 0.3
        
    def process_audio(self, audio_data: np.ndarray) -> Optional[str]:
        if len(audio_data) < config.MIN_AUDIO_LENGTH * config.SAMPLE_RATE:
            st.warning("Audio too short - please speak longer")
            return None
            
        temp_path = os.path.join(config.TEMP_DIR, f"audio_{int(time.time())}.wav")
        try:
            normalized_audio = audio_data / np.max(np.abs(audio_data))
            wavfile.write(temp_path, config.SAMPLE_RATE, 
                        (normalized_audio * 32767).astype(np.int16))
            
            with sr.AudioFile(temp_path) as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                audio = self.recognizer.record(source)
                
                try:
                    text = self.recognizer.recognize_whisper(audio, language="english")
                except:
                    try:
                        text = self.recognizer.recognize_google(
                            audio,
                            language='en-US',
                            show_all=False
                        )
                    except:
                        text = self.recognizer.recognize_sphinx(audio)
                
                return text
                    
        except Exception as e:
            st.error(f"Audio processing error: {str(e)}")
        finally:
            try:
                os.remove(temp_path)
            except:
                pass
        return None

def process_frame(frame: np.ndarray, size: Tuple[int, int]) -> Optional[Tuple[str, Dict[str, float]]]:
    """Process a single frame with emotion detection"""
    try:
        if not hasattr(process_frame, 'smoother'):
            process_frame.smoother = EmotionSmoother(config.EMOTION_BUFFER_SIZE)
        
        result = DeepFace.analyze(
            img_path=frame,
            actions=['emotion'],
            enforce_detection=False,
            detector_backend=config.DETECTOR_BACKEND,
            silent=True
        )
        
        emotions = result[0]['emotion']
        emotion_scores = {k: float(v)/100 for k, v in emotions.items()}
        smoothed_scores = process_frame.smoother.update(emotion_scores)
        dominant_emotion = max(smoothed_scores.items(), key=lambda x: x[1])[0]
        return dominant_emotion, smoothed_scores
            
    except Exception:
        return None

class ModelManager:
    def __init__(self, config: Config):
        self.config = config
        os.makedirs(config.MODELS_DIR, exist_ok=True)
        os.makedirs(config.TEXT_MODEL_CACHE, exist_ok=True)
        os.makedirs(config.FACE_MODEL_CACHE, exist_ok=True)
        os.makedirs(config.VOICE_MODEL_CACHE, exist_ok=True)
    
    def initialize_models(self) -> None:
        """Initialize and cache all models"""
        try:
            # Cache face model without any parameters
            _ = DeepFace.analyze(
                img_path=np.zeros((48, 48, 3), dtype=np.uint8),
                actions=['emotion'],
                enforce_detection=False,
                detector_backend=self.config.DETECTOR_BACKEND,
                silent=True
            )
            
            # Text model will be cached by the pipeline
            logger.info("✅ Models initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing models: {e}")
            raise
    
    def clean_old_cache(self) -> None:
        """Clean cached models older than MODEL_CACHE_TTL days"""
        current_time = time.time()
        for cache_dir in [self.config.TEXT_MODEL_CACHE, self.config.FACE_MODEL_CACHE, self.config.VOICE_MODEL_CACHE]:
            try:
                for root, _, files in os.walk(cache_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        if current_time - os.path.getmtime(file_path) > self.config.MODEL_CACHE_TTL * 24 * 3600:
                            os.remove(file_path)
            except Exception as e:
                logger.error(f"Error cleaning cache {cache_dir}: {e}")

# Initialize model manager after config and before initialize_models
model_manager = ModelManager(config)

# Initialize task manager after config
task_manager = TaskManager()

# Update session state initialization
def init_session_state():
    st.session_state.setdefault('stop', True)
    st.session_state.setdefault('recording', False)
    st.session_state.setdefault('audio_recorder', None)
    st.session_state.setdefault('current_tab', 0)
    st.session_state.setdefault('camera', None)
    st.session_state.setdefault('last_emotion', None)
    st.session_state.setdefault('show_task_recommendations', False)

# Initialize session state before UI elements
init_session_state()

@st.cache_resource(ttl=3600)  # Cache for 1 hour only
def initialize_models():
    """Initialize all models with offline support"""
    try:
        global model_manager
        if config.OFFLINE_MODE:
            logger.info("Using offline mode - loading cached models only")
        
        model_manager.initialize_models()  # This will ensure models are cached
        
        # Initialize text classifier with offline support
        text_classifier = pipeline(
            "text-classification", 
            model=config.TEXT_MODEL,
            return_all_scores=True,
            device="cpu",
            batch_size=1,
            local_files_only=config.OFFLINE_MODE,
            cache_dir=config.TEXT_MODEL_CACHE
        )
        
        # Clean up old cached models periodically
        model_manager.clean_old_cache()
        
        return text_classifier
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        logger.error(f"Model initialization error: {e}")
        return None

# Update the text_classifier initialization
text_classifier = initialize_models()
if text_classifier is None:
    st.error("Failed to initialize emotion detection models. Please refresh the page.")
    st.stop()

def create_emotion_chart(emotions_dict: Dict[str, float], title: str = "Emotion Confidence Scores"):
    FONT_FAMILY = "-apple-system, BlinkMacSystemFont, sans-serif"
    
    sorted_emotions = dict(sorted(emotions_dict.items(), key=lambda x: x[1], reverse=True))
    
    fig = go.Figure(data=[
        go.Bar(
            x=list(sorted_emotions.values()),
            y=list(sorted_emotions.keys()),
            orientation='h',
            marker=dict(
                color=list(sorted_emotions.values()),
                colorscale=[[0, '#4B73FF'], [1, '#007AFF']],
                showscale=False
            ),
            textposition='auto',
            textfont=dict(
                family=FONT_FAMILY,
                color='rgba(255,255,255,0.85)'
            ),
            hovertemplate='<b>%{y}</b><br>%{x:.1%}<extra></extra>'
        )
    ])

    fig.update_layout(
        title=dict(
            text=title,
            font=dict(
                size=20, 
                family=FONT_FAMILY, 
                color='rgba(255,255,255,0.95)'
            ),
            x=0.5
        ),
        xaxis_title=None,
        yaxis_title=None,
        height=400,
        margin=dict(l=0, r=0, t=40, b=0),
        paper_bgcolor='rgba(26,27,30,0.8)',
        plot_bgcolor='rgba(26,27,30,0.8)',
        font=dict(
            family=FONT_FAMILY,
            color='rgba(255,255,255,0.85)',
            size=14
        ),
        bargap=0.3,
        xaxis=dict(
            gridcolor='rgba(255,255,255,0.05)',
            zerolinecolor='rgba(255,255,255,0.05)',
            tickformat='.0%',
            showgrid=True,
            range=[0, 1]
        ),
        yaxis=dict(
            gridcolor='rgba(255,255,255,0.05)',
            zerolinecolor='rgba(255,255,255,0.05)',
            showgrid=False
        ),
        showlegend=False,
        hovermode='closest',
        template="plotly_dark"
    )
    
    fig.layout.update(
        modebar_remove=[
            'zoom', 'pan', 'select', 'lasso2d', 'zoomIn2d',
            'zoomOut2d', 'autoScale2d', 'resetScale2d'
        ]
    )
    
    return fig

async def webcam_loop(video_placeholder: st.empty, stats_placeholder: st.empty) -> None:
    fps_limiter = FPSLimiter(fps=config.FRAME_RATE)  # Updated to use correct parameter name
    frame_count = 0
    last_frame = None
    
    try:
        while not st.session_state.stop:
            if not fps_limiter.should_process_frame():
                await asyncio.sleep(0.001)
                continue
            
            ret, frame = st.session_state.camera.read()
            if ret:
                last_frame = frame.copy()
            
            if not ret:
                st.error("Camera error - please restart")
                break
            
            result = await asyncio.to_thread(process_frame, frame, config.FACE_DETECTION_SIZE)
            
            if result:
                emotion, scores = result
                st.session_state.last_emotion = emotion  # Store last detected emotion
                cv2.putText(
                    frame,
                    f"Emotion: {emotion.upper()}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 0),
                    2
                )
                
                try:
                    stats_placeholder.plotly_chart(
                        create_emotion_chart(scores, "Emotion Analysis"),
                        use_container_width=True,
                        key=f"emotion_chart_{frame_count}"
                    )
                except Exception as e:
                    logger.error(f"Chart error: {e}")
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            video_placeholder.image(frame_rgb, channels="RGB")
            frame_count += 1
            
    except Exception as e:
        st.error(f"Webcam error: {e}")
    finally:
        st.session_state.camera.release()
        cv2.destroyAllWindows()
        video_placeholder.empty()
        stats_placeholder.empty()
        if last_frame is not None:
            final_result = process_frame(last_frame, config.FACE_DETECTION_SIZE)
            if final_result:
                emotion, scores = final_result
                st.markdown(f"""
                    <div style='background: var(--surface-color); padding: 1rem; border-radius: 12px; margin: 1rem 0;'>
                        <h3 style='color: var(--accent-color);'>Final Expression Analysis</h3>
                        <p style='color: var(--text-color);'>{emotion}</p>
                    </div>
                """, unsafe_allow_html=True)

st.markdown("""
    <style>
        :root {
            --theme-color: #6750A4;
            --theme-secondary: #B4A7D6;
            --accent-color: #D0BCFF;
            --surface-color: #1C1B1F;
            --on-surface: #E6E1E5;
            --bg-color: #141218;
            --text-color: #E6E1E5;
        }

        .stApp {
            background: linear-gradient(145deg, var(--bg-color), var(--surface-color));
            color: var(--text-color);
        }
        
        .main-container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 2rem;
            background: transparent;
            backdrop-filter: blur(10px);
        }
        
        .stButton > button {
            background: linear-gradient(135deg, var(--theme-color), var(--theme-secondary));
            color: var(--on-surface);
            border: none;
            border-radius: 16px;
            padding: 0.8em 2em;
            font-weight: 500;
            transition: all 0.3s ease;
            box-shadow: 0 4px 12px rgba(0,0,0,0.2);
        }
        
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(var(--accent-color), 0.3);
        }
        
        h1 {
            background: linear-gradient(135deg, var(--accent-color), var(--theme-secondary));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 3.5rem;
            text-align: center;
            margin: 1rem 0 2rem 0;
            font-weight: 800;
            letter-spacing: -1px;
            text-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-container">', unsafe_allow_html=True)
st.markdown('<h1>🎭 EmotiSense AI</h1>', unsafe_allow_html=True)

tab_titles = ["Camera", "Text", "Voice"]
selected_tab = st.radio(
    label="Navigation",
    options=tab_titles,
    horizontal=True,
    key="tab_select",
    label_visibility="collapsed",
    format_func=lambda x: f"📸 {x}" if x == "Camera" else f"📝 {x}" if x == "Text" else f"🎤 {x}",
    index=st.session_state.current_tab
)
st.session_state.current_tab = tab_titles.index(selected_tab)

if st.session_state.current_tab == 0:
    st.info("💡 Make sure you're well-lit and facing the camera directly for best results")
    
    col1, col2 = st.columns([1, 1])
    with col1:
        video_placeholder = st.empty()
    with col2:
        stats_placeholder = st.empty()
    
    if st.button("📸 Start/Stop Camera"):
        st.session_state.stop = not st.session_state.stop
        if st.session_state.stop and st.session_state.camera is not None:
            st.session_state.camera.release()
            cv2.destroyAllWindows()
            st.session_state.camera = None
        elif not st.session_state.stop:
            try:
                st.session_state.camera = cv2.VideoCapture(0)
                st.session_state.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                st.session_state.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                st.session_state.camera.set(cv2.CAP_PROP_FPS, config.FRAME_RATE)
                
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(webcam_loop(video_placeholder, stats_placeholder))
            except Exception as e:
                st.error(f"Camera error: {str(e)}")
                st.session_state.stop = True

elif st.session_state.current_tab == 1:
    st.info("💡 Try writing a complete sentence to get more accurate emotion analysis")
    
    with st.form("text_analysis_form"):
        user_text = st.text_area(
            "Enter text to analyze",
            height=100,
            placeholder="Type or paste text here..."
        )
        analyze_button = st.form_submit_button("📝 Analyze Text")
        
    if analyze_button and user_text:
        with st.spinner("Analyzing text..."):
            start_time = time.time()
            result = text_classifier(user_text, truncation=True)
            emotion_scores = {
                item['label']: item['score'] 
                for item in result[0]
            }
            dominant_emotion = max(emotion_scores.items(), key=lambda x: x[1])[0]
            confidence = emotion_scores[dominant_emotion]
            st.session_state.last_emotion = dominant_emotion  # Store last detected emotion for text analysis
            
            st.markdown(f"""
                <div style='background: var(--surface-color); padding: 1.5rem; border-radius: 12px; margin: 1rem 0;'>
                    <h3 style='color: var(--accent-color); margin-bottom: 1rem;'>Text Analysis</h3>
                    <p style='color: var(--text-color); margin-bottom: 1rem;'><b>Input Text:</b> {user_text}</p>
                    <p style='color: var(--text-color); margin-bottom: 0.5rem;'><b>Detected Emotion:</b> {dominant_emotion.title()}</p>
                    <p style='color: var(--text-color);'><b>Confidence:</b> {confidence:.1%}</p>
                </div>
            """, unsafe_allow_html=True)
            
            st.plotly_chart(
                create_emotion_chart(emotion_scores, "Emotion Distribution"),
                use_container_width=True
            )
            analysis_time = time.time() - start_time
            st.info(f"⌛ Analysis completed in {analysis_time:.2f} seconds")

else:
    st.info("💡 Speak clearly and pause briefly between sentences")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        if not st.session_state.recording:
            if st.button("🎤 Start Recording"):
                st.session_state.recording = True
                st.session_state.audio_recorder = AudioRecorder()
                st.session_state.audio_recorder.start()
        else:
            if st.button("⏹️ Stop Recording"):
                audio_data = st.session_state.audio_recorder.stop()
                st.session_state.recording = False
                st.session_state.audio_recorder = None
                
                if audio_data is not None and len(audio_data) > 0:
                    with st.spinner("Processing audio..."):
                        start_time = time.time()
                        audio_processor = AudioProcessor()
                        text = audio_processor.process_audio(audio_data)
                        if text:
                            result = text_classifier(text, truncation=True)
                            emotion_scores = {
                                item['label']: item['score']
                                for item in result[0]
                            }
                            dominant_emotion = max(emotion_scores.items(), key=lambda x: x[1])[0]
                            confidence = emotion_scores[dominant_emotion]
                            
                            st.markdown(f"""
                                <div style='background: var(--surface-color); padding: 1.5rem; border-radius: 12px; margin: 1rem 0;'>
                                    <h3 style='color: var(--accent-color); margin-bottom: 1rem;'>Speech Analysis</h3>
                                    <p style='color: var(--text-color); margin-bottom: 1rem;'><b>Transcribed Text:</b> {text}</p>
                                    <p style='color: var(--text-color); margin-bottom: 0.5rem;'><b>Detected Emotion:</b> {dominant_emotion.title()}</p>
                                    <p style='color: var(--text-color);'><b>Confidence:</b> {confidence:.1%}</p>
                                </div>
                            """, unsafe_allow_html=True)
                            
                            st.plotly_chart(
                                create_emotion_chart(emotion_scores, "Emotion Distribution"),
                                use_container_width=True
                            )
                            analysis_time = time.time() - start_time
                            st.info(f"⌛ Analysis completed in {analysis_time:.2f} seconds")
                            
    with col2:
        if st.session_state.recording:
            st.error("🔴 Recording in progress...")

# Add task management UI after the main emotion detection tabs
st.markdown('<hr style="margin: 2rem 0;">', unsafe_allow_html=True)
st.markdown('<h2 style="color: var(--accent-color);">📋 Task Management</h2>', unsafe_allow_html=True)

# Task input section
col1, col2 = st.columns([3, 2])
with col1:
    with st.form("task_form"):
        new_task = st.text_input("Add a new task", placeholder="Enter your task here...")
        task_category = st.selectbox(
            "Task Category",
            ["Work", "Personal", "Health", "Learning", "Other"],
            index=0
        )
        submit_task = st.form_submit_button("➕ Add Task")
        
        if submit_task and new_task:
            task_manager.add_task(new_task, task_category)
            st.success("✅ Task added successfully!")

# Display pending tasks
with col2:
    st.markdown('<h3 style="color: var(--theme-secondary);">📝 Your Tasks</h3>', unsafe_allow_html=True)
    for i, task in enumerate(task_manager.tasks["pending"]):
        col_task, col_complete = st.columns([4, 1])
        with col_task:
            st.write(f"**{task['task']}** ({task['category']})")
        with col_complete:
            if st.button("✅", key=f"complete_task_{i}"):
                task_manager.complete_task(i)
                st.rerun()

# Show task recommendations based on emotion
if st.session_state.last_emotion:
    st.markdown('<h3 style="color: var(--theme-secondary);">🎯 Recommended Tasks</h3>', unsafe_allow_html=True)
    recommendations = task_manager.get_recommendations(st.session_state.last_emotion)
    for rec in recommendations:
        st.info(f"💡 {rec}")
else:
    st.info("🎯 Complete an emotion analysis to get task recommendations")

st.markdown('</div>', unsafe_allow_html=True)