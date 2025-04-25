# EmotiSense AI - Data Structures

## Core Data Structures

```mermaid
classDiagram
    class Config {
        +int FRAME_RATE
        +tuple FACE_DETECTION_SIZE
        +tuple WEBCAM_DISPLAY_SIZE
        +int SAMPLE_RATE
        +float MIN_AUDIO_LENGTH
        +str DETECTOR_BACKEND
        +float CONFIDENCE_THRESHOLD
        +str TEMP_DIR
        +str MODELS_DIR
        +bool OFFLINE_MODE
        +int MODEL_CACHE_TTL
        +str TEXT_MODEL
        +int EMOTION_BUFFER_SIZE
        +str THEME_COLOR
    }

    class EmotionSmoother {
        +int buffer_size
        +List buffer
        +int index
        +update(new_scores)
    }

    class TaskManager {
        +str tasks_file
        +Dict tasks
        +Dict predefined_tasks
        +Dict emotion_categories
        +add_task(task, category)
        +complete_task(task_index)
        +get_recommendations(emotion)
        -_load_tasks()
        -_save_tasks()
    }

    class FPSLimiter {
        +float interval
        +float last_time
        +should_process_frame()
    }

    class AudioRecorder {
        +int sample_rate
        +List audio_data
        +sd.InputStream stream
        +bool recording
        +callback(indata, frames, time, status)
        +start()
        +stop()
    }

    class AudioProcessor {
        +sr.Recognizer recognizer
        +process_audio(audio_data)
    }

    class ModelManager {
        +Config config
        +initialize_models()
        +clean_old_cache()
    }
```

## Data Flow Types

```mermaid
classDiagram
    class EmotionData {
        +str dominant_emotion
        +Dict~str,float~ emotion_scores
        +float confidence
        +float timestamp
    }

    class TaskData {
        +str task
        +str category
        +float created_at
        +float completed_at
        +bool is_complete
    }

    class AudioData {
        +np.ndarray samples
        +int sample_rate
        +float duration
        +str transcribed_text
    }

    class ProcessedFrame {
        +np.ndarray frame
        +EmotionData emotion
        +Dict metadata
    }
```

## Storage Structures

```mermaid
classDiagram
    class TaskStorage {
        +str filepath
        +List~TaskData~ pending_tasks
        +List~TaskData~ completed_tasks
        +save()
        +load()
    }

    class ModelCache {
        +str cache_dir
        +int ttl_days
        +Dict cached_models
        +cache_model(model)
        +get_model(name)
        +clean_old_cache()
    }

    class SessionState {
        +bool stop
        +bool recording
        +AudioRecorder audio_recorder
        +int current_tab
        +cv2.VideoCapture camera
        +str last_emotion
        +bool show_task_recommendations
    }
```

## Relationships

```mermaid
classDiagram
    Config <|-- ModelManager
    EmotionSmoother <|-- ProcessedFrame
    TaskManager <|-- TaskStorage
    AudioRecorder <|-- AudioProcessor
    AudioData <|-- AudioProcessor
    EmotionData <|-- TaskManager
    SessionState <|-- AudioRecorder
    ModelCache <|-- ModelManager

    class RelationshipMap {
        Config "1" --> "1" ModelManager
        EmotionSmoother "1" --> "*" ProcessedFrame
        TaskManager "1" --> "1" TaskStorage
        AudioRecorder "1" --> "1" AudioProcessor
        AudioProcessor "1" --> "*" AudioData
        TaskManager "1" --> "*" EmotionData
        SessionState "1" --> "1" AudioRecorder
        ModelManager "1" --> "1" ModelCache
    }
```

## Type Definitions

```typescript
// Key Type Definitions

type EmotionScores = {
    happy: number;
    sad: number;
    angry: number;
    fear: number;
    neutral: number;
    surprise: number;
    disgust: number;
}

type TaskCategory = "Work" | "Personal" | "Health" | "Learning" | "Other";

type ModelType = "face" | "text" | "voice";

type ProcessingStatus = {
    success: boolean;
    error?: string;
    duration: number;
    timestamp: number;
}
```
