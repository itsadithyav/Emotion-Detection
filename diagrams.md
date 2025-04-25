# EmotiSense AI - System Diagrams

## Use Case Diagram
```mermaid
graph LR
    User((User))
    subgraph EmotiSense AI
        UC1[Analyze Facial Emotions]
        UC2[Analyze Text Emotions]
        UC3[Analyze Voice Emotions]
        UC4[Manage Tasks]
        UC5[View Recommendations]
        UC6[View Emotion Statistics]
    end
    
    User --> UC1
    User --> UC2
    User --> UC3
    User --> UC4
    User --> UC5
    User --> UC6
    
    UC5 -.-> UC1
    UC5 -.-> UC2
    UC5 -.-> UC3
```

## Class Diagram
```mermaid
classDiagram
    class Config {
        +FRAME_RATE: int
        +FACE_DETECTION_SIZE: Tuple
        +WEBCAM_DISPLAY_SIZE: Tuple
        +SAMPLE_RATE: int
        +MIN_AUDIO_LENGTH: float
        +DETECTOR_BACKEND: str
    }

    class TaskManager {
        -tasks_file: str
        -tasks: Dict
        -predefined_tasks: Dict
        -emotion_categories: Dict
        +add_task()
        +complete_task()
        +get_recommendations()
    }

    class EmotionSmoother {
        -buffer_size: int
        -buffer: List
        -index: int
        +update()
    }

    class AudioRecorder {
        -sample_rate: int
        -audio_data: List
        -stream: Optional
        -recording: bool
        +start()
        +stop()
        +callback()
    }

    class ModelManager {
        -config: Config
        +initialize_models()
        +clean_old_cache()
    }

    TaskManager --> Config
    ModelManager --> Config
    AudioRecorder --> Config
```

## Sequence Diagram
```mermaid
sequenceDiagram
    actor User
    participant UI as Web Interface
    participant Camera as Camera Module
    participant Smoother as EmotionSmoother
    participant Models as ModelManager
    participant Tasks as TaskManager

    User->>UI: Start emotion analysis
    activate UI
    UI->>Camera: Initialize camera
    activate Camera
    Camera-->>UI: Camera ready
    
    loop Until stopped
        Camera->>Camera: Capture frame
        Camera->>Models: Process frame
        activate Models
        Models->>Smoother: Update emotions
        activate Smoother
        Smoother-->>Models: Smoothed emotions
        deactivate Smoother
        Models-->>UI: Analysis results
        deactivate Models
        UI->>Tasks: Get recommendations
        activate Tasks
        Tasks-->>UI: Task suggestions
        deactivate Tasks
        UI-->>User: Display results
    end

    User->>UI: Stop analysis
    UI->>Camera: Release camera
    deactivate Camera
    deactivate UI
```

## Collaboration Diagram
```mermaid
graph TB
    User((User))
    UI[Web Interface]
    Camera[Camera Module]
    Text[Text Analyzer]
    Voice[Voice Analyzer]
    Tasks[Task Manager]
    Models[Model Manager]

    User --- UI
    UI --- Camera
    UI --- Text
    UI --- Voice
    UI --- Tasks
    Camera --- Models
    Text --- Models
    Voice --- Models
    Tasks --- Models
```

## Deployment Diagram
```mermaid
graph TB
    subgraph User Device
        Browser
        Webcam
        Microphone
    end

    subgraph Application Server
        subgraph Streamlit Server
            App[EmotiSense AI Application]
            MM[Model Manager]
            TM[Task Manager]
        end
        
        subgraph File System
            MC[Model Cache]
            TS[Task Storage]
        end
    end

    Browser -->|HTTPS| Streamlit Server
    Webcam -->|USB/Built-in| Browser
    Microphone -->|USB/Built-in| Browser
    MM -->|reads/writes| MC
    TM -->|reads/writes| TS
```

## Activity Diagram
```mermaid
graph TB
    Start((Start))
    
    subgraph Initialization
        Init1[Initialize Models]
        Init2[Start Web Interface]
        Init3[Initialize Task Manager]
        Init4[Load Saved Tasks]
    end

    subgraph Main Loop
        Check{Application Running?}
        
        subgraph Camera Processing
            C1[Process Camera Input]
            C2{Face Detected?}
            C3[Analyze Emotions]
            C4[Update Statistics]
        end
        
        subgraph Text Processing
            T1[Handle Text Input]
            T2{Text Entered?}
            T3[Analyze Text Emotions]
            T4[Update Statistics]
        end
        
        subgraph Voice Processing
            V1[Handle Voice Input]
            V2{Voice Recorded?}
            V3[Convert to Text]
            V4[Analyze Emotions]
            V5[Update Statistics]
        end
        
        R1[Generate Recommendations]
        R2[Update UI]
    end

    Cleanup[Cleanup Resources]
    End((End))

    Start --> Init1 & Init3
    Init1 --> Init2
    Init3 --> Init4
    Init2 & Init4 --> Check
    Check -->|yes| C1 & T1 & V1
    Check -->|no| Cleanup
    
    C1 --> C2
    C2 -->|yes| C3 --> C4
    
    T1 --> T2
    T2 -->|yes| T3 --> T4
    
    V1 --> V2
    V2 -->|yes| V3 --> V4 --> V5
    
    C4 & T4 & V5 --> R1 --> R2 --> Check
    Cleanup --> End
```

## Entity-Relationship Diagram
```mermaid
erDiagram
    User ||--o{ Task : manages
    User ||--o{ EmotionRecord : generates
    Task }o--|| TaskCategory : belongs_to
    
    Task {
        number task_id PK
        text description
        text category
        datetime created_at
        datetime completed_at
        text status
    }
    
    EmotionRecord {
        number record_id PK
        datetime timestamp
        text emotion
        float confidence
        text source
    }
    
    User {
        number user_id PK
        datetime created_at
        json settings
    }
    
    TaskCategory {
        number category_id PK
        text name
        text description
    }
```
