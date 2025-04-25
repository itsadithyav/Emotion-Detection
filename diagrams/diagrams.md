# EmotiSense AI - System Diagrams

## Use Case Diagram
```mermaid
graph LR
    User((User))
    subgraph EmotiSense AI
        subgraph Analysis [Emotion Analysis]
            UC1[Analyze Facial Emotions]
            UC2[Analyze Text Emotions]
            UC3[Analyze Voice Emotions]
        end
        subgraph Management [Task Management]
            UC4[Manage Tasks]
            UC4_1[Create Task]
            UC4_2[Edit Task]
            UC4_3[Delete Task]
        end
        subgraph Insights [Insights & Reports]
            UC5[View Recommendations]
            UC6[View Emotion Statistics]
            UC6_1[View Daily Report]
            UC6_2[View Trends]
        end
    end
    
    User --> Analysis
    User --> Management
    User --> Insights
    
    UC4 --> UC4_1
    UC4 --> UC4_2
    UC4 --> UC4_3
    
    UC6 --> UC6_1
    UC6 --> UC6_2
    
    UC5 -.->|uses| UC1
    UC5 -.->|uses| UC2
    UC5 -.->|uses| UC3
    
    %% Add notes
    classDef note fill:#ffffcc
    note1[Real-time Analysis]:::note
    note2[Data-driven Insights]:::note
    
    UC1 -.-> note1
    UC5 -.-> note2
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
        +validate_settings()
        +load_from_file()
    }

    class TaskManager {
        -tasks_file: str
        -tasks: Dict
        -predefined_tasks: Dict
        -emotion_categories: Dict
        +add_task(task: Task)
        +complete_task(task_id: str)
        +get_recommendations(emotions: List)
        +update_task(task_id: str, data: Dict)
        +delete_task(task_id: str)
        +get_task_history()
    }

    class EmotionAnalyzer {
        -current_emotion: str
        -confidence: float
        -history: List
        +analyze_face(frame: np.array)
        +analyze_text(text: str)
        +analyze_voice(audio: np.array)
        +get_emotion_history()
    }

    class EmotionSmoother {
        -buffer_size: int
        -buffer: List
        -index: int
        +update(emotion: str)
        +get_smooth_emotion()
        +reset_buffer()
    }

    class AudioRecorder {
        -sample_rate: int
        -audio_data: List
        -stream: Optional
        -recording: bool
        +start()
        +stop()
        +callback()
        +get_audio_data()
    }

    class ModelManager {
        -config: Config
        -models: Dict
        -cache_dir: str
        +initialize_models()
        +clean_old_cache()
        +get_model(name: str)
        +update_model(name: str)
    }

    class StatisticsManager {
        -data: DataFrame
        +update_stats(emotion: str)
        +get_daily_report()
        +get_trends()
        +export_data()
    }

    TaskManager --> Config
    ModelManager --> Config
    AudioRecorder --> Config
    EmotionAnalyzer --> ModelManager
    EmotionAnalyzer --> EmotionSmoother
    TaskManager ..> StatisticsManager
    EmotionAnalyzer ..> StatisticsManager
```

## Sequence Diagram
```mermaid
sequenceDiagram
    actor User
    participant UI as Web Interface
    participant Auth as Authentication
    participant Camera as Camera Module
    participant Smoother as EmotionSmoother
    participant Models as ModelManager
    participant Tasks as TaskManager
    participant Stats as StatisticsManager
    participant DB as Database

    User->>UI: Start application
    activate UI
    
    UI->>Auth: Verify credentials
    activate Auth
    Auth-->>UI: Authentication success
    deactivate Auth
    
    par Initialize Components
        UI->>Camera: Initialize camera
        UI->>Models: Load models
        UI->>Tasks: Load task data
    end
    
    activate Camera
    activate Models
    activate Tasks
    
    Camera-->>UI: Camera ready
    Models-->>UI: Models loaded
    Tasks-->>UI: Tasks loaded
    
    loop Until stopped
        par Processing Streams
            Camera->>Camera: Capture frame
            Camera->>Models: Process frame
            Models->>Smoother: Update emotions
            Smoother-->>Models: Smoothed emotions
            Models-->>Stats: Update statistics
        and
            UI->>Tasks: Check pending tasks
            Tasks-->>UI: Active tasks
        end
        
        UI->>Stats: Request analytics
        Stats->>DB: Query data
        DB-->>Stats: Return data
        Stats-->>UI: Analytics results
        
        UI-->>User: Update display
    end

    User->>UI: Stop analysis
    
    par Cleanup
        UI->>Camera: Release camera
        UI->>Models: Save state
        UI->>DB: Save session data
    end
    
    deactivate Camera
    deactivate Models
    deactivate Tasks
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

    Browser -->|HTTPS| Streamlit_Server
    Webcam -->|USB/Built-in| Browser
    Microphone -->|USB/Built-in| Browser
    MM -->|reads/writes| MC
    TM -->|reads/writes| TS
```

## Query Analysis Chart
```mermaid
graph LR
    %% Query Types
    subgraph Queries[Query Types]
        direction TB
        E[Emotion Analysis]
        T[Task Operations]
        S[Statistics]
        R[Recommendations]
    end

    %% Processing Layers
    subgraph Processing[Query Processing]
        direction TB
        V[Validation Layer]
        C[Cache Check]
        P[Processing Layer]
        A[Analytics Engine]
    end

    %% Storage Layer
    subgraph Storage[Data Storage]
        direction TB
        MC[(Model Cache)]
        TS[(Task Store)]
        ES[(Emotion Store)]
        AS[(Analytics Store)]
    end

    %% Response Types
    subgraph Response[Response Types]
        direction TB
        RT[Real-time]
        BA[Batch Analysis]
        CA[Cached Analysis]
    end

    %% Flow Control
    E --> V
    T --> V
    S --> V
    R --> V

    V --> C
    C -->|Cache Hit| CA
    C -->|Cache Miss| P

    P --> MC
    P --> TS
    P --> ES
    P --> AS

    P --> RT
    P --> BA

    %% Style Definitions
    classDef query fill:#f9f,stroke:#333
    classDef process fill:#9cf,stroke:#333
    classDef storage fill:#ff9,stroke:#333
    classDef response fill:#9f9,stroke:#333

    class E,T,S,R query
    class V,C,P,A process
    class MC,TS,ES,AS storage
    class RT,BA,CA response
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
