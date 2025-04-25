# EmotiSense AI - System Diagrams

## Use Case Diagram
```plantuml
@startuml
left to right direction
actor User

rectangle "EmotiSense AI" {
  usecase "Analyze Facial Emotions" as UC1
  usecase "Analyze Text Emotions" as UC2
  usecase "Analyze Voice Emotions" as UC3
  usecase "Manage Tasks" as UC4
  usecase "View Recommendations" as UC5
  usecase "View Emotion Statistics" as UC6
}

User --> UC1
User --> UC2
User --> UC3
User --> UC4
User --> UC5
User --> UC6

UC5 ..> UC1 : <<include>>
UC5 ..> UC2 : <<include>>
UC5 ..> UC3 : <<include>>
@enduml
```

## Class Diagram
```plantuml
@startuml
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
@enduml
```

## Sequence Diagram
```plantuml
@startuml
actor User
participant "Web Interface" as UI
participant "Camera Module" as Camera
participant "EmotionSmoother" as Smoother
participant "ModelManager" as Models
participant "TaskManager" as Tasks

User -> UI: Start emotion analysis
activate UI

UI -> Camera: Initialize camera
activate Camera
Camera --> UI: Camera ready

loop Until stopped
    Camera -> Camera: Capture frame
    Camera -> Models: Process frame
    activate Models
    Models -> Smoother: Update emotions
    activate Smoother
    Smoother --> Models: Smoothed emotions
    deactivate Smoother
    Models --> UI: Analysis results
    deactivate Models
    
    UI -> Tasks: Get recommendations
    activate Tasks
    Tasks --> UI: Task suggestions
    deactivate Tasks
    
    UI --> User: Display results
end

User -> UI: Stop analysis
UI -> Camera: Release camera
deactivate Camera
deactivate UI
@enduml
```

## Collaboration Diagram
```plantuml
@startuml
object User
object "Web Interface" as UI
object "Camera Module" as Camera
object "Text Analyzer" as Text
object "Voice Analyzer" as Voice
object "Task Manager" as Tasks
object "Model Manager" as Models

User -- UI : interacts
UI -- Camera : captures
UI -- Text : processes
UI -- Voice : processes
UI -- Tasks : manages
Camera -- Models : uses
Text -- Models : uses
Voice -- Models : uses
Tasks -- Models : uses
@enduml
```

## Deployment Diagram
```plantuml
@startuml
node "User Device" {
  component Browser
  component Webcam
  component Microphone
}

node "Application Server" {
  component "Streamlit Server" {
    component "EmotiSense AI Application"
    component "Model Manager"
    component "Task Manager"
  }
  
  database "File System" {
    component "Model Cache"
    component "Task Storage"
  }
}

Browser -- "Streamlit Server" : HTTPS
Webcam -- Browser : USB/Built-in
Microphone -- Browser : USB/Built-in
"Model Manager" -- "Model Cache" : reads/writes
"Task Manager" -- "Task Storage" : reads/writes
@enduml
```

## Activity Diagram
```plantuml
@startuml
start
fork
  :Initialize Models;
  :Start Web Interface;
fork again
  :Initialize Task Manager;
  :Load Saved Tasks;
end fork

while (Application Running?) is (yes)
  fork
    :Process Camera Input;
    if (Face Detected?) then (yes)
      :Analyze Emotions;
      :Update Statistics;
    endif
  fork again
    :Handle Text Input;
    if (Text Entered?) then (yes)
      :Analyze Text Emotions;
      :Update Statistics;
    endif
  fork again
    :Handle Voice Input;
    if (Voice Recorded?) then (yes)
      :Convert to Text;
      :Analyze Emotions;
      :Update Statistics;
    endif
  end fork
  :Generate Recommendations;
  :Update UI;
endwhile (no)

:Cleanup Resources;
stop
@enduml
```

## Entity-Relationship Diagram
```plantuml
@startuml
entity "Task" {
  * task_id : number <<generated>>
  --
  * description : text
  * category : text
  * created_at : datetime
  completed_at : datetime
  status : text
}

entity "EmotionRecord" {
  * record_id : number <<generated>>
  --
  * timestamp : datetime
  * emotion : text
  * confidence : float
  source : text
}

entity "User" {
  * user_id : number <<generated>>
  --
  * created_at : datetime
  settings : json
}

entity "TaskCategory" {
  * category_id : number <<generated>>
  --
  * name : text
  description : text
}

User ||--o{ Task : manages
Task }o--|| TaskCategory : belongs_to
User ||--o{ EmotionRecord : generates
@enduml
```
