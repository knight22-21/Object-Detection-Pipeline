# Real-Time Object Change Detection System

## Project Overview
This project is built for detecting missing and newly placed objects in a scene using a real-time video analytics pipeline. It combines YOLOv8 object detection, Deep SORT tracking, and change detection logic to identify when an object disappears or appears in the frame.

The project implements a real-time video analytics system that detects:
- **Missing Object Detection**: Identifies when a previously visible object disappears from the scene.
- **New Object Placement Detection**: Detects when a new object appears in the scene.

It achieves high FPS and reliable accuracy using YOLOv8 for detection, Deep SORT for tracking, and GPU acceleration.


---

## How It Works

Each video frame goes through the following stages:

1. **Detection**  
   YOLOv8s identifies objects in the frame.

2. **Tracking**  
   Deep SORT assigns consistent IDs across frames.

3. **Change Detection**  
   A comparison is made between tracked IDs in the current and previous frame to detect:
   - 🟥 **New Object** → Red label `New Object: ID`
   - 🟦 **Missing Object** → Blue label `Missing Object: ID`

4. **Annotation**  
   Bounding boxes, ID, and class labels are drawn.

5. **Class-wise Count**  
   Displayed live at the bottom left of the screen.

6. **Logging**  
   Optionally logs data to a CSV file for analytics.


---

## Features
- Real-time object detection and tracking
- New object and missing object detection
- Annotated output video with object IDs, classes, and change labels
- Class-wise object count display
- Frame-wise logging to CSV
- Dockerized for portability
- Ready for future improvements: database storage, REST API, live dashboard

---

## Project Structure
```
├── Dockerfile
├── README.md
├── requirements.txt
├── report_Krishna_Tyagi_ML_Intern.docx
│
├── src/
│   ├── main.py
│   ├── detector.py
│   ├── tracker.py
│   ├── change_detection.py
│   ├── utils.py
│   └── video_processor.py
│
└── data/
    ├── input_video.mp4
    ├── output_video.mp4
    └── logs.csv

```

> **Note:**
> `config/` and `models/` folders were originally suggested but not used.
> - **config/**: Was planned for YAML configs, hardcoding kept things simple.
> - **models/**: Direct loading YOLOv8 model using `ultralytics`, no need to store model separately.

---

## Tech Stack

| Component        | Tech Used                  |
|------------------|----------------------------|
| Detection        | YOLOv8s (Ultralytics)      |
| Tracking         | Deep SORT                  |
| Visualization    | OpenCV                     |
| Performance      | GPU Acceleration (CUDA)    |
| Containerization | Docker                     |
| Logging          | CSV (Pandas optional)      |


---

## How to Run

### 1. Clone the Repository
```bash
git clone https://github.com/knight22-21/object-change-detection.git
cd 'Folder-Name'
```

### 2. Install Requirements
```bash
pip install -r requirements.txt
```

### 3. Run the Pipeline
```bash
python src/main.py
```

The output video will be saved as `data/output_video.mp4` with all annotations.

### 4. (Optional) Run inside Docker
Build and run the Docker container:
```bash
docker build -t object-change-detection .
docker run --gpus all -v $(pwd):/app object-change-detection
```

---

## FPS and Performance
- **Model**: YOLOv8l (Large model)
- **Average FPS**: 22 FPS
- **Resolution**: 720p
- **GPU**: NVIDIA RTX 2050 (4GB)

Optimized for a balance between speed and accuracy.

---

## Optimizations Applied
- Switched from YOLOv8n to YOLOv8s for better precision.
- Enabled GPU acceleration.
- Disabled verbose logs.
- Minimal console output (frame logs to CSV).
- Efficient frame-by-frame writing to video file.

---

## Output Video Details
The final video (`data/output_video.mp4`) includes:
- Bounding boxes with IDs and class labels.
- Red labels for **New Object Detected**.
- Blue labels for **Missing Object Detected**.
- Class-wise object count at bottom-center.

---

## Known Limitations
- Occasional double-counting (due to overlapping YOLO detections).
- Missed detections for small or far objects.
- Minor FPS drop during heavy scene density.

---

## Future Improvements
- **Webcam Stream**: Allow live feed processing (cv2.VideoCapture(0)).
- **Database Integration**: Directly save object changes into MongoDB or SQLite.
- **REST API**: Serve real-time detections through FastAPI/Flask.
- **Web Dashboard**: Live stream + stats using Streamlit or React.
- **Model Fine-tuning**: Custom train YOLO for specific environments.
- **Multi-camera Support**: Handle multiple video streams simultaneously.
- **Edge Device Optimization**: Convert models for TensorRT, ONNX for deployment on devices like Jetson Nano.


---

## Submission Details
**Submitted by**: Krishna Tyagi  

---

## License
This project is for internship evaluation purposes. Further usage should comply with the YOLO and Deep SORT licenses.

