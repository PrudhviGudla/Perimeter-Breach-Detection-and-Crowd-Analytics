# Perimeter Breach Detection and Crowd Analytics

A real-time computer vision application that detects perimeter breaches and provides crowd analytics using YOLOv8 object detection, SORT tracking, and IoT integration via MQTT. Built with FastAPI for the web interface, SQLite for data persistence, and Apache Kafka for scalable video ingestion.

## Features

### Core Functionality
- **Custom Perimeter Drawing**: Draw any polygonal perimeter shape on the video feed
- **Real-time Person Detection**: YOLOv8-based person detection and SORT multi-object tracking
- **Breach Detection**: Instant alerts when persons enter restricted zones
- **Crowd Analytics**: Real-time people counting, crowd density analysis, and unusual behavior detection
- **Database Persistence**: Save perimeters and analytics data to SQLite database
- **IoT Integration**: MQTT-based buzzer alerts via ESP32

### Web Interface
- **Live Video Streaming**: Real-time video feed with overlay graphics
- **Interactive Perimeter Management**: Save, load, and delete custom perimeter configurations
- **Real-time Dashboard**: Live analytics display with breach counts and behavior monitoring
- **Responsive Design**: Clean, modern web interface with real-time updates

### Advanced Analytics
- **People Counting**: Accurate real-time headcount
- **Crowd Density Classification**: Low/Medium/High density analysis  
- **Behavior Detection**: Rapid movement and clustering detection
- **Historical Data**: Incident logging and analytics storage

### Distributed Architecture (Kafka)
- **Decoupled Ingestion**: Separates video capture (Producer) from inference (Consumer)
- **High Throughput**: Capable of handling high-resolution streams via optimized Kafka pipelines
- **Scalability**: Architecture supports adding multiple camera producers without modifying the core backend

## Performance Engineering & Optimizations

This project implements a highly optimized **Asyncio + Multithreading** architecture to maximize FPS and server responsiveness.

### 1. Asynchronous Non-Blocking Server
- **Problem**: Standard Python web servers block the Event Loop during heavy CPU tasks (like AI inference), causing the UI to freeze.
- **Solution**: Utilized `asyncio` and `asyncio.to_thread` to offload blocking I/O (camera reading) and CPU-bound tasks (YOLO inference) to worker threads.
- **Result**: The main Event Loop remains free to handle API requests (like dashboard updates or buzzer toggles) instantly, even while processing heavy video frames.

### 2. Parallel Inference via GIL Release
- **Concept**: Leveraged the fact that **OpenCV** and **PyTorch** (underlying YOLO) release the Python Global Interpreter Lock (GIL) during heavy C++ operations.
- **Result**: Achieved **True Parallelism**. The AI model runs on a separate CPU core/thread while the web server handles network traffic on the main core, maximizing hardware utilization.

### 3. Latency-Free Database Logging
- **Problem**: Writing every incident to the database sequentially adds latency (disk I/O) to the video processing loop.
- **Solution**: Implemented FastAPI's `BackgroundTasks`.
- **Result**: Incident logs and analytics are saved **after** the video frame is sent to the client. This ensures zero impact on the video streaming frame rate.

### 4. Kafka Video Pipeline (22+ FPS)
- **Architecture**: Implemented a simple Producer and a Consumer that loads latest frame.
- **Optimization**:
    - **Producer**: Uses `lz4` compression to blast frames at maximum camera speed.
    - **Consumer**: Uses `auto_offset_reset='latest'` and a polling strategy that discards old frames (`batch[-1]`).
- **Impact**: Increased FPS from ~15 (Monolithic) to **20+ FPS** (Distributed) by paralleling the "Read" and "Process" operations across two separate processes on my laptop with RTX 4060.

## Installation and Usage

### Prerequisites
- Python 3.8+
- Docker & Docker Compose (for Kafka)
- OpenCV-compatible camera or IP webcam
- NVIDIA GPU (Optional, for TensorRT acceleration)

### Setup
```bash
git clone https://github.com/PrudhviGudla/Perimeter-Breach-Detection-and-Crowd-Analytics.git
cd Perimeter-Breach-Detection-and-Crowd-Analytics
pip install -r requirements.txt
```

### Kafka Setup (Optional for Single Camera, Required for Scalability)

If you want to use the high-performance distributed pipeline:

1. **Start Kafka Cluster**:
Navigate to the `kafka-test` directory where the docker-compose file resides:
```bash
cd kafka-test
docker-compose up -d
```

2. **Configure Config**:
Ensure `config.json` in the config directory has `"USE_KAFKA": true`.

3. **Start the Producer**:
Open a terminal and run the camera producer:
```bash
python src/kafka_producer.py
```


### Run the Application

Start the FastAPI Backend (Consumer):

```bash
uvicorn src.main:app --reload
```

Access the web interface at: `http://localhost:8000`

### Configuration (`config.json`)

The application uses a centralized `config.json` in the config directory.

```json
{
  "USE_KAFKA": true,
  "KAFKA_BOOTSTRAP_SERVERS": ["127.0.0.1:9092"],
  "KAFKA_TOPIC": "video_stream",
  "IP_WEBCAM_URL": "0",
  "MODEL_PATH": "assets/yolov8l.pt",
  "STATIC_DIR": "static",
  "TEMPLATES_DIR": "templates",
  "ASSETS_DIR": "assets",
  "DB_PATH": "assets/perimeter_detection.db",
  "MQTT_BROKER": "broker.emqx.io",
  "MQTT_PORT": 1883,
  "MQTT_TOPIC": "esp32/buzzer"
}
```

## Folder Structure

```
├── assets/                 # Logs, database file, and model artifacts
├── config/                 # Local runtime configuration
│   └── config.json         # Main configuration used by the app (KAFKA, MODEL_PATH, MQTT, etc.)
├── src/                    # Application source code
│   ├── main.py             # FastAPI backend, inference & consumer logic
│   ├── kafka_producer.py   # Kafka producer script 
│   ├── sort.py             # SORT tracking implementation
├── kafka-test/             # Local Kafka test environment (docker-compose + example scripts)
├── static/                 # Web static assets: CSS, JS, images
├── templates/              # Jinja2 HTML templates
├── tensorrt_optimization/  # Notebooks and assets to build TensorRT engines
│   ├── TENSORRT_SETUP.md   # Platform-specific TensorRT install notes (Windows)
│   ├── build_tensorrt_engine.ipynb  # Notebook to convert models to .engine (FP16/INT8) and analyze performance
│   └── assets/             # Generated ONNX / engine files (yolov8l_fp16.engine, yolov8l_int8.engine)
├── ESP32/                  # ESP32 related code
│   └── MQTT_buzzer.ino     # Arduino sketch to receive MQTT buzzer messages
├── requirements.txt        # Python dependencies
└── README.md
```

## 🚀 Performance Optimization (TensorRT)

To enable real-time inference on edge hardware (RTX 4060 Laptop GPU), I migrated the detection pipeline from standard PyTorch (`.pt`) to NVIDIA TensorRT (`.engine`).

### **Methodology**
* **Post-Training Quantization (PTQ):** Converted YOLOv8l to **INT8** precision, reducing memory bandwidth usage by **~50%**.
* **Domain-Specific Calibration:** Instead of using generic COCO datasets, I implemented a custom calibration pipeline that extracts 100 random frames from the deployment environment. This ensures the quantization parameters (dynamic range) are optimized specifically for the lighting and object scales of this facility.

> **Note:** While the TensorRT engine is capable of 90+ FPS, the end-to-end application runs at **~20 FPS** in the current build. Profiling confirmed this is due to I/O bottlenecks (video source capture rate) and visualization overhead, proving that the GPU compute is no longer the rate-limiting factor.

## Setup

Follow these steps to set up the environment and run the optimization notebooks.

### **1. Create Virtual Environment**
```bash
# Create the environment
python -m venv venv

# Activate it
# Windows:
.\venv\Scripts\activate
# Linux/WSL:
source venv/bin/activate
```

### **2. Install Dependencies & Jupyter Kernel**

```bash
# Install core requirements
pip install -r requirements.txt

# Install Jupyter and Kernel helper
pip install jupyter ipykernel

# Register this venv as a kernel for Jupyter
python -m ipykernel install --user --name=perimeter_breach --display-name "Perimeter Breach (Python 3.10)"
```

### **3. Install TensorRT**

**Important:** TensorRT installation on Windows is non-standard and requires specific DLLs (`zlibwapi.dll`) and Path configurations.
**[Read the TensorRT Installation Guide](tensorrt_optimization/TENSORRT_SETUP.md)**

### **4. Launch Optimization Studio**

```bash
jupyter notebook
# Open 'build_tensorrt_engine.ipynb'
```

## API Endpoints

### Core Endpoints

* `GET /` - Web interface
* `GET /video_feed` - Live video stream (Kafka Consumer or Direct capture)
* `POST /set_shape` - Set current perimeter
* `GET /get_analytics` - Current analytics data

### Perimeter Management

* `POST /save_perimeter` - Save new perimeter (Threaded)
* `GET /get_perimeters` - List saved perimeters (Threaded)
* `POST /load_perimeter/{id}` - Load specific perimeter (Threaded)
* `DELETE /delete_perimeter/{id}` - Delete perimeter (Threaded)

### Buzzer Control Endpoints

* `POST /toggle_buzzer` - Control buzzer state
* `GET /get_buzzer_state` - Get current buzzer status
