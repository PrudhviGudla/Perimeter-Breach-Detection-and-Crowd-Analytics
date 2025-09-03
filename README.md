# Perimeter Breach Detection and Crowd Analytics

A real-time computer vision application that detects perimeter breaches and provides crowd analytics using YOLOv8 object detection, SORT tracking, and IoT integration via MQTT. Built with FastAPI for the web interface and SQLite for data persistence.

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


## Installation and Usage

### Prerequisites
- Python 3.8+
- OpenCV-compatible camera or IP webcam
- ESP32 (optional, for buzzer alerts)

### Setup
```
git clone https://github.com/PrudhviGudla/Perimeter-Breach-Detection-and-Crowd-Analytics.git
cd Perimeter-Breach-Detection-and-Crowd-Analytics
pip install -r requirements.txt
```

### Configuration
1. **Camera Setup**: Configure your camera URL in `.env` file:
```
IP_WEBCAM_URL=<url>
```
Or use local camera (default: camera index 0)

2. **MQTT Configuration**: Update MQTT settings in the code if needed:
```
mqtt_broker = 'broker.emqx.io'
mqtt_port = 1883
mqtt_topic = "esp32/buzzer"
```

### Run the Application
```
uvicorn main:app
```

Access the web interface at: `http://localhost:8000`

## ESP32 Buzzer Setup

### Hardware Requirements
- ESP32 development board
- Active buzzer
- Jumper wires
- Breadboard (optional)

### Wiring
ESP32 Pin D4 ─ Buzzer Positive
ESP32 GND ─ Buzzer Negative

### Software Setup
1. Install Arduino IDE with ESP32 board support
2. Install required libraries:
   - `PubSubClient` for MQTT
   - `WiFi` for network connectivity
3. Flash the provided ESP32 code from `/ESP32` directory
4. Configure WiFi credentials and MQTT settings

### ESP32 Code Configuration
```
const char* ssid = "YOUR_WIFI_SSID";
const char* password = "YOUR_WIFI_PASSWORD";
const char* mqtt_server = "broker.emqx.io";
const char* mqtt_topic = "esp32/buzzer";
const int buzzerPin = 4; // D4 pin
```

### Basic Operation
1. **Start Application**: Run `uvicorn main:app`
2. **Access Web Interface**: Open `http://localhost:8000`
3. **Draw Perimeter**: Click "Set Perimeter" and draw your restricted zone
4. **Monitor**: Watch real-time detection and analytics

### Perimeter Management
- **Draw**: Click "Set Perimeter" → Click points → Double-click to finish
- **Save**: Click "Save Current Perimeter" after drawing
- **Load**: Select from dropdown to load saved perimeters
- **Delete**: Select perimeter and click "Delete Selected"

## Technical Details

### Dependencies
- **FastAPI**: Modern web framework for APIs
- **OpenCV**: Computer vision and video processing
- **Ultralytics YOLOv8**: State-of-the-art object detection
- **SORT**: Multi-object tracking algorithm
- **SQLAlchemy**: Database ORM for data persistence
- **Paho-MQTT**: MQTT client for IoT communication
- **Shapely**: Geometric calculations for perimeter detection

##  API Endpoints

### Core Endpoints
- `GET /` - Web interface
- `GET /video_feed` - Live video stream
- `POST /set_shape` - Set current perimeter
- `GET /get_analytics` - Current analytics data

### Perimeter Management
- `POST /save_perimeter` - Save new perimeter
- `GET /get_perimeters` - List saved perimeters  
- `POST /load_perimeter/{id}` - Load specific perimeter
- `DELETE /delete_perimeter/{id}` - Delete perimeter

### Control Endpoints
- `POST /toggle_buzzer` - Control buzzer state
- `GET /get_buzzer_state` - Get current buzzer status

## Acknowledgments

- **Ultralytics** for YOLOv8 object detection
- **SORT** tracking algorithm contributors
- **FastAPI** framework developers
- **OpenCV** computer vision library
- **EMQX** for free MQTT broker services


## Future Enhancements
- **Multi-Camera Support**: Handle multiple camera feeds simultaneously
- **Cloud Integration**: AWS/Azure cloud deployment options
- **Advanced Analytics**: Dwell time analysis, and trajectory tracking
- **Mobile App**: Companion mobile application for remote monitoring

