from fastapi import FastAPI, Request, Body, Depends, BackgroundTasks, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import time
import cv2
import numpy as np
import math
import paho.mqtt.client as mqtt
import logging
from shapely.geometry import Point, Polygon
from datetime import datetime, timezone
from typing import Dict, Any
import os
from dotenv import load_dotenv
from contextlib import asynccontextmanager
from ultralytics import YOLO
import uvicorn
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
import json
from typing import Annotated
import asyncio
import os
import torch 
from kafka import KafkaConsumer
from pathlib import Path
try:
    from sort import Sort
except ImportError:
    from src.sort import Sort  # Adjusted import for local module
    
# Load environment variables
# load_dotenv()

# Project root (two levels up from this file: project/ src/ main.py)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
# Ensuring assets folder exists for logs / db
assets_dir = PROJECT_ROOT / 'assets'
assets_dir.mkdir(parents=True, exist_ok=True)

# Logging (file inside project assets)
log_file = assets_dir / 'app.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(str(log_file))
    ]
)
logger = logging.getLogger(__name__)


def load_config():
    """Load project config from project-root config.json with sensible defaults."""
    config_path = PROJECT_ROOT / 'config' / 'config.json'
    defaults = {
        "USE_KAFKA": False,
        "KAFKA_BOOTSTRAP_SERVERS": ["127.0.0.1:9092"],
        "KAFKA_TOPIC": "video_stream2",
        "IP_WEBCAM_URL": "0",
        "MODEL_PATH": str(PROJECT_ROOT / 'assets' / 'yolov8l.pt'),
        "STATIC_DIR": str(PROJECT_ROOT / 'static'),
        "TEMPLATES_DIR": str(PROJECT_ROOT / 'templates'),
        "ASSETS_DIR": str(assets_dir),
        "DB_PATH": str(PROJECT_ROOT / 'assets' / 'perimeter_detection.db'),
        "MQTT_BROKER": "broker.emqx.io",
        "MQTT_PORT": 1883
    }

    if config_path.exists():
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                cfg = json.load(f)
        except Exception as e:
            logger.warning(f"Failed to read config.json, using defaults: {e}")
            cfg = {}
    else:
        logger.info(f"No config.json found at {config_path}, using defaults")
        cfg = {}

    # Merge defaults
    for k, v in defaults.items():
        cfg.setdefault(k, v)

    # Normalize booleans and paths
    cfg['USE_KAFKA'] = bool(cfg.get('USE_KAFKA', False))
    # Ensure directory paths are absolute strings
    for path_key in ('STATIC_DIR', 'TEMPLATES_DIR', 'ASSETS_DIR', 'MODEL_PATH', 'DB_PATH'):
        cfg[path_key] = str(PROJECT_ROOT / cfg[path_key]) if not Path(cfg[path_key]).is_absolute() else str(cfg[path_key])

    # KAFKA_BOOTSTRAP_SERVERS may be a list or comma-separated string
    bservers = cfg.get('KAFKA_BOOTSTRAP_SERVERS')
    if isinstance(bservers, str):
        cfg['KAFKA_BOOTSTRAP_SERVERS'] = [s.strip() for s in bservers.split(',') if s.strip()]

    return cfg

# Load configuration once
CONFIG = load_config()

mqtt_client = mqtt.Client(client_id="perimeter_detection_backend")
mqtt_broker = CONFIG.get('MQTT_BROKER', 'broker.emqx.io')
mqtt_port = CONFIG.get('MQTT_PORT', 1883)
mqtt_topic = CONFIG.get('MQTT_TOPIC', 'esp32/buzzer')
SERVER_RUNNING = True  # Global flag to control infinite loops

class AppState:
    def __init__(self):
        self.shape_coordinates = None
        self.buzzer_on = True
        self.analytics = {
            "peopleCount": 0,
            "crowdDensity": "Low",
            "breachCount": 0,
            "unusualBehavior": False,
            "behaviorDetails": []
        }

app_state = AppState()

class ResourceManager:
    """Centralized management for camera, database, and MQTT resources"""
    
    def __init__(self):
        self.cap = None
        self.engine = None
        self.Session = None
        self.mqtt_client = None
        self.model = None
        self.tracker = None
        self.kafka_consumer = None
        self.use_kafka = bool(CONFIG.get("USE_KAFKA", False))
        # device and fps 
        self.device = "cpu"
        self.fps_window_count = 0
        self.fps_window_start = time.time()
        self.last_fps = 0.0
        
    def initialize_database(self):
        """Initialize database connection and create tables"""
        try:
            db_path = CONFIG.get('DB_PATH', str(PROJECT_ROOT / 'assets' / 'perimeter_detection.db'))
            db_dir = Path(db_path).parent
            db_dir.mkdir(parents=True, exist_ok=True)
            self.engine = create_engine(f'sqlite:///{db_path}', echo=True)
            logger.info(f"Database path: {db_path}")
            Base.metadata.create_all(self.engine)
            self.Session = sessionmaker(bind=self.engine)
            logger.info("Database initialized successfully")
            # Initialize test data
            self._initialize_test_data()
            
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise
    
    def _initialize_test_data(self):
        """Add test data to database"""
        try:
            session = self.Session()
            # Check if test data already exists
            existing = session.query(Analytics).first()
            if not existing:
                test_analytics = Analytics(
                    timestamp=datetime.now(timezone.utc),
                    people_count=0,
                    crowd_density="Low",
                    breach_count=0,
                    unusual_behavior=False,
                    behavior_details=json.dumps([])
                )
                session.add(test_analytics)
                
                test_perimeter = Perimeter(
                    name="Test Perimeter",
                    points=json.dumps([[0,0], [100,0], [100,100], [0,100]]),
                    timestamp=datetime.now(timezone.utc)
                )
                session.add(test_perimeter)
                session.commit()
                logger.info("Test data initialized")
        except Exception as e:
            logger.error(f"Failed to initialize test data: {e}")
        finally:
            session.close()
    
    def initialize_camera(self):
        """Initialize Camera OR Kafka Consumer based on config"""
        if self.use_kafka:
            print("Connecting to Kafka...")
            try:
                # Simple Consumer Setup
                self.kafka_consumer = KafkaConsumer(
                    CONFIG.get('KAFKA_TOPIC', 'video_stream2'),
                    bootstrap_servers=CONFIG.get('KAFKA_BOOTSTRAP_SERVERS', ['127.0.0.1:9092']), # Force IPv4
                    auto_offset_reset='latest',           # Start at end of stream
                    group_id=None,                        # No group = No committing offsets = Faster
                    fetch_max_bytes=10 * 1024 * 1024     # Allow 10MB frames
                )
                print("Kafka Connected!")
                return True
            except Exception as e:
                print(f"Kafka Error: {e}")
                return False
        else:
            logger.info("Initializing Camera mode...")
            ip_cam_url = CONFIG.get('IP_WEBCAM_URL', '0')
            self.cap = cv2.VideoCapture(ip_cam_url)
            
            if not self.cap.isOpened():
                logger.warning(f"Failed to connect to IP webcam at {ip_cam_url}")
                self.cap = cv2.VideoCapture(0)
                if not self.cap.isOpened():
                    logger.error("Failed to open any camera source")
                    self.cap = None
                    return False
            
            logger.info("Camera initialized successfully")
            return True
        
    def read_frame(self):
        """Unified interface to read from Camera or Kafka"""
        if self.use_kafka:
            if not self.kafka_consumer: return False, None
            
            # POLL: Check for new data (Timeout 20ms)
            raw_msgs = self.kafka_consumer.poll(timeout_ms=20)
            
            if not raw_msgs:
                return False, None

            # Get the very last message from the batch
            for tp, messages in raw_msgs.items():
                if messages:
                    last_msg = messages[-1]
                    # Decode
                    nparr = np.frombuffer(last_msg.value, np.uint8)
                    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    return True, frame
            
            return False, None
        else:
            if self.cap and self.cap.isOpened():
                return self.cap.read()
            return False, None
        
    def is_source_ready(self):
        """
        Checks if the currently selected source (Kafka or Camera) is valid.
        """
        if self.use_kafka:
            return self.kafka_consumer is not None
        else:
            return self.cap is not None and self.cap.isOpened()
    
    def initialize_mqtt(self):
        """Initialize MQTT client"""
        self.mqtt_client = mqtt.Client(client_id="perimeter_detection_backend")
        
        def on_connect(client, userdata, flags, rc):
            if rc == 0:
                logger.info("Successfully connected to MQTT broker")
            else:
                logger.error(f"Failed to connect to MQTT broker with code: {rc}")
        
        def on_disconnect(client, userdata, rc):
            logger.warning(f"Disconnected from MQTT broker with code: {rc}")
        
        self.mqtt_client.on_connect = on_connect
        self.mqtt_client.on_disconnect = on_disconnect
        
        try:
            broker = CONFIG.get('MQTT_BROKER', 'broker.emqx.io')
            port = CONFIG.get('MQTT_PORT', 1883)
            self.mqtt_client.connect(broker, port, 60)
            self.mqtt_client.loop_start()
            logger.info("MQTT client started")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to MQTT broker: {e}")
            return False
    
    def initialize_ai_models(self):
        """Initialize YOLO model and SORT tracker and detect device (GPU/CPU)"""
        try:
            use_cuda = torch.cuda.is_available()
            self.device = "cuda" if use_cuda else "cpu"
            logger.info(f"torch.cuda.is_available(): {use_cuda}; using device: {self.device}")

            # load model (decide based on extension)
            model_path = CONFIG.get('MODEL_PATH', str(PROJECT_ROOT / 'assets' / 'yolov8l.pt'))
            model_path_obj = Path(model_path)

            if model_path_obj.suffix.lower() == '.engine':
                logger.info("Detected TensorRT engine file. Skipping ultralytics loader.")
                self.model_type = 'tensorrt'
                self.model = YOLO(model_path)
            else:
                logger.info(f"Loading PyTorch model at {model_path}")
                self.model_type = 'pytorch'
                self.model = YOLO(model_path)
                self.model.fuse()

                try:
                    if use_cuda:
                        self.model.to('cuda:0')
                    else:
                        self.model.to('cpu')
                except Exception:
                    logger.debug("Failed to set model device; continuing")

            # initialize tracker
            self.tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

            # Load a single canonical classes file under assets: 'assets/model_classes.json'
            class_names_file = PROJECT_ROOT / 'assets' / 'model_classes.json'
            names_mapping = None
            if class_names_file.exists():
                try:
                    names_mapping = json.loads(class_names_file.read_text(encoding='utf-8'))
                    logger.info(f"Loaded class names from {class_names_file}")
                except Exception as e:
                    logger.warning(f"Failed to load {class_names_file}: {e}")

            # If missing and we have a PyTorch model, try extracting and save with string keys
            if not names_mapping and getattr(self, 'model_type', '') == 'pytorch' and self.model is not None:
                try:
                    extracted = getattr(self.model, 'model', None)
                    if extracted is not None and hasattr(extracted, 'names'):
                        names_mapping = {str(k): v for k, v in extracted.names.items()}
                        try:
                            class_names_file.write_text(json.dumps(names_mapping, indent=2), encoding='utf-8')
                            logger.info(f"Saved extracted class names to {class_names_file}")
                        except Exception:
                            logger.debug("Could not save extracted class names")
                except Exception as e:
                    logger.warning(f"Failed to extract names from model: {e}")

            # Final fallback
            if not names_mapping:
                logger.warning("No class names found; creating numeric fallback mapping")
                names_mapping = {"0": "person"}
                try:
                    class_names_file.write_text(json.dumps(names_mapping, indent=2), encoding='utf-8')
                except Exception:
                    pass

            # Keep mapping with string keys (so detect_and_track can do str lookups reliably)
            self.CLASS_NAMES_DICT = {str(k): v for k, v in names_mapping.items()}

            # self.CLASS_NAMES_DICT = {"0": "person"}  # Forcing only person class for this application

            # expose to analytics state
            app_state.analytics["device"] = self.device

            logger.info("AI models initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize AI models: {e}")
            return False
    
    def update_fps(self):
        """Calculates FPS and updates app_state"""
        self.fps_window_count += 1
        now = time.time()
        elapsed = now - self.fps_window_start
        if elapsed >= 1.0:
            self.last_fps = self.fps_window_count / elapsed
            self.fps_window_count = 0
            self.fps_window_start = now
            # Update the global state directly
            app_state.analytics["fps"] = round(self.last_fps, 2)

    def cleanup(self):
        """Clean up all resources"""
        if self.cap and self.cap.isOpened():
            self.cap.release()
        if self.mqtt_client:
            self.mqtt_client.loop_stop()
            self.mqtt_client.disconnect()
        if self.kafka_consumer: 
            self.kafka_consumer.close() 
        logger.info("Resources cleaned up")

# Create global resource manager
resource_manager = ResourceManager()

# For handling MQTT messages
class MQTTHandler:
    @staticmethod
    def send_buzzer_signal(state: bool):
        """Simplified MQTT signal sending"""
        if not resource_manager.mqtt_client:
            logger.error("MQTT client not initialized")
            return False
        
        try:
            message = "ON" if state else "OFF"
            result = resource_manager.mqtt_client.publish("esp32/buzzer", message)
            success = result.rc == mqtt.MQTT_ERR_SUCCESS
            logger.info(f"Buzzer {message} signal sent: {'success' if success else 'failed'}")
            return success
        except Exception as e:
            logger.error(f"Failed to send MQTT message: {e}")
            return False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Clean, standard lifecycle management. 
    Let Uvicorn handle the signals; we just handle resources.
    """
    global SERVER_RUNNING
    
    # --- STARTUP ---
    logger.info("Application starting up...")
    SERVER_RUNNING = True
    
    # Initialize resources
    resource_manager.initialize_database()
    resource_manager.initialize_camera()
    resource_manager.initialize_mqtt()
    resource_manager.initialize_ai_models()
    
    yield  # The application runs here
    
    # --- SHUTDOWN ---
    logger.info("Application shutting down...")
    SERVER_RUNNING = False  # Tells video_feed loop to break immediately
    
    # Allow a brief moment for the loop to break before destroying resources
    await asyncio.sleep(0.5) 
    
    resource_manager.cleanup()
    logger.info("Shutdown complete.")

def get_session():
    """Database session dependency"""
    if not resource_manager.Session:
        raise HTTPException(status_code=503, detail="Database not initialized")
    
    session = resource_manager.Session()
    try:
        yield session
    except Exception as e:
        session.rollback()
        logger.error(f"Database session error: {e}")
        raise
    finally:
        session.close()


# App initialization
app = FastAPI(lifespan=lifespan)
# Use absolute paths from CONFIG so app works when run from src or project root
templates = Jinja2Templates(directory=CONFIG.get('TEMPLATES_DIR', str(PROJECT_ROOT / 'templates')))
app.mount("/static", StaticFiles(directory=CONFIG.get('STATIC_DIR', str(PROJECT_ROOT / 'static'))), name="static")
SessionDep = Annotated[Session, Depends(get_session)]
Base = declarative_base()

class Analytics(Base):
    __tablename__ = 'analytics'
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime)
    people_count = Column(Integer)
    crowd_density = Column(String)
    breach_count = Column(Integer)
    unusual_behavior = Column(Boolean)
    behavior_details = Column(String)  

class Incident(Base):
    __tablename__ = 'incidents'
    id = Column(Integer, primary_key=True)
    type = Column(String)
    details = Column(String)
    timestamp = Column(DateTime)

class Perimeter(Base):
    __tablename__ = 'perimeters'
    id = Column(Integer, primary_key=True)
    name = Column(String)
    points = Column(String)  
    timestamp = Column(DateTime)

# Saving Data to database
def save_analytics_to_db_sync(analytics_data: Dict[str, Any]):
    """Synchronous version for background tasks"""
    try:
        session = resource_manager.Session()
        try:
            analytics_record = Analytics(
                timestamp=datetime.now(timezone.utc),
                people_count=analytics_data['peopleCount'],
                crowd_density=analytics_data['crowdDensity'],
                breach_count=analytics_data['breachCount'],
                unusual_behavior=analytics_data['unusualBehavior'],
                behavior_details=json.dumps(analytics_data['behaviorDetails'])
            )
            session.add(analytics_record)
            session.commit()
            logger.info("Analytics saved successfully")
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to save analytics: {e}")
        finally:
            session.close()
    except Exception as e:
        logger.error(f"Database connection error: {e}")

def save_incident_to_db_sync(incident_type: str, details: str):
    """Synchronous version for background tasks"""
    try:
        session = resource_manager.Session()
        try:
            incident = Incident(
                type=incident_type,
                details=details,
                timestamp=datetime.now(timezone.utc)
            )
            session.add(incident)
            session.commit()
            logger.info("Incident saved successfully")
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to save incident: {e}")
        finally:
            session.close()
    except Exception as e:
        logger.error(f"Database connection error: {e}")

def save_data_sync(incidents, analytics_data):
    """Synchronous version for background tasks - calls both save incidents and save analytics"""
    try:
        # Save incidents
        for incident_type, details in incidents:
            save_incident_to_db_sync(incident_type, details)
        
        # Save analytics 
        if incidents or analytics_data["unusualBehavior"]:
            save_analytics_to_db_sync(analytics_data)
            
        logger.info("Data saved successfully in background")
    except Exception as e:
        logger.error(f"Error saving data in background: {e}")


# Analytics and Detection Utilities
def point_in_polygon(point, polygon_points):
    if not polygon_points:
        return False
    
    polygon = Polygon(polygon_points)
    point = Point(point)
    return polygon.contains(point)

def detect_and_track(frame):
    try:
        logger.debug("Starting object detection and tracking")
        detections = np.empty((0, 5))
        # results = resource_manager.model(frame, stream=True)
        results = resource_manager.model(
            frame, 
            stream=True, 
            classes=[0],      # 0 is 'person' in COCO dataset
            imgsz=640,        # Lower resolution for inference (faster)
            conf=0.4,          # Slightly higher confidence threshold to reduce noise
            device = 0
        )

        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                cls = box.cls[0]
                currentClass = resource_manager.CLASS_NAMES_DICT.get(str(cls), f"person")
                conf = math.ceil(box.conf[0] * 100) / 100

                if conf > 0.5 and currentClass == "person":
                    currentArray = np.array([x1, y1, x2, y2, conf])
                    detections = np.vstack((detections, currentArray))

        resultTracker = resource_manager.tracker.update(detections)
        tracked_objects = []

        for res in resultTracker:
            x1, y1, x2, y2, id = res
            x1, y1, x2, y2, id = int(x1), int(y1), int(x2), int(y2), int(id)
            w, h = x2 - x1, y2 - y1

            tracked_objects.append({
                'id': id,
                'bbox': [x1, y1, x2, y2],
                'center': (x1 + w // 2, y1 + h // 2),
                'prev_center': None  # Initialize prev_center
            })

        logger.debug(f"Detected {len(tracked_objects)} objects")
        return tracked_objects

    except Exception as e:
        logger.error(f"Error in detect_and_track: {str(e)}")
        return []

def calculate_crowd_density(people_count):
    if people_count <= 3:
        return "Low"
    elif people_count <= 6:
        return "Medium"
    else:
        return "High"
    
def detect_unusual_behavior(tracked_objects, frame):
    try:
        unusual_behaviors = []
        
        if len(tracked_objects) >= 2:
            # high Velocity detection
            velocities = []
            for obj in tracked_objects:
                if obj.get('prev_center'):
                    dx = obj['center'][0] - obj['prev_center'][0]
                    dy = obj['center'][1] - obj['prev_center'][1]
                    velocity = math.sqrt(dx*dx + dy*dy)
                    velocities.append(velocity)
                obj['prev_center'] = obj['center']

            if velocities and max(velocities) > 50:
                unusual_behaviors.append("Rapid movement detected")
                logger.info("Unusual behavior: Rapid movement detected")

        # Clustering detection
        if len(tracked_objects) > 2:
            centers = np.array([obj['center'] for obj in tracked_objects])
            distances = []
            for i in range(len(centers)):
                for j in range(i+1, len(centers)):
                    dist = np.linalg.norm(centers[i] - centers[j])
                    distances.append(dist)
            
            if distances and min(distances) < 50:
                unusual_behaviors.append("Unusual clustering detected")
                logger.info("Unusual behavior: Clustering detected")

        return unusual_behaviors

    except Exception as e:
        logger.error(f"Error in detect_unusual_behavior: {str(e)}")
        return []

def check_region(tracked_objects, polygon_points, frame):
    person_in_region = False
    incidents_to_save = [] 
    
    # Update basic analytics using app_state
    app_state.analytics["peopleCount"] = len(tracked_objects)
    app_state.analytics["crowdDensity"] = calculate_crowd_density(len(tracked_objects))
    
    # Detect unusual behavior
    unusual_behaviors = detect_unusual_behavior(tracked_objects, frame)
    app_state.analytics["unusualBehavior"] = len(unusual_behaviors) > 0
    app_state.analytics["behaviorDetails"] = unusual_behaviors

    # Draw and check polygon
    if polygon_points:
        points = np.array(polygon_points, np.int32)
        cv2.polylines(frame, [points], True, (0, 0, 255), 3)
        overlay = frame.copy()
        cv2.fillPoly(overlay, [points], (0, 0, 255))
        cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)

    # Process tracked objects
    for obj in tracked_objects:
        id = obj['id']
        bbox = obj['bbox']
        center = obj['center']

        x1, y1, x2, y2 = bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'ID: {id}', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        if polygon_points and point_in_polygon(center, polygon_points):
            person_in_region = True
            app_state.analytics["breachCount"] += 1
            cv2.putText(frame, 'ALERT: Breach', (50, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            app_state.buzzer_on = True
            MQTTHandler.send_buzzer_signal(True)
            incidents_to_save.append(("breach", f"Person ID {id} breached perimeter"))

    return person_in_region, incidents_to_save

def process_frame_pipeline(frame):
    """
    Runs Detection, Drawing, AND Encoding in a single synchronous block.
    This runs inside one worker thread to minimize context switching.
    """
    tracked_objects = detect_and_track(frame)
    person_in_region, incidents = check_region(tracked_objects, app_state.shape_coordinates, frame)
    ret, buffer = cv2.imencode('.jpg', frame)
    return buffer, incidents, tracked_objects


# Web app routes
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html",
                                      {"request": request})

@app.get("/video_feed")
async def video_feed(request: Request, background_tasks: BackgroundTasks):
    async def generate():
        while SERVER_RUNNING:
            try:
                if await request.is_disconnected():
                    logger.info("Client disconnected from video feed")
                    break

                if not resource_manager.is_source_ready():
                    # Generate Error Frame
                    blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                    msg = "Kafka Connecting..." if resource_manager.use_kafka else "Camera Not Available"
                    cv2.putText(blank_frame, msg, (150, 240),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    
                    ret, buffer = await asyncio.to_thread(cv2.imencode, '.jpg', blank_frame)
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                    
                    await asyncio.sleep(1)
                    continue

                success, frame = await asyncio.to_thread(resource_manager.read_frame)

                if not success or frame is None:
                    # If Kafka queue is empty or Camera fails, wait briefly
                    await asyncio.sleep(0.01)
                    continue

                buffer, incidents, tracked_objects = await asyncio.to_thread(process_frame_pipeline, frame)

                if incidents or app_state.analytics.get("unusualBehavior"):
                    background_tasks.add_task(
                        save_data_sync,
                        incidents,
                        app_state.analytics.copy()
                    )

                resource_manager.update_fps()

                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

                # await asyncio.sleep(0) allows other routes (like /toggle_buzzer) 
                await asyncio.sleep(0)

            except Exception as e:
                logger.error(f"Error in video feed loop: {e}")
                # Backoff slightly on error to prevent log spamming
                await asyncio.sleep(0.1)

        logger.info("Video feed generator exiting")
        
    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.post("/toggle_buzzer")
def toggle_buzzer():
    app_state.buzzer_on = not app_state.buzzer_on
    success = MQTTHandler.send_buzzer_signal(app_state.buzzer_on)
    return JSONResponse({
        "buzzer_on": app_state.buzzer_on, 
        "mqtt_success": success })

@app.get("/get_analytics")
async def get_analytics():
    return JSONResponse(app_state.analytics)

@app.post("/set_shape")
async def set_shape(data: dict = Body(...)):
    app_state.shape_coordinates = [(p['x'], p['y']) for p in data['points']]
    return {"status": "success", "message": "Perimeter shape updated"}

@app.get("/get_buzzer_state")
async def get_buzzer_state():
    """Get current buzzer state"""
    return JSONResponse({"buzzer_on": app_state.buzzer_on})

@app.post("/save_perimeter")
def save_perimeter(data: dict = Body(...), session: SessionDep = None):
    """Save perimeter configuration"""
    try:
        perimeter = Perimeter(
            name=data['name'],
            points=json.dumps(data['points']),
            timestamp=datetime.now(timezone.utc)
        )
        session.add(perimeter)
        session.commit()
        return {"status": "success", "message": "Perimeter saved successfully", "id": perimeter.id}
    except Exception as e:
        session.rollback()
        logger.error(f"Error saving perimeter: {e}")
        return {"status": "error", "message": str(e)}

@app.get("/get_perimeters")
def get_perimeters(session: SessionDep = None):
    """Get all saved perimeters"""
    try:
        perimeters = session.query(Perimeter).all()
        result = [{
            "id": p.id,
            "name": p.name,
            "points": json.loads(p.points),
            "timestamp": p.timestamp.isoformat() if p.timestamp else None
        } for p in perimeters]
        return {"status": "success", "perimeters": result}
    except Exception as e:
        logger.error(f"Error fetching perimeters: {e}")
        return {"status": "error", "message": str(e)}

@app.post("/load_perimeter/{perimeter_id}")
def load_perimeter(perimeter_id: int, session: SessionDep = None):
    """Load a saved perimeter and set it as current"""
    try:
        perimeter = session.query(Perimeter).filter(Perimeter.id == perimeter_id).first()
        if not perimeter:
            return {"status": "error", "message": "Perimeter not found"}
        
        app_state.shape_coordinates = json.loads(perimeter.points)
        return {
            "status": "success", 
            "message": "Perimeter loaded successfully",
            "perimeter": {
                "id": perimeter.id,
                "name": perimeter.name,
                "points": app_state.shape_coordinates
            }
        }
    except Exception as e:
        logger.error(f"Error loading perimeter: {e}")
        return {"status": "error", "message": str(e)}

@app.delete("/delete_perimeter/{perimeter_id}")
def delete_perimeter(perimeter_id: int, session: SessionDep = None):
    """Delete a saved perimeter"""
    try:
        perimeter = session.query(Perimeter).filter(Perimeter.id == perimeter_id).first()
        if not perimeter:
            return {"status": "error", "message": "Perimeter not found"}
        
        session.delete(perimeter)
        session.commit()
        return {"status": "success", "message": "Perimeter deleted successfully"}
    except Exception as e:
        session.rollback()
        logger.error(f"Error deleting perimeter: {e}")
        return {"status": "error", "message": str(e)}

@app.get("/get_perf")
async def get_perf():
    return JSONResponse({
        "device": getattr(resource_manager, "device", "unknown"),
        "fps": app_state.analytics.get("fps", 0) })


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, timeout_graceful_shutdown=0)

