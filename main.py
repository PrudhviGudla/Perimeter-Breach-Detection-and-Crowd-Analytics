from fastapi import FastAPI, Request, Body, Depends, BackgroundTasks
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
from sort import *
import uvicorn
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
import json
from typing import Annotated
import asyncio
import signal
import sys
import os

# Load environment variables
load_dotenv()

# logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log')
    ]
)
logger = logging.getLogger(__name__)

mqtt_client = mqtt.Client(client_id="perimeter_detection_backend")
mqtt_broker = 'broker.emqx.io'
mqtt_port = 1883
mqtt_topic = "esp32/buzzer"


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
        
    def initialize_database(self):
        """Initialize database connection and create tables"""
        try:
            db_path = 'perimeter_detection.db'
            self.engine = create_engine(f'sqlite:///{db_path}', echo=True)
            logger.info(f"Database path: {os.path.abspath(db_path)}")
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
        """Initialize camera connection"""
        ip_cam_url = os.getenv('IP_WEBCAM_URL', '0')
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
            self.mqtt_client.connect('broker.emqx.io', 1883, 60)
            self.mqtt_client.loop_start()
            logger.info("MQTT client started")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to MQTT broker: {e}")
            return False
    
    def initialize_ai_models(self):
        """Initialize YOLO model and SORT tracker"""
        try:
            self.model = YOLO("yolov8l.pt")
            self.model.fuse()
            self.tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
            self.CLASS_NAMES_DICT = self.model.model.names
            logger.info("AI models initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize AI models: {e}")
            return False
    
    def cleanup(self):
        """Clean up all resources"""
        if self.cap and self.cap.isOpened():
            self.cap.release()
        if self.mqtt_client:
            self.mqtt_client.loop_stop()
            self.mqtt_client.disconnect()
        logger.info("Resources cleaned up")

# Create global resource manager
resource_manager = ResourceManager()


shutdown_event = asyncio.Event()
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle with proper shutdown handling"""
    
    # Setup signal handlers that work with asyncio
    def signal_handler(sig, frame):
        logger.info(f"Received signal {sig}, initiating shutdown...")
        shutdown_event.set()
    
    # Install handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    if sys.platform.startswith('win'):
        signal.signal(signal.SIGBREAK, signal_handler)
    
    logger.info("Application starting up...")
    
    # Initialize resources
    resource_manager.initialize_database()
    resource_manager.initialize_camera()
    resource_manager.initialize_mqtt()
    resource_manager.initialize_ai_models()
    
    try:
        yield
    finally:
        # Cleanup
        logger.info("Application shutting down...")
        resource_manager.cleanup()
        
        # Cancel remaining tasks
        tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
        if tasks:
            logger.info(f"Cancelling {len(tasks)} outstanding tasks")
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)


# app initialization
app = FastAPI(lifespan=lifespan)
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")


def get_session():
    """Database session dependency"""
    if not resource_manager.Session:
        raise RuntimeError("Database not initialized")
    
    session = resource_manager.Session()
    try:
        yield session
    except Exception as e:
        session.rollback()
        logger.error(f"Database session error: {e}")
        raise
    finally:
        session.close()
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
    """Synchronous version for background tasks - THIS IS THE MAIN FUNCTION"""
    try:
        # Save incidents
        for incident_type, details in incidents:
            save_incident_to_db_sync(incident_type, details)
        
        # Save analytics if needed
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
        results = resource_manager.model(frame, stream=True)

        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                cls = int(box.cls[0])
                currentClass = resource_manager.CLASS_NAMES_DICT[cls]
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


# web app routes
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html",
                                      {"request": request})

@app.get("/video_feed")
def video_feed(background_tasks: BackgroundTasks):  # Add BackgroundTasks parameter
    def generate():
        while True:
            try:
                if not resource_manager.cap or not resource_manager.cap.isOpened():
                    # Return error frame
                    blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(blank_frame, "Camera Not Available", (180, 240),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    ret, buffer = cv2.imencode('.jpg', blank_frame)
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                    time.sleep(1)
                    continue

                # Read and process frame (synchronous)
                success, frame = resource_manager.cap.read()
                if not success:
                    continue

                tracked_objects = detect_and_track(frame)
                person_in_region, incidents = check_region(tracked_objects, app_state.shape_coordinates, frame)

                # Use BackgroundTasks instead of asyncio.create_task
                if incidents or app_state.analytics["unusualBehavior"]:
                    background_tasks.add_task(
                        save_data_sync, 
                        incidents, 
                        app_state.analytics.copy()
                    )

                # Encode and yield frame
                ret, buffer = cv2.imencode('.jpg', frame)
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                
                time.sleep(0.033)  # ~30 FPS
                
            except Exception as e:
                logger.error(f"Error processing frame: {e}")
                time.sleep(0.1)

    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.post("/toggle_buzzer")
async def toggle_buzzer():
    app_state.buzzer_on = not app_state.buzzer_on
    success = MQTTHandler.send_buzzer_signal(app_state.buzzer_on)
    return JSONResponse({
        "buzzer_on": app_state.buzzer_on, 
        "mqtt_success": success
    })

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
async def save_perimeter(data: dict = Body(...), session: SessionDep = None):
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
async def get_perimeters(session: SessionDep = None):
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
async def load_perimeter(perimeter_id: int, session: SessionDep = None):
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
async def delete_perimeter(perimeter_id: int, session: SessionDep = None):
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



if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
