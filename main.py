from fastapi import FastAPI, Request, Body  #, BackgroundTasks
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles  
import cv2
import numpy as np
# import threading
# import requests
import time
import math
import paho.mqtt.client as mqtt
import logging
from shapely.geometry import Point, Polygon
import asyncio
from motor.motor_asyncio import AsyncIOMotorClient
from datetime import datetime, timezone
from typing import Dict, Any
import os
from dotenv import load_dotenv
from pymongo import MongoClient
from contextlib import asynccontextmanager
from ultralytics import YOLO
from sort import *


# Setup function
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Application starting up...")
    try:
        # Initialize MongoDB connection
        await db_client.client.admin.command("ping")
        logger.info("Successfully connected to MongoDB")
    except Exception as e:
        logger.error(f"Failed to connect to MongoDB: {e}")
    
    yield
    # Shutdown
    logger.info("Application shutting down...")
    if cap.isOpened():
        cap.release()
    mqtt_client.loop_stop()
    mqtt_client.disconnect()
    if db_client:
        db_client.client.close()

app = FastAPI(lifespan=lifespan)
templates = Jinja2Templates(directory="templates")

# Mount the static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log')
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# MongoDB setup
MONGO_URI = os.getenv('MONGO_URI', 'mongodb://localhost:27017')

class Database:
    client: AsyncIOMotorClient = None
    db = None

    @classmethod
    def get_database(cls):
        if not cls.client:
            try:
                cls.client = AsyncIOMotorClient(MONGO_URI, maxPoolSize=10)
                cls.db = cls.client.perimeter_detection
                logger.info("MongoDB client initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize MongoDB client: {e}")
        return cls.db

# Initialize database and collections
db_client = Database()
db = db_client.get_database()
analytics_collection = db['analytics']
incidents_collection = db['incidents']
perimeters_collection = db['perimeters']

# Global variables
shape_coordinates = None
buzzer_on = True
set_line_mode = False
mqtt_client = mqtt.Client(client_id="perimeter_detection_backend")
mqtt_broker = 'broker.emqx.io'
mqtt_port = 1883
mqtt_topic = "esp32/buzzer"
analytics = {
    "peopleCount": 0,
    "crowdDensity": "Low",
    "breachCount": 0,
    "unusualBehavior": False,
    "behaviorDetails": []
}

# Initialize YOLOv8 model
model = YOLO("yolov8l.pt")
model.fuse()
CLASS_NAMES_DICT = model.model.names

# Initialize SORT tracker
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

# Video capture
cap = cv2.VideoCapture(0)

#----------------------------------------------------------------------------------------
# Add MQTT callbacks
def on_connect(client, userdata, flags, rc):
    if rc == 0:
        logging.info("Successfully connected to MQTT broker")
        client.subscribe(mqtt_topic)
    else:
        logging.error(f"Failed to connect to MQTT broker with code: {rc}")

def on_disconnect(client, userdata, rc):
    logging.warning(f"Disconnected from MQTT broker with code: {rc}")

def on_publish(client, userdata, mid):
    logging.debug(f"Message {mid} published successfully")

mqtt_client.on_connect = on_connect
mqtt_client.on_disconnect = on_disconnect
mqtt_client.on_publish = on_publish

# Update MQTT connection with better error handling
try:
    mqtt_client.connect(mqtt_broker, mqtt_port, 60)
    mqtt_client.loop_start()
    logging.info("MQTT client started")
except Exception as e:
    logging.error(f"Failed to connect to MQTT broker: {e}")

#----------------------------------------------------------------------------------

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
        results = model(frame, stream=True)

        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                cls = int(box.cls[0])
                currentClass = CLASS_NAMES_DICT[cls]
                conf = math.ceil(box.conf[0] * 100) / 100

                if conf > 0.5 and currentClass == "person":
                    currentArray = np.array([x1, y1, x2, y2, conf])
                    detections = np.vstack((detections, currentArray))

        resultTracker = tracker.update(detections)
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

#---------------------------------------------------------------------------------
async def save_analytics_to_db(analytics_data: Dict[str, Any]):
    try:
        analytics_data['timestamp'] = datetime.now(timezone.utc)
        result = await analytics_collection.insert_one(analytics_data)
        if result.inserted_id:
            logger.info(f"Analytics saved with ID: {result.inserted_id}")
        else:
            logger.error("Failed to save analytics")
    except Exception as e:
        logger.error(f"Failed to save analytics to database: {e}")

async def save_incident_to_db(incident_type: str, details: str):
    try:
        incident = {
            'type': incident_type,
            'details': details,
            'timestamp': datetime.now(timezone.utc)
        }
        result = await incidents_collection.insert_one(incident)
        if result.inserted_id:
            logger.info(f"Incident saved with ID: {result.inserted_id}")
        else:
            logger.error("Failed to save incident")
    except Exception as e:
        logger.error(f"Failed to save incident to database: {e}")

async def save_data(incidents, analytics_data):
    try:
        # Save incidents
        for incident_type, details in incidents:
            await save_incident_to_db(incident_type, details)
        
        # Save analytics if there are incidents or unusual behavior
        if incidents or analytics_data["unusualBehavior"]:
            await save_analytics_to_db(analytics_data.copy())
            
        logger.info("Data saved successfully")
    except Exception as e:
        logger.error(f"Error saving data to database: {e}")

#--------------------------------------------------------------------------------------

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
    global buzzer_on, analytics
    person_in_region = False
    incidents_to_save = [] 
    
    # Update basic analytics
    analytics["peopleCount"] = len(tracked_objects)
    analytics["crowdDensity"] = calculate_crowd_density(len(tracked_objects))
    
    # Detect unusual behavior
    unusual_behaviors = detect_unusual_behavior(tracked_objects, frame)
    analytics["unusualBehavior"] = len(unusual_behaviors) > 0
    analytics["behaviorDetails"] = unusual_behaviors

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
            analytics["breachCount"] += 1
            cv2.putText(frame, 'ALERT: Breach', (50, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            buzzer_on = True
            send_buzzer_signal()
            incidents_to_save.append(("breach", f"Person ID {id} breached perimeter"))

    return person_in_region, incidents_to_save

def send_buzzer_signal():
    try:
        result = mqtt_client.publish(mqtt_topic, "ON")
        if result.rc == mqtt.MQTT_ERR_SUCCESS:
            logging.info("Buzzer ON signal sent successfully")
        else:
            logging.error(f"Failed to send buzzer ON signal, error code: {result.rc}")
    except Exception as e:
        logging.error(f"Failed to send MQTT message: {e}")

def send_buzzer_off_signal():
    try:
        result = mqtt_client.publish(mqtt_topic, "OFF")
        if result.rc == mqtt.MQTT_ERR_SUCCESS:
            logging.info("Buzzer OFF signal sent successfully")
        else:
            logging.error(f"Failed to send buzzer OFF signal, error code: {result.rc}")
    except Exception as e:
        logging.error(f"Failed to send MQTT message: {e}")

#---------------------------------------------------------------------------------------------------------------------
def generate_frames():
    global shape_coordinates, buzzer_on
    
    async def process_frame():
        while True:
            success, frame = cap.read()
            if not success:
                break

            tracked_objects = detect_and_track(frame)
            person_in_region, incidents = check_region(tracked_objects, shape_coordinates, frame)

            # Create a background task for saving data
            if incidents or analytics["unusualBehavior"]:
                asyncio.create_task(save_data(incidents, analytics.copy()))

            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            await asyncio.sleep(0.03)

    return StreamingResponse(process_frame(), media_type="multipart/x-mixed-replace; boundary=frame")

#-----------------------------------------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html",
                                      {"request": request})

@app.get("/video_feed")
def video_feed():
    return generate_frames()

@app.post("/set_shape")
async def set_shape(data: dict = Body(...)):
    global shape_coordinates
    shape_coordinates = [(p['x'], p['y']) for p in data['points']]
    return {"status": "success"}

@app.post("/toggle_buzzer")
async def toggle_buzzer():
    global buzzer_on
    buzzer_on = not buzzer_on
    if buzzer_on:
        send_buzzer_signal()
    else:
        send_buzzer_off_signal()
    return JSONResponse({"buzzer_on": buzzer_on})

@app.get("/get_buzzer_state")
async def get_buzzer_state():
    global buzzer_on
    return JSONResponse({"buzzer_on": buzzer_on})

@app.get("/get_analytics")
async def get_analytics():
    return JSONResponse(analytics)

@app.post("/save_perimeter")
async def save_perimeter(data: dict = Body(...)):
    try:
        perimeter = {
            'name': data['name'],
            'points': data['points'],
            'timestamp': datetime.now(timezone.utc)
        }
        result = await perimeters_collection.insert_one(perimeter)
        if result.inserted_id:
            logger.info(f"Perimeter saved with ID: {result.inserted_id}")
            return {"status": "success", "message": "Perimeter saved successfully", "id": str(result.inserted_id)}
        else:
            return {"status": "error", "message": "Failed to save perimeter"}
    except Exception as e:
        logger.error(f"Error saving perimeter: {e}")
        return {"status": "error", "message": str(e)}

@app.get("/get_perimeters")
async def get_perimeters():
    try:
        perimeters = await perimeters_collection.find().to_list(length=None)
        logger.info(f"Retrieved {len(perimeters)} perimeters")
        return [{"id": str(p["_id"]), "name": p["name"], "points": p["points"]} for p in perimeters]
    except Exception as e:
        logger.error(f"Error fetching perimeters: {e}")
        return []

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
