import cv2
import time
import json
from pathlib import Path
from kafka import KafkaProducer

# Load from project config.json (project root)
def load_kafka_config():
    project_root = Path(__file__).resolve().parent.parent
    config_path = project_root / 'config.json'
    defaults = {
        'KAFKA_BOOTSTRAP_SERVERS': ['127.0.0.1:9092'],
        'KAFKA_TOPIC': 'video_stream2',
        'IP_WEBCAM_URL': '0'
    }

    if config_path.exists():
        try:
            cfg = json.loads(config_path.read_text(encoding='utf-8'))
        except Exception:
            cfg = {}
    else:
        cfg = {}

    # merge defaults
    for k, v in defaults.items():
        cfg.setdefault(k, v)

    bservers = cfg.get('KAFKA_BOOTSTRAP_SERVERS')
    if isinstance(bservers, str):
        servers = [s.strip() for s in bservers.split(',') if s.strip()]
    elif isinstance(bservers, list):
        servers = bservers
    else:
        servers = defaults['KAFKA_BOOTSTRAP_SERVERS']

    topic = cfg.get('KAFKA_TOPIC', defaults['KAFKA_TOPIC'])
    ip_webcam_url = cfg.get('IP_WEBCAM_URL', defaults['IP_WEBCAM_URL'])
    return servers, topic, ip_webcam_url


def start_stream(BOOTSTRAP_SERVERS, TOPIC_NAME, IP_WEBCAM_URL):
    # Simple Producer
    producer = KafkaProducer(
        bootstrap_servers=BOOTSTRAP_SERVERS,
        batch_size=16384, # 16KB
        linger_ms=0, 
        compression_type='lz4'
    )

    if IP_WEBCAM_URL == '0':
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(IP_WEBCAM_URL)

    if not cap.isOpened():
        print("Error: Camera not found.")
        return

    print(f"Streaming to topic '{TOPIC_NAME}'...")

    try:
        while True:
            success, frame = cap.read()
            if not success:
                break
            
            ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
            
            if ret:
                producer.send(TOPIC_NAME, buffer.tobytes())
                
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        cap.release()
        producer.close()

if __name__ == "__main__":
    BOOTSTRAP_SERVERS, TOPIC_NAME, IP_WEBCAM_URL = load_kafka_config()
    start_stream(BOOTSTRAP_SERVERS, TOPIC_NAME, IP_WEBCAM_URL)