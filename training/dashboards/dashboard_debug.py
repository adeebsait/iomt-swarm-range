"""Debug version with extra logging"""
from flask import Flask, render_template
from flask_socketio import SocketIO
import paho.mqtt.client as mqtt
import json
import logging
from datetime import datetime
from collections import deque

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'iomt-swarm-range-secret'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')

device_data = {
    'ecg_001': deque(maxlen=50),
    'vent_001': deque(maxlen=50),
    'pump_001': deque(maxlen=50)
}

device_status = {
    'ecg_001': {'online': False, 'last_seen': None},
    'vent_001': {'online': False, 'last_seen': None},
    'pump_001': {'online': False, 'last_seen': None}
}

mqtt_client = None
message_count = 0

def on_mqtt_connect(client, userdata, flags, reason_code, properties):
    logger.info(f"MQTT Connected: {reason_code}")
    client.subscribe("iomt/telemetry/#")
    logger.info("Subscribed to iomt/telemetry/#")

def on_mqtt_message(client, userdata, msg):
    global message_count
    message_count += 1
    
    logger.info(f"Message #{message_count} on topic: {msg.topic}")
    
    try:
        data = json.loads(msg.payload.decode())
        device_id = data.get('device_id')
        
        logger.info(f"Device ID: {device_id}")
        
        if device_id in device_data:
            device_data[device_id].append(data)
            device_status[device_id]['online'] = True
            device_status[device_id]['last_seen'] = datetime.now()
            
            logger.info(f"Updated {device_id} to ONLINE")
            
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)

def start_mqtt_client():
    global mqtt_client
    
    mqtt_client = mqtt.Client(
        client_id="dashboard_debug",
        protocol=mqtt.MQTTv5,
        callback_api_version=mqtt.CallbackAPIVersion.VERSION2
    )
    mqtt_client.on_connect = on_mqtt_connect
    mqtt_client.on_message = on_mqtt_message
    
    try:
        logger.info("Connecting to MQTT at localhost:1883...")
        mqtt_client.connect("localhost", 1883, 60)
        mqtt_client.loop_start()
        logger.info("MQTT loop started")
    except Exception as e:
        logger.error(f"MQTT connection failed: {e}")

@app.route('/')
def index():
    return "<h1>Debug Dashboard</h1><p>Check terminal for logs</p>"

def run_dashboard(host='0.0.0.0', port=5001):
    start_mqtt_client()
    logger.info(f"Starting DEBUG dashboard on http://{host}:{port}")
    socketio.run(app, host=host, port=port, debug=False)

if __name__ == '__main__':
    run_dashboard()
