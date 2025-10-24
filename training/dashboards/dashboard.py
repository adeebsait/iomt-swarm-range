"""
Real-Time IoMT Cyber Range Dashboard
"""
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import paho.mqtt.client as mqtt
import json
import logging
from datetime import datetime
from collections import deque
from threading import Lock

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'iomt-swarm-range-secret'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')

# Thread-safe data storage
data_lock = Lock()
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


def serialize_device_status(status):
    """Convert device status to JSON-serializable format"""
    serialized = {}
    for device_id, info in status.items():
        serialized[device_id] = {
            'online': info['online'],
            'last_seen': info['last_seen'].isoformat() if info['last_seen'] else None
        }
    return serialized


def on_mqtt_connect(client, userdata, flags, reason_code, properties):
    """MQTT connection callback"""
    logger.info(f"Dashboard connected to MQTT: {reason_code}")
    client.subscribe("iomt/telemetry/#")
    logger.info("Subscribed to iomt/telemetry/#")


def on_mqtt_message(client, userdata, msg):
    """MQTT message callback - CRITICAL: Must emit to socketio namespace"""
    try:
        data = json.loads(msg.payload.decode())
        device_id = data.get('device_id')
        
        if device_id in device_data:
            with data_lock:
                device_data[device_id].append(data)
                device_status[device_id]['online'] = True
                device_status[device_id]['last_seen'] = datetime.now()
            
            # FIXED: Use socketio.emit directly (not through Flask context)
            socketio.emit('telemetry_update', {
                'device_id': device_id,
                'data': data
            }, namespace='/')
            
            logger.debug(f"Emitted telemetry for {device_id}")
                
    except Exception as e:
        logger.error(f"Error processing message: {e}")


def start_mqtt_client():
    """Start MQTT client in background thread"""
    global mqtt_client
    
    mqtt_client = mqtt.Client(
        client_id="dashboard",
        protocol=mqtt.MQTTv5,
        callback_api_version=mqtt.CallbackAPIVersion.VERSION2
    )
    mqtt_client.on_connect = on_mqtt_connect
    mqtt_client.on_message = on_mqtt_message
    
    try:
        mqtt_client.connect("localhost", 1883, 60)
        mqtt_client.loop_start()
        logger.info("MQTT client started")
    except Exception as e:
        logger.error(f"Failed to connect to MQTT: {e}")


@app.route('/')
def index():
    """Dashboard home page"""
    return render_template('dashboard.html')


@socketio.on('connect')
def handle_connect(auth=None):
    """Client connected to dashboard"""
    logger.info("Client connected to dashboard")
    
    # Send initial status
    with data_lock:
        emit('device_status', serialize_device_status(device_status))


@socketio.on('disconnect')
def handle_disconnect():
    """Client disconnected from dashboard"""
    logger.info("Client disconnected from dashboard")


def run_dashboard(host='0.0.0.0', port=5000):
    """Run the dashboard server"""
    start_mqtt_client()
    logger.info(f"Starting dashboard on http://{host}:{port}")
    socketio.run(app, host=host, port=port, debug=False)


if __name__ == '__main__':
    run_dashboard()
