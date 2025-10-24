"""
Real-Time IoMT Cyber Range Dashboard

Displays live telemetry and swarm algorithm status in real-time.
"""
from flask import Flask, render_template
from flask_socketio import SocketIO
import paho.mqtt.client as mqtt
import json
import logging
from datetime import datetime
from collections import deque
from threading import Thread

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'iomt-swarm-range-secret'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')

# Data storage (last 50 points per device)
device_data = {
    'ecg_001': deque(maxlen=50),
    'vent_001': deque(maxlen=50),
    'pump_001': deque(maxlen=50)
}

# Device status
device_status = {
    'ecg_001': {'online': False, 'last_seen': None},
    'vent_001': {'online': False, 'last_seen': None},
    'pump_001': {'online': False, 'last_seen': None}
}

# Swarm status (will be populated when swarm runs)
swarm_status = {
    'abc': {'active': False, 'coverage': 0, 'sensors': 0},
    'pso': {'active': False, 'fitness': 0, 'iteration': 0},
    'aco': {'active': False, 'best_cost': 0, 'paths': 0}
}

# MQTT Client
mqtt_client = None


def on_mqtt_connect(client, userdata, flags, reason_code, properties):
    """MQTT connection callback"""
    logger.info(f"Dashboard connected to MQTT: {reason_code}")
    client.subscribe("iomt/telemetry/#")
    client.subscribe("swarm/#")
    logger.info("Subscribed to iomt/telemetry/# and swarm/#")


def on_mqtt_message(client, userdata, msg):
    """MQTT message callback"""
    try:
        data = json.loads(msg.payload.decode())
        topic = msg.topic
        
        if topic.startswith("iomt/telemetry"):
            # Device telemetry
            device_id = data.get('device_id')
            if device_id in device_data:
                device_data[device_id].append(data)
                device_status[device_id]['online'] = True
                device_status[device_id]['last_seen'] = datetime.now()
                
                # Emit to web clients
                socketio.emit('telemetry_update', {
                    'device_id': device_id,
                    'data': data
                })
        
        elif topic.startswith("swarm/"):
            # Swarm algorithm updates
            algorithm = topic.split('/')[1]  # abc, pso, or aco
            if algorithm in swarm_status:
                swarm_status[algorithm].update(data)
                socketio.emit('swarm_update', {
                    'algorithm': algorithm,
                    'data': data
                })
                
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
def handle_connect():
    """Client connected to dashboard"""
    logger.info("Client connected to dashboard")
    
    # Send initial data
    socketio.emit('device_status', device_status)
    socketio.emit('swarm_status', swarm_status)


@socketio.on('disconnect')
def handle_disconnect():
    """Client disconnected from dashboard"""
    logger.info("Client disconnected from dashboard")


@socketio.on('request_history')
def handle_history_request(data):
    """Client requests historical data"""
    device_id = data.get('device_id')
    if device_id in device_data:
        history = list(device_data[device_id])
        socketio.emit('history_data', {
            'device_id': device_id,
            'history': history
        })


def run_dashboard(host='0.0.0.0', port=5000):
    """Run the dashboard server"""
    # Start MQTT client
    start_mqtt_client()
    
    # Start Flask app
    logger.info(f"Starting dashboard on http://{host}:{port}")
    socketio.run(app, host=host, port=port, debug=False)


if __name__ == '__main__':
    run_dashboard()
