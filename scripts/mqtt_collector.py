"""
Simple MQTT Telemetry Collector

Subscribes to all IoMT telemetry and saves to CSV for analysis.
Uses separate files per device type to avoid field conflicts.
"""
import json
import csv
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict
import paho.mqtt.client as mqtt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Output directory
OUTPUT_DIR = Path("results/raw")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# CSV writers for each device type
csv_files: Dict[str, any] = {}
csv_writers: Dict[str, any] = {}

def get_csv_writer(device_type: str):
    """Get or create CSV writer for device type"""
    if device_type not in csv_writers:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = OUTPUT_DIR / f"{device_type}_{timestamp}.csv"
        
        csv_files[device_type] = open(filename, 'w', newline='')
        logger.info(f"Created output file: {filename}")
    
    return csv_files[device_type], csv_writers.get(device_type)

def flatten_telemetry(data: dict) -> dict:
    """Flatten nested telemetry data for CSV"""
    device_type = data.get("device_type")
    
    flat_data = {
        "timestamp": data.get("timestamp"),
        "device_id": data.get("device_id"),
        "device_type": device_type,
        "ward": data.get("ward")
    }
    
    # Add device-specific fields
    if device_type == "ecg_monitor":
        vital_signs = data.get("vital_signs", {})
        flat_data.update({
            "heart_rate_bpm": vital_signs.get("heart_rate_bpm"),
            "spo2_percent": vital_signs.get("spo2_percent"),
            "systolic_bp_mmhg": vital_signs.get("systolic_bp_mmhg"),
            "diastolic_bp_mmhg": vital_signs.get("diastolic_bp_mmhg"),
            "temperature_c": vital_signs.get("temperature_c"),
            "respiratory_rate": vital_signs.get("respiratory_rate"),
            "alarm_active": data.get("alarm_active"),
            "alarm_type": data.get("alarm_type")
        })
    
    elif device_type == "ventilator":
        measurements = data.get("measurements", {})
        settings = data.get("settings", {})
        status = data.get("status", {})
        flat_data.update({
            "mode": settings.get("mode"),
            "tidal_volume_ml": settings.get("tidal_volume_ml"),
            "respiratory_rate": settings.get("respiratory_rate"),
            "fio2_percent": settings.get("fio2_percent"),
            "minute_volume_l": measurements.get("minute_volume_l"),
            "peak_pressure_cmh2o": measurements.get("peak_pressure_cmh2o"),
            "spo2_percent": measurements.get("spo2_percent"),
            "etco2_mmhg": measurements.get("etco2_mmhg"),
            "alarm_active": status.get("alarm_active"),
            "alarm_type": status.get("alarm_type")
        })
    
    elif device_type == "infusion_pump":
        flat_data.update({
            "status": data.get("status"),
            "infusion_rate_ml_hr": data.get("infusion_rate_ml_hr"),
            "volume_infused_ml": data.get("volume_infused_ml"),
            "volume_remaining_ml": data.get("volume_remaining_ml"),
            "battery_percent": data.get("battery_percent"),
            "medication": data.get("medication"),
            "patient_id": data.get("patient_id"),
            "alarm_active": data.get("alarm_active"),
            "alarm_type": data.get("alarm_type")
        })
    
    elif device_type == "dicom_pacs":
        storage = data.get("storage", {})
        status = data.get("status", {})
        flat_data.update({
            "storage_used_gb": storage.get("used_gb"),
            "storage_total_gb": storage.get("total_gb"),
            "storage_percent": storage.get("percent_used"),
            "active_connections": status.get("active_connections"),
            "operations_today": status.get("operations_today")
        })
    
    elif device_type == "fhir_gateway":
        metrics = data.get("metrics", {})
        status = data.get("status", {})
        flat_data.update({
            "connected_devices": status.get("connected_devices"),
            "requests_today": metrics.get("requests_today"),
            "observations_created": metrics.get("observations_created"),
            "errors_today": metrics.get("errors_today"),
            "error_rate_percent": metrics.get("error_rate_percent")
        })
    
    return flat_data

def on_connect(client, userdata, flags, reason_code, properties):
    """Callback when connected to broker"""
    logger.info(f"Connected to MQTT broker: {reason_code}")
    client.subscribe("iomt/telemetry/#")
    logger.info("Subscribed to iomt/telemetry/#")

def on_message(client, userdata, msg):
    """Callback when message received"""
    try:
        # Parse JSON payload
        data = json.loads(msg.payload.decode())
        device_type = data.get("device_type")
        
        if not device_type:
            return
        
        # Flatten data
        flat_data = flatten_telemetry(data)
        
        # Get or create CSV writer for this device type
        csv_file, csv_writer = get_csv_writer(device_type)
        
        # Create writer if needed
        if csv_writer is None:
            csv_writer = csv.DictWriter(csv_file, fieldnames=flat_data.keys())
            csv_writer.writeheader()
            csv_writers[device_type] = csv_writer
        
        # Write row
        csv_writer.writerow(flat_data)
        csv_file.flush()
        
        logger.debug(f"Recorded: {data['device_id']} ({device_type})")
        
    except Exception as e:
        logger.error(f"Error processing message: {e}")

def main():
    """Main entry point"""
    client = mqtt.Client(
        client_id="telemetry_collector",
        protocol=mqtt.MQTTv5,
        callback_api_version=mqtt.CallbackAPIVersion.VERSION2
    )
    
    client.on_connect = on_connect
    client.on_message = on_message
    
    try:
        logger.info("Connecting to MQTT broker at localhost:1883...")
        client.connect("localhost", 1883, 60)
        
        logger.info("Starting data collection (Press Ctrl+C to stop)...")
        client.loop_forever()
        
    except KeyboardInterrupt:
        logger.info("\nStopping data collection...")
        for csv_file in csv_files.values():
            csv_file.close()
        logger.info(f"Data saved to {OUTPUT_DIR}")
    except Exception as e:
        logger.error(f"Error: {e}")
    finally:
        client.disconnect()

if __name__ == "__main__":
    main()
