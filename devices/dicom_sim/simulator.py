"""
DICOM/PACS Node Simulator

Simulates a DICOM Picture Archiving and Communication System that:
- Handles medical imaging queries (C-FIND)
- Simulates image retrieval (C-GET/C-MOVE)
- Publishes DICOM operation telemetry via MQTT
- Tracks storage and network usage
"""
import json
import time
import random
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
import paho.mqtt.client as mqtt

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModalityType(Enum):
    """DICOM modality types"""
    CT = "Computed Tomography"
    MRI = "Magnetic Resonance Imaging"
    XR = "X-Ray"
    US = "Ultrasound"
    CR = "Computed Radiography"


class DICOMOperation(Enum):
    """DICOM operations"""
    C_FIND = "Query"
    C_GET = "Retrieve"
    C_STORE = "Store"
    C_MOVE = "Move"


@dataclass
class DICOMNodeState:
    """State of DICOM node"""
    device_id: str
    ward: str = "Radiology"
    is_online: bool = True
    storage_used_gb: float = 1500.0
    storage_total_gb: float = 10000.0
    active_connections: int = 0
    total_studies: int = 50000
    operations_today: int = 0


class DICOMSimulator:
    """Simulates a DICOM PACS node"""
    
    def __init__(
        self,
        device_id: str,
        mqtt_broker: str = "localhost",
        mqtt_port: int = 1883,
        ward: str = "Radiology"
    ):
        self.state = DICOMNodeState(device_id=device_id, ward=ward)
        
        self.mqtt_broker = mqtt_broker
        self.mqtt_port = mqtt_port
        
        # MQTT client
        self.client = mqtt.Client(
            client_id=device_id,
            protocol=mqtt.MQTTv5,
            callback_api_version=mqtt.CallbackAPIVersion.VERSION2
        )
        self.client.on_connect = self._on_connect
        
        # Telemetry interval
        self.telemetry_interval = 5  # seconds
        
        # Operation simulation
        self.operation_rate = 2.0  # operations per minute
        
    def _on_connect(self, client, userdata, flags, reason_code, properties):
        """Callback when connected to MQTT broker"""
        logger.info(f"[{self.state.device_id}] Connected to MQTT broker: {reason_code}")
    
    def connect(self) -> None:
        """Connect to MQTT broker"""
        try:
            self.client.connect(self.mqtt_broker, self.mqtt_port, 60)
            self.client.loop_start()
            logger.info(f"[{self.state.device_id}] MQTT loop started")
        except Exception as e:
            logger.error(f"[{self.state.device_id}] Connection failed: {e}")
            raise
    
    def disconnect(self) -> None:
        """Disconnect from MQTT broker"""
        self.client.loop_stop()
        self.client.disconnect()
        logger.info(f"[{self.state.device_id}] Disconnected")
    
    def simulate_operation(self) -> Dict[str, Any]:
        """Simulate a DICOM operation"""
        operation = random.choice(list(DICOMOperation))
        modality = random.choice(list(ModalityType))
        
        # Simulate operation duration and size
        if operation == DICOMOperation.C_FIND:
            duration_ms = random.randint(10, 100)
            data_size_mb = 0.0
        elif operation == DICOMOperation.C_GET:
            duration_ms = random.randint(500, 3000)
            data_size_mb = random.uniform(100, 500)  # Image retrieval
        elif operation == DICOMOperation.C_STORE:
            duration_ms = random.randint(300, 2000)
            data_size_mb = random.uniform(50, 300)
            # Update storage
            self.state.storage_used_gb += data_size_mb / 1024.0
        else:  # C_MOVE
            duration_ms = random.randint(200, 1500)
            data_size_mb = random.uniform(10, 100)
        
        self.state.operations_today += 1
        
        operation_data = {
            "operation": operation.value,
            "modality": modality.value,
            "duration_ms": duration_ms,
            "data_size_mb": round(data_size_mb, 2),
            "timestamp": datetime.now().isoformat(),
            "study_id": f"STUDY_{random.randint(10000, 99999)}",
            "patient_id": f"PATIENT_{random.randint(100, 999)}"
        }
        
        logger.debug(f"[{self.state.device_id}] Operation: {operation.value} - "
                    f"{duration_ms}ms, {data_size_mb:.1f}MB")
        
        return operation_data
    
    def update_state(self, delta_t: float) -> None:
        """Update DICOM node state"""
        if not self.state.is_online:
            return
        
        # Simulate active connections
        self.state.active_connections = random.randint(0, 5)
        
        # Check storage capacity
        storage_percent = (self.state.storage_used_gb / self.state.storage_total_gb) * 100
        if storage_percent > 90:
            self._publish_event("storage_warning", {
                "storage_used_gb": self.state.storage_used_gb,
                "storage_percent": round(storage_percent, 1)
            })
    
    def generate_telemetry(self) -> Dict[str, Any]:
        """Generate current telemetry data"""
        storage_percent = (self.state.storage_used_gb / self.state.storage_total_gb) * 100
        
        return {
            "device_id": self.state.device_id,
            "timestamp": datetime.now().isoformat(),
            "device_type": "dicom_pacs",
            "ward": self.state.ward,
            "status": {
                "is_online": self.state.is_online,
                "active_connections": self.state.active_connections,
                "operations_today": self.state.operations_today
            },
            "storage": {
                "used_gb": round(self.state.storage_used_gb, 2),
                "total_gb": self.state.storage_total_gb,
                "percent_used": round(storage_percent, 2),
                "total_studies": self.state.total_studies
            }
        }
    
    def _publish_telemetry(self) -> None:
        """Publish telemetry to MQTT"""
        telemetry = self.generate_telemetry()
        topic = f"iomt/telemetry/{self.state.ward}/{self.state.device_id}"
        
        payload = json.dumps(telemetry)
        self.client.publish(topic, payload, qos=1)
        
        logger.debug(f"[{self.state.device_id}] Published telemetry")
    
    def _publish_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Publish an event to MQTT"""
        event = {
            "device_id": self.state.device_id,
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "data": data
        }
        
        topic = f"iomt/events/{self.state.ward}/{self.state.device_id}"
        payload = json.dumps(event)
        self.client.publish(topic, payload, qos=2)
        
        logger.info(f"[{self.state.device_id}] Event: {event_type}")
    
    def run(self, duration: Optional[float] = None) -> None:
        """Run the simulator"""
        self.connect()
        
        start_time = time.time()
        last_telemetry_time = start_time
        last_operation_time = start_time
        
        operation_interval = 60.0 / self.operation_rate  # seconds between operations
        
        try:
            while True:
                current_time = time.time()
                
                # Check duration limit
                if duration and (current_time - start_time) >= duration:
                    break
                
                # Simulate DICOM operation
                if (current_time - last_operation_time) >= operation_interval:
                    operation_data = self.simulate_operation()
                    # Publish operation event
                    self._publish_event("dicom_operation", operation_data)
                    last_operation_time = current_time
                
                # Update state
                delta_t = current_time - last_telemetry_time
                self.update_state(delta_t)
                
                # Publish telemetry
                if delta_t >= self.telemetry_interval:
                    self._publish_telemetry()
                    last_telemetry_time = current_time
                
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            logger.info(f"[{self.state.device_id}] Interrupted by user")
        finally:
            self.disconnect()


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="DICOM PACS Simulator")
    parser.add_argument("--device-id", default="dicom_001", help="Device ID")
    parser.add_argument("--broker", default="localhost", help="MQTT broker host")
    parser.add_argument("--port", type=int, default=1883, help="MQTT broker port")
    parser.add_argument("--ward", default="Radiology", help="Ward/department")
    parser.add_argument("--duration", type=float, help="Run duration in seconds")
    
    args = parser.parse_args()
    
    simulator = DICOMSimulator(
        device_id=args.device_id,
        mqtt_broker=args.broker,
        mqtt_port=args.port,
        ward=args.ward
    )
    
    logger.info(f"Starting DICOM simulator: {args.device_id}")
    simulator.run(duration=args.duration)


if __name__ == "__main__":
    main()
