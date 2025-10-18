"""
FHIR Gateway Simulator

Simulates an HL7 FHIR API gateway that:
- Bridges IoMT devices to EHR systems
- Translates device data to FHIR Observations
- Handles FHIR REST API operations (GET, POST, PUT)
- Publishes integration telemetry via MQTT
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


class FHIRResourceType(Enum):
    """FHIR resource types"""
    OBSERVATION = "Observation"
    PATIENT = "Patient"
    DEVICE = "Device"
    MEDICATION_ADMINISTRATION = "MedicationAdministration"


class HTTPMethod(Enum):
    """HTTP methods"""
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    DELETE = "DELETE"


@dataclass
class FHIRGatewayState:
    """State of FHIR gateway"""
    device_id: str
    ward: str = "Hospital"
    is_online: bool = True
    connected_devices: int = 0
    requests_today: int = 0
    observations_created: int = 0
    errors_today: int = 0


class FHIRGatewaySimulator:
    """Simulates a FHIR integration gateway"""
    
    def __init__(
        self,
        device_id: str,
        mqtt_broker: str = "localhost",
        mqtt_port: int = 1883,
        ward: str = "Hospital"
    ):
        self.state = FHIRGatewayState(device_id=device_id, ward=ward)
        
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
        
        # Request simulation rate
        self.request_rate = 5.0  # requests per minute
        
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
    
    def create_fhir_observation(
        self,
        patient_id: str,
        device_id: str,
        code: str,
        value: float,
        unit: str
    ) -> Dict[str, Any]:
        """Create a FHIR Observation resource"""
        observation = {
            "resourceType": "Observation",
            "id": f"obs-{random.randint(10000, 99999)}",
            "status": "final",
            "code": {
                "coding": [{
                    "system": "http://loinc.org",
                    "code": code,
                    "display": self._get_code_display(code)
                }]
            },
            "subject": {
                "reference": f"Patient/{patient_id}"
            },
            "device": {
                "reference": f"Device/{device_id}"
            },
            "effectiveDateTime": datetime.now().isoformat(),
            "valueQuantity": {
                "value": value,
                "unit": unit,
                "system": "http://unitsofmeasure.org"
            }
        }
        
        self.state.observations_created += 1
        return observation
    
    def _get_code_display(self, code: str) -> str:
        """Get display name for LOINC code"""
        code_map = {
            "8867-4": "Heart rate",
            "2708-6": "Oxygen saturation",
            "8480-6": "Systolic blood pressure",
            "8462-4": "Diastolic blood pressure",
            "9279-1": "Respiratory rate",
            "8310-5": "Body temperature"
        }
        return code_map.get(code, "Unknown")
    
    def simulate_fhir_request(self) -> Dict[str, Any]:
        """Simulate a FHIR API request"""
        method = random.choice(list(HTTPMethod))
        resource_type = random.choice(list(FHIRResourceType))
        
        # Simulate response time based on operation
        if method == HTTPMethod.GET:
            response_time_ms = random.randint(10, 100)
        elif method == HTTPMethod.POST:
            response_time_ms = random.randint(50, 300)
        else:  # PUT, DELETE
            response_time_ms = random.randint(30, 200)
        
        # Simulate success/failure (95% success rate)
        status_code = 200 if random.random() < 0.95 else random.choice([400, 404, 500])
        
        if status_code != 200:
            self.state.errors_today += 1
        
        self.state.requests_today += 1
        
        request_data = {
            "method": method.value,
            "resource_type": resource_type.value,
            "response_time_ms": response_time_ms,
            "status_code": status_code,
            "timestamp": datetime.now().isoformat(),
            "request_id": f"req-{random.randint(100000, 999999)}"
        }
        
        logger.debug(f"[{self.state.device_id}] FHIR {method.value} {resource_type.value} - "
                    f"{status_code} ({response_time_ms}ms)")
        
        return request_data
    
    def update_state(self, delta_t: float) -> None:
        """Update gateway state"""
        if not self.state.is_online:
            return
        
        # Simulate connected devices (fluctuates)
        self.state.connected_devices = random.randint(10, 50)
        
        # Check error rate
        if self.state.requests_today > 0:
            error_rate = self.state.errors_today / self.state.requests_today
            if error_rate > 0.1:  # More than 10% errors
                self._publish_event("high_error_rate", {
                    "error_rate": round(error_rate * 100, 2),
                    "errors_today": self.state.errors_today,
                    "requests_today": self.state.requests_today
                })
    
    def generate_telemetry(self) -> Dict[str, Any]:
        """Generate current telemetry data"""
        error_rate = 0.0
        if self.state.requests_today > 0:
            error_rate = (self.state.errors_today / self.state.requests_today) * 100
        
        return {
            "device_id": self.state.device_id,
            "timestamp": datetime.now().isoformat(),
            "device_type": "fhir_gateway",
            "ward": self.state.ward,
            "status": {
                "is_online": self.state.is_online,
                "connected_devices": self.state.connected_devices
            },
            "metrics": {
                "requests_today": self.state.requests_today,
                "observations_created": self.state.observations_created,
                "errors_today": self.state.errors_today,
                "error_rate_percent": round(error_rate, 2)
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
        
        logger.warning(f"[{self.state.device_id}] Event: {event_type}")
    
    def run(self, duration: Optional[float] = None) -> None:
        """Run the simulator"""
        self.connect()
        
        start_time = time.time()
        last_telemetry_time = start_time
        last_request_time = start_time
        
        request_interval = 60.0 / self.request_rate  # seconds between requests
        
        try:
            while True:
                current_time = time.time()
                
                # Check duration limit
                if duration and (current_time - start_time) >= duration:
                    break
                
                # Simulate FHIR request
                if (current_time - last_request_time) >= request_interval:
                    request_data = self.simulate_fhir_request()
                    # Publish request event
                    self._publish_event("fhir_request", request_data)
                    last_request_time = current_time
                
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
    
    parser = argparse.ArgumentParser(description="FHIR Gateway Simulator")
    parser.add_argument("--device-id", default="fhir_001", help="Device ID")
    parser.add_argument("--broker", default="localhost", help="MQTT broker host")
    parser.add_argument("--port", type=int, default=1883, help="MQTT broker port")
    parser.add_argument("--ward", default="Hospital", help="Ward/department")
    parser.add_argument("--duration", type=float, help="Run duration in seconds")
    
    args = parser.parse_args()
    
    simulator = FHIRGatewaySimulator(
        device_id=args.device_id,
        mqtt_broker=args.broker,
        mqtt_port=args.port,
        ward=args.ward
    )
    
    logger.info(f"Starting FHIR gateway simulator: {args.device_id}")
    simulator.run(duration=args.duration)


if __name__ == "__main__":
    main()
