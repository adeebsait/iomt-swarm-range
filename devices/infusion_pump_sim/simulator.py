"""
Infusion Pump IoMT Device Simulator

Simulates a smart infusion pump that:
- Publishes telemetry via MQTT
- Generates HL7/FHIR medication administration records
- Emulates realistic device behavior (infusion rates, alarms, battery)
"""
import json
import time
import random
import logging
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import paho.mqtt.client as mqtt

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class PumpState:
    """Internal state of the infusion pump"""
    device_id: str
    is_running: bool = False
    infusion_rate_ml_hr: float = 0.0
    volume_infused_ml: float = 0.0
    volume_to_infuse_ml: float = 0.0
    battery_percent: float = 100.0
    pressure_psi: float = 10.0
    occlusion_detected: bool = False
    air_in_line_detected: bool = False
    alarm_active: bool = False
    alarm_type: Optional[str] = None
    medication_name: str = "Normal Saline"
    concentration_mg_ml: float = 0.9
    patient_id: str = "PATIENT_000"
    start_time: Optional[datetime] = None


class InfusionPumpSimulator:
    """Simulates a network-connected infusion pump"""
    
    def __init__(
        self,
        device_id: str,
        mqtt_broker: str = "localhost",
        mqtt_port: int = 1883,
        ward: str = "ICU"
    ):
        self.state = PumpState(device_id=device_id)
        self.mqtt_broker = mqtt_broker
        self.mqtt_port = mqtt_port
        self.ward = ward
        
        # MQTT client
        self.client = mqtt.Client(client_id=device_id, protocol=mqtt.MQTTv5, callback_api_version=mqtt.CallbackAPIVersion.VERSION2)
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        
        # Telemetry interval
        self.telemetry_interval = 5  # seconds
        
        # Simulate realistic variations
        self.pressure_noise_amplitude = 1.0
        self.battery_drain_rate = 0.002  # percent per second when running
        
    def _on_connect(self, client, userdata, flags, reason_code, properties):
        """Callback when connected to MQTT broker"""
        logger.info(f"[{self.state.device_id}] Connected to MQTT broker: {reason_code}")
        
        # Subscribe to command topic
        command_topic = f"iomt/commands/{self.ward}/{self.state.device_id}"
        self.client.subscribe(command_topic)
        logger.info(f"[{self.state.device_id}] Subscribed to {command_topic}")
    
    def _on_disconnect(self, client, userdata, flags, reason_code, properties):
        """Callback when disconnected from MQTT broker"""
        logger.warning(f"[{self.state.device_id}] Disconnected: {reason_code}")
    
    def connect(self) -> None:
        """Connect to MQTT broker"""
        try:
            self.client.connect(self.mqtt_broker, self.mqtt_port, 60)
            self.client.loop_start()
            logger.info(f"[{self.state.device_id}] Starting MQTT loop")
        except Exception as e:
            logger.error(f"[{self.state.device_id}] Connection failed: {e}")
            raise
    
    def disconnect(self) -> None:
        """Disconnect from MQTT broker"""
        self.client.loop_stop()
        self.client.disconnect()
        logger.info(f"[{self.state.device_id}] Disconnected")
    
    def start_infusion(
        self,
        rate_ml_hr: float,
        volume_ml: float,
        medication: str = "Normal Saline",
        concentration: float = 0.9,
        patient_id: str = "PATIENT_000"
    ) -> None:
        """Start an infusion"""
        self.state.is_running = True
        self.state.infusion_rate_ml_hr = rate_ml_hr
        self.state.volume_to_infuse_ml = volume_ml
        self.state.volume_infused_ml = 0.0
        self.state.medication_name = medication
        self.state.concentration_mg_ml = concentration
        self.state.patient_id = patient_id
        self.state.start_time = datetime.now()
        self.state.alarm_active = False
        
        logger.info(f"[{self.state.device_id}] Started infusion: "
                   f"{rate_ml_hr} mL/hr, {volume_ml} mL total")
        
        # Publish infusion started event
        self._publish_event("infusion_started", {
            "rate_ml_hr": rate_ml_hr,
            "volume_ml": volume_ml,
            "medication": medication,
            "patient_id": patient_id
        })
    
    def stop_infusion(self) -> None:
        """Stop the infusion"""
        self.state.is_running = False
        logger.info(f"[{self.state.device_id}] Stopped infusion")
        
        self._publish_event("infusion_stopped", {
            "volume_infused_ml": self.state.volume_infused_ml
        })
    
    def update_state(self, delta_t: float) -> None:
        """
        Update pump state based on time elapsed.
        
        Args:
            delta_t: Time elapsed in seconds
        """
        if not self.state.is_running:
            return
        
        # Update volume infused
        ml_per_second = self.state.infusion_rate_ml_hr / 3600.0
        volume_increment = ml_per_second * delta_t
        self.state.volume_infused_ml += volume_increment
        
        # Check if target volume reached
        if self.state.volume_infused_ml >= self.state.volume_to_infuse_ml:
            self.state.volume_infused_ml = self.state.volume_to_infuse_ml
            self.stop_infusion()
            self._trigger_alarm("infusion_complete")
            return
        
        # Update battery
        self.state.battery_percent -= self.battery_drain_rate * delta_t
        self.state.battery_percent = max(0.0, self.state.battery_percent)
        
        # Check battery level
        if self.state.battery_percent < 10.0 and not self.state.alarm_active:
            self._trigger_alarm("low_battery")
        
        # Simulate pressure variations
        self.state.pressure_psi = 10.0 + random.uniform(
            -self.pressure_noise_amplitude,
            self.pressure_noise_amplitude
        )
        
        # Random occlusion detection (very rare)
        if random.random() < 0.0001:  # 0.01% chance per update
            self.state.occlusion_detected = True
            self._trigger_alarm("occlusion_detected")
        
        # Random air-in-line detection (very rare)
        if random.random() < 0.00005:  # 0.005% chance per update
            self.state.air_in_line_detected = True
            self._trigger_alarm("air_in_line")
    
    def _trigger_alarm(self, alarm_type: str) -> None:
        """Trigger an alarm"""
        self.state.alarm_active = True
        self.state.alarm_type = alarm_type
        
        logger.warning(f"[{self.state.device_id}] ALARM: {alarm_type}")
        
        self._publish_event("alarm", {
            "alarm_type": alarm_type,
            "timestamp": datetime.now().isoformat()
        })
    
    def generate_telemetry(self) -> Dict[str, Any]:
        """Generate current telemetry data"""
        return {
            "device_id": self.state.device_id,
            "timestamp": datetime.now().isoformat(),
            "device_type": "infusion_pump",
            "ward": self.ward,
            "status": "running" if self.state.is_running else "idle",
            "infusion_rate_ml_hr": round(self.state.infusion_rate_ml_hr, 2),
            "volume_infused_ml": round(self.state.volume_infused_ml, 2),
            "volume_remaining_ml": round(
                self.state.volume_to_infuse_ml - self.state.volume_infused_ml, 2
            ),
            "battery_percent": round(self.state.battery_percent, 1),
            "pressure_psi": round(self.state.pressure_psi, 2),
            "occlusion_detected": self.state.occlusion_detected,
            "air_in_line_detected": self.state.air_in_line_detected,
            "alarm_active": self.state.alarm_active,
            "alarm_type": self.state.alarm_type,
            "medication": self.state.medication_name,
            "patient_id": self.state.patient_id
        }
    
    def _publish_telemetry(self) -> None:
        """Publish telemetry to MQTT"""
        telemetry = self.generate_telemetry()
        topic = f"iomt/telemetry/{self.ward}/{self.state.device_id}"
        
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
        
        topic = f"iomt/events/{self.ward}/{self.state.device_id}"
        payload = json.dumps(event)
        self.client.publish(topic, payload, qos=2)  # QoS 2 for events
        
        logger.info(f"[{self.state.device_id}] Event: {event_type}")
    
    def run(self, duration: Optional[float] = None) -> None:
        """
        Run the simulator.
        
        Args:
            duration: Run duration in seconds (None for infinite)
        """
        self.connect()
        
        # Start a default infusion for simulation
        self.start_infusion(
            rate_ml_hr=random.uniform(50, 200),
            volume_ml=random.uniform(100, 500),
            medication=random.choice([
                "Normal Saline",
                "Dextrose 5%",
                "Lactated Ringer's",
                "Insulin",
                "Morphine"
            ]),
            patient_id=f"PATIENT_{random.randint(100, 999)}"
        )
        
        start_time = time.time()
        last_telemetry_time = start_time
        
        try:
            while True:
                current_time = time.time()
                
                # Check duration limit
                if duration and (current_time - start_time) >= duration:
                    break
                
                # Update state
                delta_t = current_time - last_telemetry_time
                self.update_state(delta_t)
                
                # Publish telemetry at regular intervals
                if delta_t >= self.telemetry_interval:
                    self._publish_telemetry()
                    last_telemetry_time = current_time
                
                # Sleep briefly to avoid busy waiting
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            logger.info(f"[{self.state.device_id}] Interrupted by user")
        finally:
            self.disconnect()


def main():
    """Main entry point for standalone execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Infusion Pump Simulator")
    parser.add_argument("--device-id", default="pump_001", help="Device ID")
    parser.add_argument("--broker", default="localhost", help="MQTT broker host")
    parser.add_argument("--port", type=int, default=1883, help="MQTT broker port")
    parser.add_argument("--ward", default="ICU", help="Ward/department")
    parser.add_argument("--duration", type=float, help="Run duration in seconds")
    
    args = parser.parse_args()
    
    simulator = InfusionPumpSimulator(
        device_id=args.device_id,
        mqtt_broker=args.broker,
        mqtt_port=args.port,
        ward=args.ward
    )
    
    logger.info(f"Starting infusion pump simulator: {args.device_id}")
    simulator.run(duration=args.duration)


if __name__ == "__main__":
    main()
