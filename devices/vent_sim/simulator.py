"""
Ventilator IoMT Device Simulator

Simulates a mechanical ventilator that:
- Controls respiratory parameters (tidal volume, pressure, FiO2, PEEP)
- Publishes respiratory telemetry via MQTT
- Monitors ventilation modes (AC, SIMV, CPAP, etc.)
- Triggers alarms for unsafe conditions
"""
import json
import time
import random
import logging
from typing import Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
import paho.mqtt.client as mqtt
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class VentilationMode(Enum):
    """Ventilation modes"""
    AC = "Assist Control"
    SIMV = "Synchronized Intermittent Mandatory Ventilation"
    CPAP = "Continuous Positive Airway Pressure"
    PSV = "Pressure Support Ventilation"


@dataclass
class VentilatorSettings:
    """Ventilator configuration settings"""
    mode: VentilationMode = VentilationMode.AC
    tidal_volume_ml: int = 500
    respiratory_rate: int = 12
    fio2_percent: int = 40  # Fraction of inspired oxygen
    peep_cmh2o: int = 5  # Positive end-expiratory pressure
    inspiratory_pressure_cmh2o: int = 20
    ie_ratio: str = "1:2"  # Inspiration:Expiration ratio


@dataclass
class RespiratoryData:
    """Current respiratory measurements"""
    minute_volume_l: float = 6.0
    peak_pressure_cmh2o: int = 20
    plateau_pressure_cmh2o: int = 15
    compliance_ml_cmh2o: float = 50.0
    resistance_cmh2o_l_s: float = 10.0
    spo2_percent: float = 98.0
    etco2_mmhg: int = 35  # End-tidal CO2


@dataclass
class VentilatorState:
    """Internal state of ventilator"""
    device_id: str
    patient_id: str = "PATIENT_000"
    ward: str = "ICU"
    is_ventilating: bool = True
    alarm_active: bool = False
    alarm_type: Optional[str] = None
    battery_percent: float = 100.0
    ventilator_hours: float = 0.0


class VentilatorSimulator:
    """Simulates a network-connected mechanical ventilator"""
    
    def __init__(
        self,
        device_id: str,
        mqtt_broker: str = "localhost",
        mqtt_port: int = 1883,
        ward: str = "ICU"
    ):
        self.state = VentilatorState(device_id=device_id, ward=ward)
        self.settings = VentilatorSettings()
        self.respiratory = RespiratoryData()
        
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
        self.telemetry_interval = 3  # seconds
        
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
    
    def set_ventilation_mode(self, mode: VentilationMode) -> None:
        """Change ventilation mode"""
        old_mode = self.settings.mode
        self.settings.mode = mode
        
        logger.info(f"[{self.state.device_id}] Mode changed: {old_mode.value} -> {mode.value}")
        
        self._publish_event("mode_changed", {
            "old_mode": old_mode.value,
            "new_mode": mode.value
        })
    
    def update_respiratory_data(self, delta_t: float) -> None:
        """
        Update respiratory measurements based on current settings.
        
        Args:
            delta_t: Time elapsed in seconds
        """
        if not self.state.is_ventilating:
            return
        
        # Update ventilator hours
        self.state.ventilator_hours += delta_t / 3600.0
        
        # Calculate minute volume (tidal volume * respiratory rate)
        self.respiratory.minute_volume_l = (
            self.settings.tidal_volume_ml * self.settings.respiratory_rate / 1000.0
        )
        
        # Add physiological variation
        self.respiratory.minute_volume_l += random.uniform(-0.5, 0.5)
        self.respiratory.minute_volume_l = max(0.0, self.respiratory.minute_volume_l)
        
        # Peak pressure (depends on tidal volume and resistance)
        base_peak = self.settings.inspiratory_pressure_cmh2o
        self.respiratory.peak_pressure_cmh2o = int(
            base_peak + random.uniform(-2, 2)
        )
        
        # Plateau pressure (slightly lower than peak)
        self.respiratory.plateau_pressure_cmh2o = int(
            self.respiratory.peak_pressure_cmh2o * 0.8 + random.uniform(-1, 1)
        )
        
        # Lung compliance (tidal volume / (plateau - PEEP))
        pressure_diff = max(1, self.respiratory.plateau_pressure_cmh2o - self.settings.peep_cmh2o)
        self.respiratory.compliance_ml_cmh2o = round(
            self.settings.tidal_volume_ml / pressure_diff + random.uniform(-5, 5),
            1
        )
        
        # Airway resistance
        self.respiratory.resistance_cmh2o_l_s = round(
            10.0 + random.uniform(-2, 2),
            1
        )
        
        # SpO2 (depends on FiO2)
        target_spo2 = 95 + (self.settings.fio2_percent - 40) * 0.1
        self.respiratory.spo2_percent = round(
            np.clip(target_spo2 + random.uniform(-1, 1), 85, 100),
            1
        )
        
        # EtCO2 (end-tidal CO2) - inversely related to minute ventilation
        target_etco2 = 40 - (self.respiratory.minute_volume_l - 6.0) * 2
        self.respiratory.etco2_mmhg = int(
            np.clip(target_etco2 + random.uniform(-2, 2), 20, 60)
        )
        
        # Check for alarm conditions
        self._check_alarms()
    
    def _check_alarms(self) -> None:
        """Check for alarm conditions"""
        alarm_triggered = False
        alarm_type = None
        
        # High pressure alarm
        if self.respiratory.peak_pressure_cmh2o > 35:
            alarm_triggered = True
            alarm_type = "high_pressure"
        
        # Low minute volume
        elif self.respiratory.minute_volume_l < 4.0:
            alarm_triggered = True
            alarm_type = "low_minute_volume"
        
        # Hypoxia
        elif self.respiratory.spo2_percent < 90:
            alarm_triggered = True
            alarm_type = "hypoxia"
        
        # Hypercapnia (high CO2)
        elif self.respiratory.etco2_mmhg > 50:
            alarm_triggered = True
            alarm_type = "hypercapnia"
        
        # Low compliance (stiff lungs)
        elif self.respiratory.compliance_ml_cmh2o < 30:
            alarm_triggered = True
            alarm_type = "low_compliance"
        
        if alarm_triggered and not self.state.alarm_active:
            self._trigger_alarm(alarm_type)
        elif not alarm_triggered and self.state.alarm_active:
            self._clear_alarm()
    
    def _trigger_alarm(self, alarm_type: str) -> None:
        """Trigger an alarm"""
        self.state.alarm_active = True
        self.state.alarm_type = alarm_type
        
        logger.warning(f"[{self.state.device_id}] ALARM: {alarm_type}")
        
        self._publish_event("alarm", {
            "alarm_type": alarm_type,
            "peak_pressure": self.respiratory.peak_pressure_cmh2o,
            "minute_volume": self.respiratory.minute_volume_l,
            "spo2": self.respiratory.spo2_percent
        })
    
    def _clear_alarm(self) -> None:
        """Clear active alarm"""
        logger.info(f"[{self.state.device_id}] Alarm cleared: {self.state.alarm_type}")
        self.state.alarm_active = False
        self.state.alarm_type = None
    
    def generate_telemetry(self) -> Dict[str, Any]:
        """Generate current telemetry data"""
        return {
            "device_id": self.state.device_id,
            "timestamp": datetime.now().isoformat(),
            "device_type": "ventilator",
            "ward": self.state.ward,
            "patient_id": self.state.patient_id,
            "settings": {
                "mode": self.settings.mode.value,
                "tidal_volume_ml": self.settings.tidal_volume_ml,
                "respiratory_rate": self.settings.respiratory_rate,
                "fio2_percent": self.settings.fio2_percent,
                "peep_cmh2o": self.settings.peep_cmh2o,
                "inspiratory_pressure_cmh2o": self.settings.inspiratory_pressure_cmh2o,
                "ie_ratio": self.settings.ie_ratio
            },
            "measurements": {
                "minute_volume_l": round(self.respiratory.minute_volume_l, 2),
                "peak_pressure_cmh2o": self.respiratory.peak_pressure_cmh2o,
                "plateau_pressure_cmh2o": self.respiratory.plateau_pressure_cmh2o,
                "compliance_ml_cmh2o": self.respiratory.compliance_ml_cmh2o,
                "resistance_cmh2o_l_s": self.respiratory.resistance_cmh2o_l_s,
                "spo2_percent": self.respiratory.spo2_percent,
                "etco2_mmhg": self.respiratory.etco2_mmhg
            },
            "status": {
                "is_ventilating": self.state.is_ventilating,
                "alarm_active": self.state.alarm_active,
                "alarm_type": self.state.alarm_type,
                "ventilator_hours": round(self.state.ventilator_hours, 2),
                "battery_percent": round(self.state.battery_percent, 1)
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
        """
        Run the simulator.
        
        Args:
            duration: Run duration in seconds (None for infinite)
        """
        self.connect()
        
        start_time = time.time()
        last_telemetry_time = start_time
        
        try:
            while True:
                current_time = time.time()
                
                # Check duration limit
                if duration and (current_time - start_time) >= duration:
                    break
                
                # Update respiratory data
                delta_t = current_time - last_telemetry_time
                self.update_respiratory_data(delta_t)
                
                # Publish telemetry at regular intervals
                if delta_t >= self.telemetry_interval:
                    self._publish_telemetry()
                    last_telemetry_time = current_time
                
                # Sleep briefly
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            logger.info(f"[{self.state.device_id}] Interrupted by user")
        finally:
            self.disconnect()


def main():
    """Main entry point for standalone execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Ventilator Simulator")
    parser.add_argument("--device-id", default="vent_001", help="Device ID")
    parser.add_argument("--broker", default="localhost", help="MQTT broker host")
    parser.add_argument("--port", type=int, default=1883, help="MQTT broker port")
    parser.add_argument("--ward", default="ICU", help="Ward/department")
    parser.add_argument("--duration", type=float, help="Run duration in seconds")
    
    args = parser.parse_args()
    
    simulator = VentilatorSimulator(
        device_id=args.device_id,
        mqtt_broker=args.broker,
        mqtt_port=args.port,
        ward=args.ward
    )
    
    logger.info(f"Starting ventilator simulator: {args.device_id}")
    simulator.run(duration=args.duration)


if __name__ == "__main__":
    main()
