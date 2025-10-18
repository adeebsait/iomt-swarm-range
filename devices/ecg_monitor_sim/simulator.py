"""
ECG/Cardiac Monitor IoMT Device Simulator

Simulates a bedside cardiac monitor that:
- Generates synthetic ECG waveforms
- Monitors vital signs (HR, SpO2, BP, RR)
- Publishes continuous telemetry via MQTT
- Triggers alarms for abnormal values
"""
import json
import time
import random
import math
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from dataclasses import dataclass
import paho.mqtt.client as mqtt
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class VitalSigns:
    """Patient vital signs"""
    heart_rate_bpm: int = 75
    spo2_percent: float = 98.0
    respiratory_rate: int = 16
    systolic_bp_mmhg: int = 120
    diastolic_bp_mmhg: int = 80
    temperature_c: float = 37.0


@dataclass
class MonitorState:
    """Internal state of cardiac monitor"""
    device_id: str
    patient_id: str = "PATIENT_000"
    ward: str = "ICU"
    bed_number: str = "BED_001"
    is_monitoring: bool = True
    alarm_active: bool = False
    alarm_type: Optional[str] = None
    battery_percent: float = 100.0
    leads_connected: int = 5  # 5-lead ECG


class ECGMonitorSimulator:
    """Simulates a network-connected cardiac monitor"""
    
    def __init__(
        self,
        device_id: str,
        mqtt_broker: str = "localhost",
        mqtt_port: int = 1883,
        ward: str = "ICU"
    ):
        self.state = MonitorState(device_id=device_id, ward=ward)
        self.vitals = VitalSigns()
        
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
        self.telemetry_interval = 2  # seconds
        
        # Waveform parameters
        self.sample_rate = 250  # Hz
        self.waveform_duration = 1.0  # seconds
        
        # Physiological variation parameters
        self.hr_drift = 0.0
        self.spo2_drift = 0.0
        
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
    
    def generate_ecg_waveform(self, duration_sec: float = 1.0) -> List[float]:
        """
        Generate synthetic ECG waveform (Lead II).
        
        Simplified P-QRS-T complex simulation using sine waves.
        
        Args:
            duration_sec: Waveform duration in seconds
            
        Returns:
            List of voltage samples in mV
        """
        samples = int(duration_sec * self.sample_rate)
        waveform = []
        
        # Heart rate in beats per second
        hr_hz = self.vitals.heart_rate_bpm / 60.0
        
        for i in range(samples):
            t = i / self.sample_rate
            
            # Determine position in cardiac cycle
            cycle_position = (t * hr_hz) % 1.0
            
            voltage = 0.0
            
            # P wave (atrial depolarization) - 0.0 to 0.2
            if 0.0 <= cycle_position < 0.2:
                phase = cycle_position * 5  # Scale to 0-1
                voltage = 0.15 * math.sin(math.pi * phase)
            
            # QRS complex (ventricular depolarization) - 0.2 to 0.35
            elif 0.2 <= cycle_position < 0.35:
                phase = (cycle_position - 0.2) / 0.15
                # Sharp spike
                voltage = 1.2 * math.sin(math.pi * phase)
            
            # T wave (ventricular repolarization) - 0.4 to 0.6
            elif 0.4 <= cycle_position < 0.6:
                phase = (cycle_position - 0.4) / 0.2
                voltage = 0.3 * math.sin(math.pi * phase)
            
            # Add baseline noise
            noise = random.uniform(-0.02, 0.02)
            voltage += noise
            
            waveform.append(round(voltage, 4))
        
        return waveform
    
    def update_vitals(self, delta_t: float) -> None:
        """
        Update vital signs with realistic physiological variations.
        
        Args:
            delta_t: Time elapsed in seconds
        """
        if not self.state.is_monitoring:
            return
        
        # Heart rate variation (respiratory sinus arrhythmia)
        self.hr_drift += random.uniform(-0.5, 0.5)
        self.hr_drift = np.clip(self.hr_drift, -5, 5)
        self.vitals.heart_rate_bpm = int(
            np.clip(75 + self.hr_drift + random.uniform(-1, 1), 50, 150)
        )
        
        # SpO2 variation
        self.spo2_drift += random.uniform(-0.1, 0.1)
        self.spo2_drift = np.clip(self.spo2_drift, -2, 2)
        self.vitals.spo2_percent = round(
            np.clip(98.0 + self.spo2_drift + random.uniform(-0.2, 0.2), 85, 100),
            1
        )
        
        # Respiratory rate (slower variation)
        self.vitals.respiratory_rate = int(
            np.clip(16 + random.uniform(-1, 1), 10, 30)
        )
        
        # Blood pressure (correlated with HR)
        hr_factor = (self.vitals.heart_rate_bpm - 75) / 75.0
        self.vitals.systolic_bp_mmhg = int(
            np.clip(120 + hr_factor * 10 + random.uniform(-5, 5), 80, 180)
        )
        self.vitals.diastolic_bp_mmhg = int(
            np.clip(80 + hr_factor * 5 + random.uniform(-3, 3), 50, 120)
        )
        
        # Temperature (very stable)
        self.vitals.temperature_c = round(
            np.clip(37.0 + random.uniform(-0.1, 0.1), 35.5, 39.0),
            1
        )
        
        # Check for alarm conditions
        self._check_alarms()
    
    def _check_alarms(self) -> None:
        """Check vital signs for alarm conditions"""
        alarm_triggered = False
        alarm_type = None
        
        # Tachycardia
        if self.vitals.heart_rate_bpm > 120:
            alarm_triggered = True
            alarm_type = "tachycardia"
        
        # Bradycardia
        elif self.vitals.heart_rate_bpm < 50:
            alarm_triggered = True
            alarm_type = "bradycardia"
        
        # Hypoxia
        elif self.vitals.spo2_percent < 90:
            alarm_triggered = True
            alarm_type = "hypoxia"
        
        # Hypertension
        elif self.vitals.systolic_bp_mmhg > 160:
            alarm_triggered = True
            alarm_type = "hypertension"
        
        # Hypotension
        elif self.vitals.systolic_bp_mmhg < 90:
            alarm_triggered = True
            alarm_type = "hypotension"
        
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
            "heart_rate": self.vitals.heart_rate_bpm,
            "spo2": self.vitals.spo2_percent,
            "systolic_bp": self.vitals.systolic_bp_mmhg
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
            "device_type": "ecg_monitor",
            "ward": self.state.ward,
            "bed_number": self.state.bed_number,
            "patient_id": self.state.patient_id,
            "vital_signs": {
                "heart_rate_bpm": self.vitals.heart_rate_bpm,
                "spo2_percent": self.vitals.spo2_percent,
                "respiratory_rate": self.vitals.respiratory_rate,
                "systolic_bp_mmhg": self.vitals.systolic_bp_mmhg,
                "diastolic_bp_mmhg": self.vitals.diastolic_bp_mmhg,
                "temperature_c": self.vitals.temperature_c
            },
            "ecg_lead_ii": self.generate_ecg_waveform(duration_sec=0.5),
            "alarm_active": self.state.alarm_active,
            "alarm_type": self.state.alarm_type,
            "leads_connected": self.state.leads_connected,
            "battery_percent": round(self.state.battery_percent, 1)
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
                
                # Update vitals
                delta_t = current_time - last_telemetry_time
                self.update_vitals(delta_t)
                
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
    
    parser = argparse.ArgumentParser(description="ECG Monitor Simulator")
    parser.add_argument("--device-id", default="ecg_001", help="Device ID")
    parser.add_argument("--broker", default="localhost", help="MQTT broker host")
    parser.add_argument("--port", type=int, default=1883, help="MQTT broker port")
    parser.add_argument("--ward", default="ICU", help="Ward/department")
    parser.add_argument("--duration", type=float, help="Run duration in seconds")
    
    args = parser.parse_args()
    
    simulator = ECGMonitorSimulator(
        device_id=args.device_id,
        mqtt_broker=args.broker,
        mqtt_port=args.port,
        ward=args.ward
    )
    
    logger.info(f"Starting ECG monitor simulator: {args.device_id}")
    simulator.run(duration=args.duration)


if __name__ == "__main__":
    main()
