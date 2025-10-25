"""
Integrated Real-World Experiment Runner

Uses actual device telemetry, attack injections, and swarm detection
"""
import time
import json
import numpy as np
from datetime import datetime
from pathlib import Path
import paho.mqtt.client as mqtt
from threading import Thread
import logging

from evaluation.metrics.detector_metrics import DetectionMetrics
from attacks.injectors.dos_attack import DoSAttack
from attacks.injectors.data_injection import DataInjectionAttack
from detection.detectors.abc_detector import ABCThreatDetector
from detection.features.traffic_analyzer import TrafficAnalyzer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IntegratedExperiment:
    """Run experiments with real MQTT integration"""
    
    def __init__(self):
        self.mqtt_client = None
        self.traffic_analyzer = TrafficAnalyzer(window_size=50)
        self.baseline_traffic = {}
        self.attack_active = False
        self.message_count = 0
        
    def setup_mqtt(self):
        """Connect to MQTT broker"""
        self.mqtt_client = mqtt.Client(
            client_id=f"experiment_runner",
            protocol=mqtt.MQTTv5,
            callback_api_version=mqtt.CallbackAPIVersion.VERSION2
        )
        self.mqtt_client.on_message = self.on_message
        self.mqtt_client.connect("localhost", 1883, 60)
        self.mqtt_client.subscribe("iomt/#")
        self.mqtt_client.loop_start()
        logger.info("Connected to MQTT broker")
        
    def on_message(self, client, userdata, msg):
        """Record MQTT messages for analysis"""
        self.message_count += 1
        self.traffic_analyzer.add_message(
            timestamp=time.time(),
            size=len(msg.payload),
            topic=msg.topic,
            device_id=msg.topic.split('/')[-1] if '/' in msg.topic else 'unknown'
        )
    
    def run_baseline_collection(self, duration: int = 30):
        """Collect normal traffic baseline"""
        logger.info(f"Collecting baseline traffic for {duration}s...")
        start_time = time.time()
        
        while time.time() - start_time < duration:
            time.sleep(1)
        
        features = self.traffic_analyzer.extract_features()
        if features:
            # Calculate baseline statistics
            for key, val in features.items():
                self.baseline_traffic[key] = {
                    'mean': val,
                    'std': val * 0.1  # 10% variance
                }
        
        logger.info(f"Baseline collected: {self.message_count} messages")
        self.message_count = 0
    
    def run_attack_experiment(self, attack_type: str, duration: int = 60):
        """Run experiment with real attack"""
        logger.info(f"Running {attack_type} attack for {duration}s...")
        
        metrics = DetectionMetrics()
        start_time = time.time()
        
        # Start attack in background
        if attack_type == 'DoS':
            attacker = DoSAttack()
            attack_thread = Thread(
                target=attacker.message_flood,
                args=(duration, 500)  # 500 msg/s
            )
        elif attack_type == 'DataInjection':
            attacker = DataInjectionAttack()
            attack_thread = Thread(
                target=attacker.inject_critical_vitals,
                args=(duration,)
            )
        else:
            return None
        
        attack_thread.start()
        self.attack_active = True
        
        # Monitor and detect
        while time.time() - start_time < duration:
            features = self.traffic_analyzer.extract_features()
            
            if features and self.baseline_traffic:
                # Detect anomaly
                is_anomaly = self.traffic_analyzer.is_anomalous(
                    features, self.baseline_traffic, threshold=2.5
                )
                
                # Record detection
                detection_latency = time.time() - start_time if is_anomaly else 0
                metrics.record_detection(
                    predicted=is_anomaly,
                    actual=self.attack_active,
                    latency=detection_latency
                )
            
            time.sleep(0.5)  # Check twice per second
        
        attack_thread.join()
        self.attack_active = False
        
        return metrics.calculate_metrics()
    
    def run_full_experiment(self):
        """Run complete integrated experiment"""
        logger.info("="*80)
        logger.info("INTEGRATED REAL-WORLD EXPERIMENT")
        logger.info("="*80)
        
        # Setup
        self.setup_mqtt()
        time.sleep(2)  # Wait for connection
        
        # Phase 1: Baseline
        self.run_baseline_collection(duration=30)
        
        # Phase 2: DoS Attack
        dos_results = self.run_attack_experiment('DoS', duration=60)
        logger.info(f"DoS Results: {dos_results}")
        
        # Wait between attacks
        time.sleep(10)
        
        # Phase 3: Data Injection Attack  
        injection_results = self.run_attack_experiment('DataInjection', duration=60)
        logger.info(f"Injection Results: {injection_results}")
        
        # Cleanup
        self.mqtt_client.loop_stop()
        
        logger.info("="*80)
        logger.info("EXPERIMENT COMPLETE")
        logger.info("="*80)


if __name__ == "__main__":
    experiment = IntegratedExperiment()
    
    print("\n" + "="*80)
    print("REAL INTEGRATED EXPERIMENT")
    print("This will take ~3 minutes (30s baseline + 60s DoS + 60s injection)")
    print("Make sure Docker devices are running!")
    print("="*80 + "\n")
    
    experiment.run_full_experiment()
