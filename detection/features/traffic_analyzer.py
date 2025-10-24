"""
Network Traffic Feature Extractor

Extracts features from MQTT telemetry for anomaly detection
"""
import numpy as np
from collections import deque
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TrafficAnalyzer:
    """Analyzes MQTT traffic patterns to extract detection features"""
    
    def __init__(self, window_size: int = 50):
        self.window_size = window_size
        self.message_times = deque(maxlen=window_size)
        self.message_sizes = deque(maxlen=window_size)
        self.topics = deque(maxlen=window_size)
        self.devices = deque(maxlen=window_size)
        
    def add_message(self, timestamp: float, size: int, topic: str, device_id: str):
        """Record a new MQTT message"""
        self.message_times.append(timestamp)
        self.message_sizes.append(size)
        self.topics.append(topic)
        self.devices.append(device_id)
    
    def extract_features(self) -> dict:
        """
        Extract detection features from recent traffic
        
        Returns:
            dict: Feature vector for anomaly detection
        """
        if len(self.message_times) < 5:
            return None
        
        # Calculate time-based features
        times = np.array(self.message_times)
        intervals = np.diff(times)
        
        # Calculate size-based features
        sizes = np.array(self.message_sizes)
        
        # Calculate diversity features
        unique_topics = len(set(self.topics))
        unique_devices = len(set(self.devices))
        
        features = {
            # Rate features
            'message_rate': len(self.message_times) / (times[-1] - times[0]) if len(times) > 1 else 0,
            'avg_interval': np.mean(intervals) if len(intervals) > 0 else 0,
            'std_interval': np.std(intervals) if len(intervals) > 0 else 0,
            
            # Size features
            'avg_size': np.mean(sizes),
            'std_size': np.std(sizes),
            'max_size': np.max(sizes),
            'min_size': np.min(sizes),
            
            # Diversity features
            'topic_diversity': unique_topics / len(self.topics),
            'device_diversity': unique_devices / len(self.devices),
            
            # Burstiness (coefficient of variation)
            'burstiness': np.std(intervals) / np.mean(intervals) if len(intervals) > 0 and np.mean(intervals) > 0 else 0,
        }
        
        return features
    
    def is_anomalous(self, features: dict, baseline: dict, threshold: float = 2.5) -> bool:
        """
        Detect if current traffic is anomalous compared to baseline
        
        Args:
            features: Current traffic features
            baseline: Normal traffic baseline statistics
            threshold: Number of standard deviations for anomaly
            
        Returns:
            bool: True if anomalous
        """
        if not features or not baseline:
            return False
        
        # Check each feature for deviation from baseline
        anomaly_score = 0
        
        for key in features:
            if key in baseline:
                mean = baseline[key]['mean']
                std = baseline[key]['std']
                
                if std > 0:
                    z_score = abs(features[key] - mean) / std
                    if z_score > threshold:
                        anomaly_score += z_score
        
        # Return True if cumulative anomaly exceeds threshold
        return anomaly_score > threshold * len(features) * 0.3


if __name__ == "__main__":
    # Test traffic analyzer
    analyzer = TrafficAnalyzer(window_size=20)
    
    import time
    current_time = time.time()
    
    # Simulate normal traffic
    for i in range(15):
        analyzer.add_message(
            timestamp=current_time + i * 2.0,
            size=512,
            topic="iomt/telemetry/ICU/ecg_001",
            device_id="ecg_001"
        )
    
    features = analyzer.extract_features()
    print("Normal traffic features:")
    for key, val in features.items():
        print(f"  {key}: {val:.4f}")
    
    # Simulate attack traffic (high rate)
    for i in range(10):
        analyzer.add_message(
            timestamp=current_time + 30 + i * 0.1,  # 10x faster
            size=128,  # Smaller
            topic=f"iomt/attack/flood/{i}",
            device_id=f"attacker_{i}"
        )
    
    attack_features = analyzer.extract_features()
    print("\nAttack traffic features:")
    for key, val in attack_features.items():
        print(f"  {key}: {val:.4f}")
