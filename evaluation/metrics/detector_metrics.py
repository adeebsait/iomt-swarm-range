"""
Detection Performance Metrics

Calculates standard metrics for threat detection systems
"""
import numpy as np
from typing import Dict, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DetectionMetrics:
    """Calculate and track detection performance metrics"""
    
    def __init__(self):
        self.true_positives = 0
        self.false_positives = 0
        self.true_negatives = 0
        self.false_negatives = 0
        self.detection_latencies = []
        
    def record_detection(self, predicted: bool, actual: bool, latency: float = 0.0):
        """
        Record a detection result
        
        Args:
            predicted: Was threat predicted?
            actual: Was there actually a threat?
            latency: Time to detect (seconds)
        """
        if predicted and actual:
            self.true_positives += 1
            self.detection_latencies.append(latency)
        elif predicted and not actual:
            self.false_positives += 1
        elif not predicted and not actual:
            self.true_negatives += 1
        else:  # not predicted and actual
            self.false_negatives += 1
    
    def calculate_metrics(self) -> Dict[str, float]:
        """
        Calculate all performance metrics
        
        Returns:
            dict: Performance metrics
        """
        total = self.true_positives + self.false_positives + self.true_negatives + self.false_negatives
        
        if total == 0:
            return {}
        
        # Accuracy
        accuracy = (self.true_positives + self.true_negatives) / total
        
        # Precision (PPV)
        precision = (self.true_positives / (self.true_positives + self.false_positives) 
                    if (self.true_positives + self.false_positives) > 0 else 0.0)
        
        # Recall (Sensitivity/TPR)
        recall = (self.true_positives / (self.true_positives + self.false_negatives)
                 if (self.true_positives + self.false_negatives) > 0 else 0.0)
        
        # F1 Score
        f1_score = (2 * precision * recall / (precision + recall)
                   if (precision + recall) > 0 else 0.0)
        
        # False Positive Rate
        fpr = (self.false_positives / (self.false_positives + self.true_negatives)
              if (self.false_positives + self.true_negatives) > 0 else 0.0)
        
        # Detection latency stats
        avg_latency = np.mean(self.detection_latencies) if self.detection_latencies else 0.0
        std_latency = np.std(self.detection_latencies) if self.detection_latencies else 0.0
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'false_positive_rate': fpr,
            'true_positives': self.true_positives,
            'false_positives': self.false_positives,
            'true_negatives': self.true_negatives,
            'false_negatives': self.false_negatives,
            'avg_detection_latency': avg_latency,
            'std_detection_latency': std_latency,
            'total_samples': total
        }
    
    def reset(self):
        """Reset all counters"""
        self.true_positives = 0
        self.false_positives = 0
        self.true_negatives = 0
        self.false_negatives = 0
        self.detection_latencies = []
    
    def __str__(self) -> str:
        """Pretty print metrics"""
        metrics = self.calculate_metrics()
        if not metrics:
            return "No data recorded"
        
        return f"""
Detection Performance Metrics:
  Accuracy:  {metrics['accuracy']:.2%}
  Precision: {metrics['precision']:.2%}
  Recall:    {metrics['recall']:.2%}
  F1-Score:  {metrics['f1_score']:.2%}
  FPR:       {metrics['false_positive_rate']:.2%}
  
  True Positives:  {metrics['true_positives']}
  False Positives: {metrics['false_positives']}
  True Negatives:  {metrics['true_negatives']}
  False Negatives: {metrics['false_negatives']}
  
  Avg Detection Latency: {metrics['avg_detection_latency']:.3f}s ± {metrics['std_detection_latency']:.3f}s
"""


if __name__ == "__main__":
    # Test metrics
    metrics = DetectionMetrics()
    
    # Simulate detection results
    np.random.seed(42)
    for i in range(100):
        actual_threat = np.random.random() > 0.7  # 30% attack rate
        detected = actual_threat and np.random.random() > 0.1  # 90% detection rate
        latency = np.random.exponential(2.0) if detected else 0
        
        metrics.record_detection(detected, actual_threat, latency)
    
    print(metrics)
    
    # Get metrics dict
    results = metrics.calculate_metrics()
    print("\nMetrics Dictionary:")
    for key, val in results.items():
        if isinstance(val, float):
            print(f"  {key}: {val:.4f}")
        else:
            print(f"  {key}: {val}")
