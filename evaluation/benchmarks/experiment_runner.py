"""
Automated Experiment Runner

Runs comparative experiments between swarm algorithms and baselines
"""
import time
import json
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List
import logging

from evaluation.metrics.detector_metrics import DetectionMetrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExperimentRunner:
    """Automates detection experiments and benchmarking"""
    
    def __init__(self, output_dir: str = "evaluation/results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = []
        
    def run_single_experiment(self, 
                             algorithm: str,
                             attack_type: str,
                             duration: int = 60,
                             trial: int = 1) -> Dict:
        """
        Run a single experiment trial
        
        Args:
            algorithm: Detection algorithm (ABC, PSO, ACO, Hybrid, Baseline)
            attack_type: Type of attack (DoS, DataInjection, etc.)
            duration: Experiment duration in seconds
            trial: Trial number
            
        Returns:
            dict: Experiment results
        """
        logger.info(f"Starting Trial {trial}: {algorithm} vs {attack_type}")
        
        start_time = time.time()
        metrics = DetectionMetrics()
        
        # Simulate detection process
        # In real implementation, this would:
        # 1. Start attack
        # 2. Run detection algorithm
        # 3. Record detections
        
        # For now, simulate results
        np.random.seed(trial)
        num_samples = duration * 2  # 2 samples per second
        
        for i in range(num_samples):
            # Simulate attack presence (30% of time)
            actual_threat = np.random.random() < 0.3
            
            # Algorithm-specific detection rates
            detection_rate = {
                'ABC': 0.92,
                'PSO': 0.89,
                'ACO': 0.91,
                'Hybrid': 0.95,
                'Baseline': 0.78
            }.get(algorithm, 0.80)
            
            # Add noise based on attack type
            if attack_type == 'DoS':
                detection_rate *= 1.05  # Easier to detect
            elif attack_type == 'DataInjection':
                detection_rate *= 0.9   # Harder to detect
            
            # Simulate detection
            detected = actual_threat and (np.random.random() < detection_rate)
            latency = np.random.exponential(1.5) if detected else 0
            
            metrics.record_detection(detected, actual_threat, latency)
        
        elapsed_time = time.time() - start_time
        
        # Calculate metrics
        results = metrics.calculate_metrics()
        results.update({
            'algorithm': algorithm,
            'attack_type': attack_type,
            'duration': duration,
            'trial': trial,
            'elapsed_time': elapsed_time,
            'timestamp': datetime.now().isoformat()
        })
        
        logger.info(f"  Accuracy: {results['accuracy']:.2%}, "
                   f"F1: {results['f1_score']:.2%}, "
                   f"Latency: {results['avg_detection_latency']:.3f}s")
        
        return results
    
    def run_comparative_experiment(self, 
                                   algorithms: List[str],
                                   attack_types: List[str],
                                   trials: int = 30,
                                   duration: int = 60) -> List[Dict]:
        """
        Run complete comparative experiment
        
        Args:
            algorithms: List of algorithms to compare
            attack_types: List of attack types to test
            trials: Number of trials per configuration
            duration: Duration per trial
            
        Returns:
            list: All experiment results
        """
        logger.info("="*70)
        logger.info("STARTING COMPARATIVE EXPERIMENT")
        logger.info(f"Algorithms: {algorithms}")
        logger.info(f"Attack Types: {attack_types}")
        logger.info(f"Trials per config: {trials}")
        logger.info("="*70)
        
        all_results = []
        total_experiments = len(algorithms) * len(attack_types) * trials
        completed = 0
        
        for algorithm in algorithms:
            for attack_type in attack_types:
                for trial in range(1, trials + 1):
                    result = self.run_single_experiment(
                        algorithm=algorithm,
                        attack_type=attack_type,
                        duration=duration,
                        trial=trial
                    )
                    all_results.append(result)
                    
                    completed += 1
                    progress = 100 * completed / total_experiments
                    logger.info(f"Progress: {progress:.1f}% ({completed}/{total_experiments})")
        
        # Save results
        self.results = all_results
        self.save_results()
        
        logger.info("="*70)
        logger.info("EXPERIMENT COMPLETE")
        logger.info(f"Total trials: {len(all_results)}")
        logger.info(f"Results saved to: {self.output_dir}")
        logger.info("="*70)
        
        return all_results
    
    def save_results(self, filename: str = None):
        """Save experiment results to JSON"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"experiment_results_{timestamp}.json"
        
        output_file = self.output_dir / filename
        
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        logger.info(f"Results saved to: {output_file}")
    
    def generate_summary(self) -> Dict:
        """Generate summary statistics from results"""
        if not self.results:
            return {}
        
        summary = {}
        
        # Group by algorithm
        for algorithm in set(r['algorithm'] for r in self.results):
            algo_results = [r for r in self.results if r['algorithm'] == algorithm]
            
            summary[algorithm] = {
                'accuracy': {
                    'mean': np.mean([r['accuracy'] for r in algo_results]),
                    'std': np.std([r['accuracy'] for r in algo_results])
                },
                'f1_score': {
                    'mean': np.mean([r['f1_score'] for r in algo_results]),
                    'std': np.std([r['f1_score'] for r in algo_results])
                },
                'latency': {
                    'mean': np.mean([r['avg_detection_latency'] for r in algo_results]),
                    'std': np.std([r['avg_detection_latency'] for r in algo_results])
                },
                'trials': len(algo_results)
            }
        
        return summary


if __name__ == "__main__":
    # Run sample experiment
    runner = ExperimentRunner()
    
    algorithms = ['ABC', 'PSO', 'ACO', 'Hybrid', 'Baseline']
    attack_types = ['DoS', 'DataInjection']
    
    print("Starting sample experiment (5 trials per config)...")
    results = runner.run_comparative_experiment(
        algorithms=algorithms,
        attack_types=attack_types,
        trials=5,  # Small test
        duration=10  # 10 seconds per trial
    )
    
    # Generate summary
    summary = runner.generate_summary()
    
    print("\n=== EXPERIMENT SUMMARY ===")
    for algo, stats in summary.items():
        print(f"\n{algo}:")
        print(f"  Accuracy: {stats['accuracy']['mean']:.2%} ± {stats['accuracy']['std']:.2%}")
        print(f"  F1-Score: {stats['f1_score']['mean']:.2%} ± {stats['f1_score']['std']:.2%}")
        print(f"  Latency:  {stats['latency']['mean']:.3f}s ± {stats['latency']['std']:.3f}s")
