"""
ABC-Based Threat Detection Optimizer

Uses Artificial Bee Colony to optimize sensor placement
for maximum attack detection coverage
"""
import numpy as np
from swarm.abc import ArtificialBeeColony
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ABCThreatDetector:
    """Uses ABC to optimize detection sensor placement"""
    
    def __init__(self, num_sensors: int = 10, network_size: int = 100):
        self.num_sensors = num_sensors
        self.network_size = network_size
        self.sensor_positions = None
        self.coverage_map = None
        
    def optimize_placement(self, threat_history: np.ndarray, max_iterations: int = 100):
        """
        Optimize sensor placement using ABC
        
        Args:
            threat_history: Historical threat locations (Nx2 array)
            max_iterations: ABC iterations
            
        Returns:
            dict: Optimization results
        """
        logger.info(f"Optimizing placement of {self.num_sensors} sensors...")
        
        # Define fitness function: maximize coverage of threat-prone areas
        def fitness_function(positions):
            # positions is flattened array of sensor coords
            sensors = positions.reshape(self.num_sensors, 2)
            
            # Calculate coverage: sum of inverse distances to threats
            coverage = 0
            for threat in threat_history:
                distances = np.linalg.norm(sensors - threat, axis=1)
                # Closer sensors = better coverage
                coverage += np.sum(1.0 / (distances + 1.0))
            
            return coverage
        
        # Run ABC optimization
        abc = ArtificialBeeColony(
            objective_function=fitness_function,
            dim=self.num_sensors * 2,  # x,y for each sensor
            n_bees=30,
            n_iterations=max_iterations,
            limits=(0, self.network_size),
            maximize=True
        )
        
        best_solution, best_fitness = abc.optimize()
        
        # Store optimized positions
        self.sensor_positions = best_solution.reshape(self.num_sensors, 2)
        
        logger.info(f"Optimization complete! Coverage score: {best_fitness:.2f}")
        
        return {
            'sensor_positions': self.sensor_positions,
            'coverage_score': best_fitness,
            'iterations': max_iterations
        }
    
    def detect_threat(self, location: np.ndarray, threshold: float = 10.0) -> bool:
        """
        Detect if a threat is near any sensor
        
        Args:
            location: Threat location (x, y)
            threshold: Detection distance threshold
            
        Returns:
            bool: True if threat detected
        """
        if self.sensor_positions is None:
            return False
        
        distances = np.linalg.norm(self.sensor_positions - location, axis=1)
        return np.any(distances < threshold)
    
    def get_detection_coverage(self, test_points: np.ndarray, threshold: float = 10.0) -> float:
        """
        Calculate detection coverage percentage
        
        Args:
            test_points: Points to test (Nx2 array)
            threshold: Detection radius
            
        Returns:
            float: Coverage percentage (0-100)
        """
        if self.sensor_positions is None:
            return 0.0
        
        detected = sum(self.detect_threat(point, threshold) for point in test_points)
        return 100.0 * detected / len(test_points)


if __name__ == "__main__":
    # Test ABC detector
    detector = ABCThreatDetector(num_sensors=5, network_size=100)
    
    # Simulate historical threat locations (clustered around 50,50)
    np.random.seed(42)
    threats = np.random.randn(50, 2) * 10 + 50
    
    # Optimize sensor placement
    results = detector.optimize_placement(threats, max_iterations=50)
    
    print("\nOptimized Sensor Positions:")
    for i, pos in enumerate(results['sensor_positions']):
        print(f"  Sensor {i+1}: ({pos[0]:.1f}, {pos[1]:.1f})")
    
    # Test coverage
    test_points = np.random.uniform(0, 100, (200, 2))
    coverage = detector.get_detection_coverage(test_points, threshold=15.0)
    print(f"\nDetection Coverage: {coverage:.1f}%")
