"""
ACO-Based Alert Routing Optimizer

Uses Ant Colony Optimization to find optimal paths for
routing security alerts through the network
"""
import numpy as np
from swarm.aco import AntColonyOptimization
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ACOAlertRouter:
    """Uses ACO to optimize alert routing paths"""
    
    def __init__(self, num_nodes: int = 10):
        self.num_nodes = num_nodes
        self.distance_matrix = None
        self.best_path = None
        
    def set_network_topology(self, distances: np.ndarray):
        """
        Set network distance/latency matrix
        
        Args:
            distances: NxN matrix of latencies between nodes
        """
        self.distance_matrix = distances
    
    def optimize_routing(self, source: int, destination: int, max_iterations: int = 100):
        """
        Find optimal alert routing path using ACO
        
        Args:
            source: Source node ID
            destination: Destination node ID
            max_iterations: ACO iterations
            
        Returns:
            dict: Optimal path and latency
        """
        if self.distance_matrix is None:
            raise ValueError("Network topology not set!")
        
        logger.info(f"Optimizing route from node {source} to {destination}...")
        
        # ACO will find shortest path
        aco = AntColonyOptimization(
            distance_matrix=self.distance_matrix,
            n_ants=20,
            n_iterations=max_iterations,
            alpha=1.0,
            beta=2.0,
            evaporation_rate=0.5
        )
        
        best_path, best_distance = aco.optimize()
        
        # Adjust path to start from source and end at destination
        # For simplicity, use the ACO tour and extract relevant segment
        try:
            source_idx = list(best_path).index(source)
            dest_idx = list(best_path).index(destination)
            
            if source_idx < dest_idx:
                self.best_path = best_path[source_idx:dest_idx+1]
            else:
                self.best_path = best_path[source_idx:] + best_path[:dest_idx+1]
        except ValueError:
            self.best_path = [source, destination]  # Direct path
            best_distance = self.distance_matrix[source, destination]
        
        logger.info(f"Optimal path found with latency: {best_distance:.2f}ms")
        
        return {
            'path': self.best_path,
            'latency': best_distance,
            'hops': len(self.best_path) - 1
        }
    
    def route_alert(self, alert_data: dict) -> list:
        """
        Route an alert through the optimized path
        
        Args:
            alert_data: Alert information
            
        Returns:
            list: Path nodes
        """
        if self.best_path is None:
            return []
        
        logger.info(f"Routing alert: {alert_data.get('type', 'unknown')} via {len(self.best_path)} nodes")
        return self.best_path
    
    def calculate_latency(self, path: list) -> float:
        """Calculate total latency for a path"""
        if self.distance_matrix is None or len(path) < 2:
            return 0.0
        
        latency = sum(
            self.distance_matrix[path[i], path[i+1]]
            for i in range(len(path) - 1)
        )
        return latency


if __name__ == "__main__":
    # Test ACO router
    router = ACOAlertRouter(num_nodes=6)
    
    # Create sample network with latencies (ms)
    np.random.seed(42)
    distances = np.random.uniform(10, 100, (6, 6))
    np.fill_diagonal(distances, 0)
    distances = (distances + distances.T) / 2  # Make symmetric
    
    router.set_network_topology(distances)
    
    # Find optimal path from node 0 to node 5
    result = router.optimize_routing(source=0, destination=5, max_iterations=50)
    
    print("\nOptimal Alert Routing:")
    print(f"  Path: {' -> '.join(map(str, result['path']))}")
    print(f"  Latency: {result['latency']:.2f}ms")
    print(f"  Hops: {result['hops']}")
    
    # Test alert routing
    alert = {'type': 'dos_attack', 'severity': 'high'}
    path = router.route_alert(alert)
    print(f"\nAlert routed through: {path}")
