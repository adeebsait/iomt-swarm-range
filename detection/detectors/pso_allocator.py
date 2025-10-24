"""
PSO-Based Resource Allocation Optimizer

Uses Particle Swarm Optimization to allocate detection resources
across devices for optimal threat detection
"""
import numpy as np
from swarm.pso import ParticleSwarmOptimization
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PSOResourceAllocator:
    """Uses PSO to optimize resource allocation for threat detection"""
    
    def __init__(self, num_devices: int = 10, total_resources: float = 100.0):
        self.num_devices = num_devices
        self.total_resources = total_resources
        self.allocation = None
        
    def optimize_allocation(self, threat_levels: np.ndarray, device_costs: np.ndarray, 
                           max_iterations: int = 100):
        """
        Optimize resource allocation using PSO
        
        Args:
            threat_levels: Threat level for each device (0-1)
            device_costs: Resource cost per device
            max_iterations: PSO iterations
            
        Returns:
            dict: Optimization results
        """
        logger.info(f"Optimizing resource allocation for {self.num_devices} devices...")
        
        # Define fitness function: maximize detection while minimizing cost
        def fitness_function(allocation):
            # Normalize allocation to sum to total_resources
            normalized = allocation * (self.total_resources / np.sum(allocation))
            
            # Calculate detection capability (higher allocation = better detection)
            detection_score = np.sum(normalized * threat_levels)
            
            # Calculate cost efficiency (penalize expensive allocations)
            cost = np.sum(normalized * device_costs)
            cost_penalty = cost / self.total_resources
            
            # Balance detection and cost
            return detection_score * (1.0 / (1.0 + cost_penalty))
        
        # Run PSO optimization
        pso = ParticleSwarmOptimization(
            objective_function=fitness_function,
            dim=self.num_devices,
            n_particles=30,
            n_iterations=max_iterations,
            bounds=(0, self.total_resources),
            maximize=True
        )
        
        best_solution, best_fitness = pso.optimize()
        
        # Normalize final allocation
        self.allocation = best_solution * (self.total_resources / np.sum(best_solution))
        
        logger.info(f"Optimization complete! Fitness score: {best_fitness:.4f}")
        
        return {
            'allocation': self.allocation,
            'fitness_score': best_fitness,
            'total_allocated': np.sum(self.allocation),
            'efficiency': best_fitness / self.total_resources
        }
    
    def get_allocation(self, device_id: int) -> float:
        """Get allocated resources for a device"""
        if self.allocation is None:
            return 0.0
        return self.allocation[device_id]
    
    def reallocate_dynamic(self, current_threats: np.ndarray):
        """
        Dynamically reallocate based on current threat levels
        
        Args:
            current_threats: Current threat levels (0-1)
        """
        if self.allocation is None:
            return
        
        # Adjust allocation based on threat changes
        threat_ratio = current_threats / (np.sum(current_threats) + 1e-6)
        self.allocation = threat_ratio * self.total_resources


if __name__ == "__main__":
    # Test PSO allocator
    allocator = PSOResourceAllocator(num_devices=5, total_resources=100.0)
    
    # Device threat levels (higher = more threatened)
    threats = np.array([0.9, 0.3, 0.7, 0.2, 0.8])
    
    # Device costs (complexity)
    costs = np.array([1.0, 1.5, 0.8, 1.2, 0.9])
    
    # Optimize allocation
    results = allocator.optimize_allocation(threats, costs, max_iterations=50)
    
    print("\nOptimized Resource Allocation:")
    for i, alloc in enumerate(results['allocation']):
        print(f"  Device {i+1}: {alloc:.2f} units (Threat: {threats[i]:.1f}, Cost: {costs[i]:.1f})")
    
    print(f"\nTotal Allocated: {results['total_allocated']:.2f}")
    print(f"Efficiency Score: {results['efficiency']:.4f}")
