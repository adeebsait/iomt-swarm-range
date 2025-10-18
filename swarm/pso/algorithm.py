"""
Particle Swarm Optimization (PSO) for Resource Allocation

Minimizes detection latency by optimally allocating computational resources
(CPU, memory, bandwidth) to distributed detection agents in IoMT networks.
"""
import numpy as np
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class Particle:
    """Represents a particle (candidate resource allocation)"""
    position: np.ndarray  # Resource allocation vector
    velocity: np.ndarray  # Velocity vector
    best_position: np.ndarray  # Personal best position
    best_fitness: float = float('inf')  # Personal best fitness (minimize)
    
    def copy(self) -> 'Particle':
        """Create a deep copy of this particle"""
        return Particle(
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            best_position=self.best_position.copy(),
            best_fitness=self.best_fitness
        )


class PSOAlgorithm:
    """
    Particle Swarm Optimization for resource allocation in distributed IoMT detection.
    
    Minimizes: J = w1*latency + w2*queue_length + w3*drop_rate
    
    Each particle represents resource allocation: [cpu1, mem1, bw1, cpu2, mem2, bw2, ...]
    """
    
    def __init__(self, config: Dict[str, Any]):
        # PSO parameters
        self.n_particles = config.get("particles", 32)
        self.omega = config.get("omega", 0.72)      # Inertia weight
        self.c1 = config.get("c1", 1.49)            # Cognitive parameter
        self.c2 = config.get("c2", 1.49)            # Social parameter
        self.max_iters = config.get("iters_per_window", 3)
        
        # Resource bounds
        bounds = config.get("resource_bounds", {})
        self.cpu_bounds = tuple(bounds.get("cpu", [0.5, 4.0]))
        self.mem_bounds = tuple(bounds.get("mem", [1.0, 8.0]))
        self.bw_bounds = tuple(bounds.get("bw", [10.0, 1000.0]))
        
        # Fitness weights
        self.w1 = 0.6  # Latency weight
        self.w2 = 0.3  # Queue length weight
        self.w3 = 0.1  # Drop rate weight
        
        # State
        self.particles: List[Particle] = []
        self.global_best_position: np.ndarray = None
        self.global_best_fitness = float('inf')
        self.iteration = 0
        self.n_agents = 0
        self.dim = 0  # Dimension (n_agents * 3)
        
        # History
        self.fitness_history: List[float] = []
        
    def initialize_swarm(self, n_agents: int) -> None:
        """
        Initialize particle swarm for n_agents.
        
        Args:
            n_agents: Number of detection agents to allocate resources for
        """
        self.n_agents = n_agents
        self.dim = n_agents * 3  # CPU, memory, bandwidth per agent
        
        logger.info(f"Initializing PSO with {self.n_particles} particles "
                   f"for {n_agents} agents (dimension={self.dim})")
        
        # Create particles with random initialization
        for _ in range(self.n_particles):
            position = self._random_position()
            velocity = np.random.uniform(-0.5, 0.5, self.dim)
            
            particle = Particle(
                position=position,
                velocity=velocity,
                best_position=position.copy(),
                best_fitness=float('inf')
            )
            self.particles.append(particle)
        
        logger.info(f"PSO initialized with {len(self.particles)} particles")
    
    def _random_position(self) -> np.ndarray:
        """Generate random position within resource bounds"""
        position = []
        for _ in range(self.n_agents):
            cpu = np.random.uniform(*self.cpu_bounds)
            mem = np.random.uniform(*self.mem_bounds)
            bw = np.random.uniform(*self.bw_bounds)
            position.extend([cpu, mem, bw])
        return np.array(position)
    
    def _evaluate_fitness(
        self, 
        position: np.ndarray,
        agent_metrics: Dict[int, Dict[str, float]]
    ) -> float:
        """
        Evaluate fitness (objective function to minimize).
        
        J = w1*latency + w2*queue_length + w3*drop_rate
        
        Args:
            position: Resource allocation vector
            agent_metrics: Current metrics from each agent
            
        Returns:
            Fitness value (lower is better)
        """
        total_latency = 0.0
        total_queue = 0.0
        total_drops = 0.0
        
        for agent_id in range(self.n_agents):
            idx = agent_id * 3
            cpu_alloc = position[idx]
            mem_alloc = position[idx + 1]
            bw_alloc = position[idx + 2]
            
            # Get current agent metrics
            metrics = agent_metrics.get(agent_id, {
                "cpu_usage": 0.5,
                "mem_usage": 0.4,
                "queue_length": 10.0,
                "packet_rate": 100.0
            })
            
            current_cpu = metrics.get("cpu_usage", 0.5)
            current_mem = metrics.get("mem_usage", 0.4)
            current_queue = metrics.get("queue_length", 10.0)
            packet_rate = metrics.get("packet_rate", 100.0)
            
            # Latency model: inversely proportional to allocated resources
            # If queue is high and resources are low, latency increases
            cpu_factor = max(0.1, cpu_alloc - current_cpu)
            mem_factor = max(0.1, mem_alloc - current_mem)
            
            latency = current_queue / (cpu_factor * mem_factor + 0.1)
            
            # Queue length model: depends on processing capacity vs arrival rate
            processing_capacity = cpu_alloc * 50  # packets/sec per core
            queue_growth = max(0, packet_rate - processing_capacity)
            queue_factor = (current_queue + queue_growth) / (mem_alloc + 0.1)
            
            # Drop rate model: occurs when resources are insufficient
            cpu_deficit = max(0, current_cpu - cpu_alloc)
            mem_deficit = max(0, current_mem - mem_alloc)
            bw_deficit = max(0, packet_rate * 0.001 - bw_alloc)  # Convert to Mbps
            
            drop_rate = cpu_deficit + mem_deficit + bw_deficit * 0.001
            
            total_latency += latency
            total_queue += queue_factor
            total_drops += drop_rate
        
        # Compute weighted objective
        fitness = (
            self.w1 * total_latency + 
            self.w2 * total_queue + 
            self.w3 * total_drops
        )
        
        return fitness
    
    def step(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute one PSO iteration.
        
        Args:
            observation: Dictionary containing:
                - agent_metrics: Dict[agent_id, metrics_dict]
                
        Returns:
            Dictionary with resource allocation and metrics
        """
        if not self.particles:
            raise RuntimeError("Swarm not initialized. Call initialize_swarm() first.")
        
        agent_metrics = observation.get("agent_metrics", {})
        
        # Evaluate all particles
        for particle in self.particles:
            fitness = self._evaluate_fitness(particle.position, agent_metrics)
            
            # Update personal best
            if fitness < particle.best_fitness:
                particle.best_fitness = fitness
                particle.best_position = particle.position.copy()
            
            # Update global best
            if fitness < self.global_best_fitness:
                self.global_best_fitness = fitness
                self.global_best_position = particle.position.copy()
                logger.info(f"PSO iter {self.iteration}: "
                          f"New global best fitness={fitness:.4f}")
        
        # Update velocities and positions
        for particle in self.particles:
            # Random factors
            r1 = np.random.random(self.dim)
            r2 = np.random.random(self.dim)
            
            # Cognitive component (personal best)
            cognitive = self.c1 * r1 * (particle.best_position - particle.position)
            
            # Social component (global best)
            social = self.c2 * r2 * (self.global_best_position - particle.position)
            
            # Update velocity
            particle.velocity = (
                self.omega * particle.velocity + 
                cognitive + 
                social
            )
            
            # Velocity clamping (prevent explosion)
            max_velocity = 1.0
            particle.velocity = np.clip(particle.velocity, -max_velocity, max_velocity)
            
            # Update position
            particle.position = particle.position + particle.velocity
            
            # Enforce bounds
            particle.position = self._apply_bounds(particle.position)
        
        self.iteration += 1
        self.fitness_history.append(self.global_best_fitness)
        
        # Parse allocation for output
        allocation = self._parse_allocation(self.global_best_position)
        
        return {
            "resource_allocation": allocation,
            "fitness": self.global_best_fitness,
            "iteration": self.iteration,
            "converged": self.iteration >= self.max_iters,
            "estimated_latency": self._estimate_latency(agent_metrics),
            "total_resources": self._total_resources(allocation)
        }
    
    def _apply_bounds(self, position: np.ndarray) -> np.ndarray:
        """Clip position vector to resource bounds"""
        bounded = position.copy()
        for i in range(self.n_agents):
            idx = i * 3
            bounded[idx] = np.clip(bounded[idx], *self.cpu_bounds)
            bounded[idx + 1] = np.clip(bounded[idx + 1], *self.mem_bounds)
            bounded[idx + 2] = np.clip(bounded[idx + 2], *self.bw_bounds)
        return bounded
    
    def _parse_allocation(self, position: np.ndarray) -> Dict[int, Dict[str, float]]:
        """Convert position vector to human-readable allocation dict"""
        allocation = {}
        for agent_id in range(self.n_agents):
            idx = agent_id * 3
            allocation[agent_id] = {
                "cpu_cores": round(position[idx], 2),
                "memory_gb": round(position[idx + 1], 2),
                "bandwidth_mbps": round(position[idx + 2], 2)
            }
        return allocation
    
    def _estimate_latency(self, agent_metrics: Dict[int, Dict[str, float]]) -> float:
        """Estimate average latency with current allocation"""
        if not self.global_best_position is not None:
            return 0.0
        
        total_latency = 0.0
        for agent_id in range(self.n_agents):
            idx = agent_id * 3
            cpu_alloc = self.global_best_position[idx]
            
            metrics = agent_metrics.get(agent_id, {"queue_length": 10.0})
            queue = metrics.get("queue_length", 10.0)
            
            latency = queue / (cpu_alloc + 0.1)
            total_latency += latency
        
        return total_latency / self.n_agents if self.n_agents > 0 else 0.0
    
    def _total_resources(self, allocation: Dict[int, Dict[str, float]]) -> Dict[str, float]:
        """Calculate total allocated resources"""
        total = {"cpu": 0.0, "memory": 0.0, "bandwidth": 0.0}
        for agent_alloc in allocation.values():
            total["cpu"] += agent_alloc["cpu_cores"]
            total["memory"] += agent_alloc["memory_gb"]
            total["bandwidth"] += agent_alloc["bandwidth_mbps"]
        return total
    
    def state_dict(self) -> Dict[str, Any]:
        """Serialize algorithm state"""
        return {
            "iteration": self.iteration,
            "global_best_fitness": self.global_best_fitness,
            "global_best_position": self.global_best_position.tolist() 
                if self.global_best_position is not None else [],
            "fitness_history": self.fitness_history,
            "n_agents": self.n_agents,
            "config": {
                "n_particles": self.n_particles,
                "omega": self.omega,
                "c1": self.c1,
                "c2": self.c2
            }
        }
    
    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Restore algorithm state"""
        self.iteration = state["iteration"]
        self.global_best_fitness = state["global_best_fitness"]
        self.fitness_history = state["fitness_history"]
        self.n_agents = state["n_agents"]
        self.dim = self.n_agents * 3
        
        if state["global_best_position"]:
            self.global_best_position = np.array(state["global_best_position"])
        
        logger.info(f"PSO state loaded: iteration={self.iteration}, "
                   f"fitness={self.global_best_fitness:.4f}")
