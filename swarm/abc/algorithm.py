"""
Artificial Bee Colony (ABC) Algorithm for Sensor Placement Optimization

Optimizes the placement of network sensors to maximize coverage while
minimizing cost and overlap. Used for distributed monitoring in IoMT networks.
"""
import numpy as np
from typing import Dict, List, Any, Set
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class SensorPlacement:
    """Represents a candidate sensor placement solution (food source)"""
    positions: List[int]  # Sensor positions (node/link IDs)
    fitness: float = 0.0
    stagnation_count: int = 0
    
    def copy(self) -> 'SensorPlacement':
        """Create a deep copy of this placement"""
        return SensorPlacement(
            positions=self.positions.copy(),
            fitness=self.fitness,
            stagnation_count=self.stagnation_count
        )


class ABCAlgorithm:
    """
    Artificial Bee Colony for optimizing sensor placement in IoMT networks.
    
    Three bee phases:
    1. Employed bees: Exploit current food sources (placements)
    2. Onlooker bees: Probabilistically select and exploit best sources
    3. Scout bees: Abandon stagnant sources and explore new ones
    """
    
    def __init__(self, config: Dict[str, Any]):
        # Algorithm parameters
        self.alpha = config.get("alpha", 1.0)      # Coverage weight
        self.beta = config.get("beta", 0.3)        # Cost weight
        self.gamma = config.get("gamma", 0.1)      # Overlap penalty
        self.max_iters = config.get("iters", 100)
        self.stagnation_limit = config.get("stagnation_limit", 10)
        
        # Population settings
        self.n_bees = config.get("n_bees", 50)
        self.n_employed = self.n_bees // 2
        self.n_onlookers = self.n_bees // 2
        
        # State
        self.population: List[SensorPlacement] = []
        self.best_solution: SensorPlacement = None
        self.iteration = 0
        self.topology: Dict[int, List[int]] = {}
        self.n_positions = 0
        self.max_sensors = 0
        
        # History for analysis
        self.fitness_history: List[float] = []
        
    def initialize_population(
        self, 
        n_positions: int,
        max_sensors: int,
        topology_graph: Dict[int, List[int]]
    ) -> None:
        """
        Initialize bee population with random sensor placements.
        
        Args:
            n_positions: Total number of possible sensor positions
            max_sensors: Maximum number of sensors that can be deployed
            topology_graph: Network topology {node_id: [neighbor_ids]}
        """
        self.topology = topology_graph
        self.n_positions = n_positions
        self.max_sensors = max_sensors
        
        logger.info(f"Initializing ABC with {self.n_bees} bees, "
                   f"{n_positions} positions, max {max_sensors} sensors")
        
        # Generate initial population
        for _ in range(self.n_bees):
            # Random number of sensors (at least 1)
            n_sensors = np.random.randint(1, max_sensors + 1)
            
            # Random positions without replacement
            positions = np.random.choice(
                n_positions, 
                size=n_sensors, 
                replace=False
            ).tolist()
            
            placement = SensorPlacement(positions=positions)
            placement.fitness = self._evaluate_fitness(placement)
            self.population.append(placement)
        
        # Sort by fitness (descending)
        self.population.sort(key=lambda x: x.fitness, reverse=True)
        self.best_solution = self.population[0].copy()
        self.fitness_history.append(self.best_solution.fitness)
        
        logger.info(f"ABC initialized: best fitness={self.best_solution.fitness:.4f}, "
                   f"sensors={len(self.best_solution.positions)}")
    
    def _evaluate_fitness(self, placement: SensorPlacement) -> float:
        """
        Evaluate fitness of a sensor placement.
        
        Fitness = α*coverage - β*cost - γ*overlap
        
        Returns:
            Fitness score (higher is better)
        """
        if not placement.positions:
            return 0.0
        
        positions_set = set(placement.positions)
        
        # 1. Coverage: How many nodes are monitored
        covered_nodes: Set[int] = set()
        for pos in placement.positions:
            # Sensor covers its own position
            covered_nodes.add(pos)
            # Sensor covers neighboring nodes
            if pos in self.topology:
                covered_nodes.update(self.topology[pos])
        
        coverage = len(covered_nodes) / self.n_positions
        
        # 2. Cost: Number of sensors (normalized)
        cost = len(placement.positions) / self.max_sensors
        
        # 3. Overlap: Redundant coverage penalty
        overlap = 0.0
        if len(placement.positions) > 1:
            overlap_count = 0
            for i, pos1 in enumerate(placement.positions):
                neighbors1 = set(self.topology.get(pos1, []))
                for pos2 in placement.positions[i+1:]:
                    neighbors2 = set(self.topology.get(pos2, []))
                    overlap_count += len(neighbors1 & neighbors2)
            
            # Normalize by possible pairs
            max_pairs = len(placement.positions) * (len(placement.positions) - 1) / 2
            overlap = overlap_count / (max_pairs + 1)  # +1 to avoid division by zero
        
        # Compute fitness
        fitness = self.alpha * coverage - self.beta * cost - self.gamma * overlap
        
        return max(0.0, fitness)  # Ensure non-negative
    
    def _employed_bee_phase(self) -> None:
        """
        Employed bees exploit current food sources.
        Each bee generates a neighbor solution and applies greedy selection.
        """
        for i in range(self.n_employed):
            current = self.population[i]
            
            # Generate neighbor solution
            neighbor = self._generate_neighbor(current)
            neighbor.fitness = self._evaluate_fitness(neighbor)
            
            # Greedy selection
            if neighbor.fitness > current.fitness:
                self.population[i] = neighbor
                self.population[i].stagnation_count = 0
            else:
                self.population[i].stagnation_count += 1
    
    def _generate_neighbor(self, placement: SensorPlacement) -> SensorPlacement:
        """
        Generate a neighbor solution using local search.
        
        Operations:
        - Swap: Replace one sensor position with another
        - Add: Add a new sensor (if below max)
        - Remove: Remove a sensor (if above 1)
        """
        neighbor = placement.copy()
        
        # Randomly select operation
        operations = ["swap", "add", "remove"]
        weights = [0.5, 0.25, 0.25]  # Prefer swaps
        operation = np.random.choice(operations, p=weights)
        
        if operation == "swap" and len(neighbor.positions) > 0:
            # Replace a random sensor with a new position
            idx = np.random.randint(len(neighbor.positions))
            new_pos = np.random.randint(self.n_positions)
            
            # Ensure new position is not already used
            attempts = 0
            while new_pos in neighbor.positions and attempts < 10:
                new_pos = np.random.randint(self.n_positions)
                attempts += 1
            
            neighbor.positions[idx] = new_pos
            
        elif operation == "add" and len(neighbor.positions) < self.max_sensors:
            # Add a new sensor
            new_pos = np.random.randint(self.n_positions)
            
            # Ensure new position is not already used
            attempts = 0
            while new_pos in neighbor.positions and attempts < 10:
                new_pos = np.random.randint(self.n_positions)
                attempts += 1
            
            if new_pos not in neighbor.positions:
                neighbor.positions.append(new_pos)
                
        elif operation == "remove" and len(neighbor.positions) > 1:
            # Remove a random sensor
            idx = np.random.randint(len(neighbor.positions))
            neighbor.positions.pop(idx)
        
        return neighbor
    
    def _onlooker_bee_phase(self) -> None:
        """
        Onlooker bees probabilistically select food sources based on fitness.
        Better sources are more likely to be selected.
        """
        # Calculate selection probabilities (fitness proportional)
        fitnesses = np.array([p.fitness for p in self.population[:self.n_employed]])
        
        # Avoid division by zero
        if fitnesses.sum() == 0:
            probabilities = np.ones(self.n_employed) / self.n_employed
        else:
            probabilities = fitnesses / fitnesses.sum()
        
        for _ in range(self.n_onlookers):
            # Select a food source probabilistically
            idx = np.random.choice(self.n_employed, p=probabilities)
            current = self.population[idx]
            
            # Generate and evaluate neighbor
            neighbor = self._generate_neighbor(current)
            neighbor.fitness = self._evaluate_fitness(neighbor)
            
            # Greedy selection
            if neighbor.fitness > current.fitness:
                self.population[idx] = neighbor
                self.population[idx].stagnation_count = 0
    
    def _scout_bee_phase(self) -> None:
        """
        Scout bees abandon stagnant food sources and explore new ones.
        """
        for i in range(self.n_employed):
            if self.population[i].stagnation_count >= self.stagnation_limit:
                # Generate completely new random solution
                n_sensors = np.random.randint(1, self.max_sensors + 1)
                positions = np.random.choice(
                    self.n_positions,
                    size=n_sensors,
                    replace=False
                ).tolist()
                
                new_placement = SensorPlacement(positions=positions)
                new_placement.fitness = self._evaluate_fitness(new_placement)
                
                self.population[i] = new_placement
                
                logger.debug(f"Scout bee: abandoned stagnant source, "
                           f"new fitness={new_placement.fitness:.4f}")
    
    def step(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute one ABC iteration.
        
        Args:
            observation: Current environment state (unused for ABC)
            
        Returns:
            Dictionary with sensor positions and metrics
        """
        if not self.population:
            raise RuntimeError("Population not initialized. "
                             "Call initialize_population() first.")
        
        # Execute three bee phases
        self._employed_bee_phase()
        self._onlooker_bee_phase()
        self._scout_bee_phase()
        
        # Update best solution
        self.population.sort(key=lambda x: x.fitness, reverse=True)
        
        if self.population[0].fitness > self.best_solution.fitness:
            self.best_solution = self.population[0].copy()
            logger.info(f"ABC iter {self.iteration}: "
                       f"New best fitness={self.best_solution.fitness:.4f}, "
                       f"sensors={len(self.best_solution.positions)}")
        
        self.iteration += 1
        self.fitness_history.append(self.best_solution.fitness)
        
        # Calculate coverage for reporting
        coverage = self._calculate_coverage()
        
        return {
            "sensor_positions": self.best_solution.positions,
            "fitness": self.best_solution.fitness,
            "coverage": coverage,
            "n_sensors": len(self.best_solution.positions),
            "iteration": self.iteration,
            "converged": self.iteration >= self.max_iters
        }
    
    def _calculate_coverage(self) -> float:
        """Calculate percentage of nodes covered by best solution"""
        if not self.best_solution or not self.best_solution.positions:
            return 0.0
        
        covered_nodes = set()
        for pos in self.best_solution.positions:
            covered_nodes.add(pos)
            if pos in self.topology:
                covered_nodes.update(self.topology[pos])
        
        return len(covered_nodes) / self.n_positions
    
    def state_dict(self) -> Dict[str, Any]:
        """Serialize algorithm state for checkpointing"""
        return {
            "iteration": self.iteration,
            "best_solution": {
                "positions": self.best_solution.positions if self.best_solution else [],
                "fitness": self.best_solution.fitness if self.best_solution else 0.0
            },
            "population_size": len(self.population),
            "fitness_history": self.fitness_history,
            "config": {
                "alpha": self.alpha,
                "beta": self.beta,
                "gamma": self.gamma,
                "n_bees": self.n_bees
            }
        }
    
    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Restore algorithm state from checkpoint"""
        self.iteration = state["iteration"]
        self.fitness_history = state["fitness_history"]
        
        best = state["best_solution"]
        self.best_solution = SensorPlacement(
            positions=best["positions"],
            fitness=best["fitness"]
        )
        
        logger.info(f"ABC state loaded: iteration={self.iteration}, "
                   f"fitness={self.best_solution.fitness:.4f}")
