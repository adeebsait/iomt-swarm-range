"""
Ant Colony Optimization (ACO) for Alert Routing

Optimizes alert routing paths through distributed agent networks using
pheromone-based path selection. Ensures reliable, low-latency alert delivery.
"""
import numpy as np
from typing import Dict, Any, List, Tuple, Set
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class Path:
    """Represents a routing path from source to destination"""
    nodes: List[int]  # Sequence of node IDs
    cost: float = float('inf')  # Path cost (latency)
    reliability: float = 0.0  # Path reliability
    
    def length(self) -> int:
        """Return number of hops"""
        return len(self.nodes) - 1 if len(self.nodes) > 1 else 0


class ACOAlgorithm:
    """
    Ant Colony Optimization for alert routing in distributed IoMT detection.
    
    Uses pheromone trails to find optimal paths that minimize latency
    and maximize reliability for alert propagation.
    """
    
    def __init__(self, config: Dict[str, Any]):
        # ACO parameters
        self.n_ants = config.get("n_ants", 20)
        self.alpha = config.get("alpha", 1.0)      # Pheromone importance
        self.beta = config.get("beta", 2.0)        # Heuristic importance
        self.rho = config.get("rho", 0.3)          # Evaporation rate
        self.q = config.get("q", 100)              # Pheromone deposit constant
        self.pheromone_init = config.get("pheromone_init", 0.1)
        
        # State
        self.pheromone: Dict[Tuple[int, int], float] = {}
        self.heuristic: Dict[Tuple[int, int], float] = {}
        self.graph: Dict[int, List[int]] = {}
        self.best_path: Path = None
        self.iteration = 0
        
        # History
        self.best_cost_history: List[float] = []
        
    def initialize(
        self,
        graph: Dict[int, List[int]],
        edge_costs: Dict[Tuple[int, int], float],
        edge_reliability: Dict[Tuple[int, int], float] = None
    ) -> None:
        """
        Initialize ACO with network graph.
        
        Args:
            graph: Network topology {node_id: [neighbor_ids]}
            edge_costs: Edge costs (latency) {(from, to): cost}
            edge_reliability: Edge reliability {(from, to): reliability}
        """
        self.graph = graph
        
        # Initialize pheromones uniformly
        for node, neighbors in graph.items():
            for neighbor in neighbors:
                edge = (node, neighbor)
                self.pheromone[edge] = self.pheromone_init
        
        # Initialize heuristic information (1/cost)
        for edge, cost in edge_costs.items():
            self.heuristic[edge] = 1.0 / (cost + 0.1)  # Avoid division by zero
        
        # Store reliability if provided
        self.edge_reliability = edge_reliability or {}
        
        logger.info(f"ACO initialized: {self.n_ants} ants, "
                   f"{len(self.pheromone)} edges, "
                   f"α={self.alpha}, β={self.beta}, ρ={self.rho}")
    
    def construct_solution(
        self,
        source: int,
        destination: int,
        max_hops: int = 10
    ) -> Path:
        """
        Construct a path from source to destination using probabilistic selection.
        
        Args:
            source: Starting node
            destination: Target node
            max_hops: Maximum path length
            
        Returns:
            Path object
        """
        current = source
        path_nodes = [current]
        visited = {current}
        total_cost = 0.0
        total_reliability = 1.0
        
        for _ in range(max_hops):
            if current == destination:
                break
            
            # Get unvisited neighbors
            neighbors = [n for n in self.graph.get(current, []) if n not in visited]
            
            if not neighbors:
                # Dead end - return invalid path
                return Path(nodes=path_nodes, cost=float('inf'), reliability=0.0)
            
            # Calculate selection probabilities
            probabilities = self._calculate_probabilities(current, neighbors, visited)
            
            # Select next node probabilistically
            next_node = np.random.choice(neighbors, p=probabilities)
            
            # Update path
            edge = (current, next_node)
            edge_cost = 1.0 / (self.heuristic.get(edge, 0.1) + 0.1)
            edge_rel = self.edge_reliability.get(edge, 0.95)
            
            path_nodes.append(next_node)
            visited.add(next_node)
            total_cost += edge_cost
            total_reliability *= edge_rel
            
            current = next_node
        
        # Check if destination was reached
        if current != destination:
            return Path(nodes=path_nodes, cost=float('inf'), reliability=0.0)
        
        return Path(nodes=path_nodes, cost=total_cost, reliability=total_reliability)
    
    def _calculate_probabilities(
        self,
        current: int,
        neighbors: List[int],
        visited: Set[int]
    ) -> np.ndarray:
        """
        Calculate selection probabilities for neighboring nodes.
        
        P(i,j) = [τ(i,j)^α * η(i,j)^β] / Σ[τ(i,k)^α * η(i,k)^β]
        """
        attractiveness = []
        
        for neighbor in neighbors:
            edge = (current, neighbor)
            
            # Pheromone level
            tau = self.pheromone.get(edge, self.pheromone_init)
            
            # Heuristic information
            eta = self.heuristic.get(edge, 0.1)
            
            # Combined attractiveness
            attract = (tau ** self.alpha) * (eta ** self.beta)
            attractiveness.append(attract)
        
        # Normalize to probabilities
        attractiveness = np.array(attractiveness)
        total = attractiveness.sum()
        
        if total == 0:
            # Uniform distribution if all zero
            return np.ones(len(neighbors)) / len(neighbors)
        
        return attractiveness / total
    
    def update_pheromones(self, paths: List[Path]) -> None:
        """
        Update pheromone levels based on ant paths.
        
        1. Evaporation: τ(i,j) = (1-ρ) * τ(i,j)
        2. Deposit: τ(i,j) += Σ Δτ(i,j)
        """
        # Evaporation
        for edge in self.pheromone:
            self.pheromone[edge] *= (1 - self.rho)
        
        # Deposit pheromones
        for path in paths:
            if path.cost == float('inf'):
                continue  # Skip invalid paths
            
            # Pheromone amount inversely proportional to cost
            delta_tau = self.q / path.cost
            
            # Bonus for high reliability
            delta_tau *= path.reliability
            
            # Update edges in path
            for i in range(len(path.nodes) - 1):
                edge = (path.nodes[i], path.nodes[i + 1])
                if edge in self.pheromone:
                    self.pheromone[edge] += delta_tau
        
        # Ensure minimum pheromone level
        for edge in self.pheromone:
            self.pheromone[edge] = max(self.pheromone[edge], 0.01)
    
    def step(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute one ACO iteration.
        
        Args:
            observation: Dictionary containing:
                - source: Source node for routing
                - destination: Destination node
                - max_hops: Maximum path length (optional)
                
        Returns:
            Dictionary with best path and metrics
        """
        if not self.graph:
            raise RuntimeError("ACO not initialized. Call initialize() first.")
        
        source = observation.get("source", 0)
        destination = observation.get("destination", len(self.graph) - 1)
        max_hops = observation.get("max_hops", 10)
        
        # Construct solutions with all ants
        paths = []
        for _ in range(self.n_ants):
            path = self.construct_solution(source, destination, max_hops)
            paths.append(path)
        
        # Find best path in this iteration
        valid_paths = [p for p in paths if p.cost != float('inf')]
        
        if valid_paths:
            # Sort by cost (lower is better)
            valid_paths.sort(key=lambda p: p.cost)
            iteration_best = valid_paths[0]
            
            # Update global best
            if self.best_path is None or iteration_best.cost < self.best_path.cost:
                self.best_path = iteration_best
                logger.info(f"ACO iter {self.iteration}: "
                          f"New best path cost={self.best_path.cost:.2f}, "
                          f"hops={self.best_path.length()}, "
                          f"reliability={self.best_path.reliability:.3f}")
        
        # Update pheromones
        self.update_pheromones(paths)
        
        self.iteration += 1
        
        if self.best_path:
            self.best_cost_history.append(self.best_path.cost)
        
        # Calculate statistics
        avg_cost = np.mean([p.cost for p in valid_paths]) if valid_paths else float('inf')
        success_rate = len(valid_paths) / self.n_ants
        
        return {
            "best_path": self.best_path.nodes if self.best_path else [],
            "best_cost": self.best_path.cost if self.best_path else float('inf'),
            "best_reliability": self.best_path.reliability if self.best_path else 0.0,
            "path_length": self.best_path.length() if self.best_path else 0,
            "avg_cost": avg_cost,
            "success_rate": success_rate,
            "iteration": self.iteration,
            "pheromone_stats": self._pheromone_statistics()
        }
    
    def _pheromone_statistics(self) -> Dict[str, float]:
        """Calculate pheromone statistics"""
        if not self.pheromone:
            return {"min": 0.0, "max": 0.0, "mean": 0.0}
        
        values = list(self.pheromone.values())
        return {
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "mean": float(np.mean(values))
        }
    
    def get_routing_table(self, sources: List[int], destination: int) -> Dict[int, List[int]]:
        """
        Generate routing table from multiple sources to destination.
        
        Args:
            sources: List of source nodes
            destination: Target node
            
        Returns:
            Dictionary {source: path_to_destination}
        """
        routing_table = {}
        
        for source in sources:
            path = self.construct_solution(source, destination)
            if path.cost != float('inf'):
                routing_table[source] = path.nodes
            else:
                routing_table[source] = []  # No valid path
        
        return routing_table
    
    def state_dict(self) -> Dict[str, Any]:
        """Serialize algorithm state"""
        return {
            "iteration": self.iteration,
            "best_path": {
                "nodes": self.best_path.nodes if self.best_path else [],
                "cost": self.best_path.cost if self.best_path else float('inf'),
                "reliability": self.best_path.reliability if self.best_path else 0.0
            },
            "pheromone": {f"{k[0]}-{k[1]}": v for k, v in self.pheromone.items()},
            "best_cost_history": self.best_cost_history,
            "config": {
                "n_ants": self.n_ants,
                "alpha": self.alpha,
                "beta": self.beta,
                "rho": self.rho,
                "q": self.q
            }
        }
    
    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Restore algorithm state"""
        self.iteration = state["iteration"]
        self.best_cost_history = state["best_cost_history"]
        
        # Restore best path
        best = state["best_path"]
        if best["nodes"]:
            self.best_path = Path(
                nodes=best["nodes"],
                cost=best["cost"],
                reliability=best["reliability"]
            )
        
        # Restore pheromones
        self.pheromone = {
            tuple(map(int, k.split('-'))): v 
            for k, v in state["pheromone"].items()
        }
        
        logger.info(f"ACO state loaded: iteration={self.iteration}, "
                   f"best_cost={self.best_path.cost if self.best_path else 'N/A'}")
