"""Unit tests for ACO algorithm"""
import pytest
import numpy as np
from swarm.aco.algorithm import ACOAlgorithm, Path


@pytest.fixture
def aco_config():
    """Basic ACO configuration"""
    return {
        "n_ants": 10,
        "alpha": 1.0,
        "beta": 2.0,
        "rho": 0.3,
        "q": 100,
        "pheromone_init": 0.1
    }


@pytest.fixture
def simple_graph():
    """Simple linear graph: 0-1-2-3-4"""
    return {
        0: [1],
        1: [0, 2],
        2: [1, 3],
        3: [2, 4],
        4: [3]
    }


@pytest.fixture
def edge_costs():
    """Edge costs for simple graph"""
    return {
        (0, 1): 1.0, (1, 0): 1.0,
        (1, 2): 1.0, (2, 1): 1.0,
        (2, 3): 1.0, (3, 2): 1.0,
        (3, 4): 1.0, (4, 3): 1.0
    }


@pytest.fixture
def complex_graph():
    """Graph with multiple paths"""
    return {
        0: [1, 2],
        1: [0, 2, 3],
        2: [0, 1, 3, 4],
        3: [1, 2, 4],
        4: [2, 3]
    }


@pytest.fixture
def complex_edge_costs():
    """Edge costs for complex graph"""
    costs = {}
    # Symmetric edges with varying costs
    edges = [
        ((0, 1), 2.0), ((1, 0), 2.0),
        ((0, 2), 5.0), ((2, 0), 5.0),
        ((1, 2), 1.0), ((2, 1), 1.0),
        ((1, 3), 3.0), ((3, 1), 3.0),
        ((2, 3), 2.0), ((3, 2), 2.0),
        ((2, 4), 4.0), ((4, 2), 4.0),
        ((3, 4), 1.0), ((4, 3), 1.0)
    ]
    for edge, cost in edges:
        costs[edge] = cost
    return costs


def test_aco_initialization(aco_config, simple_graph, edge_costs):
    """Test ACO initializes correctly"""
    aco = ACOAlgorithm(aco_config)
    aco.initialize(simple_graph, edge_costs)
    
    assert len(aco.pheromone) == 8  # 4 edges * 2 directions
    assert all(v == 0.1 for v in aco.pheromone.values())
    assert len(aco.heuristic) == 8


def test_aco_path_construction(aco_config, simple_graph, edge_costs):
    """Test path construction finds valid paths"""
    aco = ACOAlgorithm(aco_config)
    aco.initialize(simple_graph, edge_costs)
    
    path = aco.construct_solution(source=0, destination=4, max_hops=10)
    
    assert path.nodes[0] == 0
    assert path.nodes[-1] == 4
    assert path.cost < float('inf')
    assert len(path.nodes) == 5  # Shortest path is 0-1-2-3-4


def test_aco_finds_shortest_path(aco_config, complex_graph, complex_edge_costs):
    """Test ACO finds good paths in complex graph"""
    aco = ACOAlgorithm(aco_config)
    aco.initialize(complex_graph, complex_edge_costs)
    
    # Run multiple iterations
    observation = {"source": 0, "destination": 4, "max_hops": 10}
    
    for _ in range(20):
        result = aco.step(observation)
    
    # Should find a valid path
    assert result["best_path"]
    assert result["best_cost"] < float('inf')
    assert result["best_reliability"] > 0
    
    # Path should start at 0 and end at 4
    assert result["best_path"][0] == 0
    assert result["best_path"][-1] == 4


def test_aco_pheromone_update(aco_config, simple_graph, edge_costs):
    """Test pheromone levels update correctly"""
    aco = ACOAlgorithm(aco_config)
    aco.initialize(simple_graph, edge_costs)
    
    # Initial pheromone
    initial_pheromone = dict(aco.pheromone)
    
    # Run one iteration
    observation = {"source": 0, "destination": 4}
    aco.step(observation)
    
    # Pheromone levels should have changed
    changed = False
    for edge in aco.pheromone:
        if abs(aco.pheromone[edge] - initial_pheromone[edge]) > 0.01:
            changed = True
            break
    
    assert changed, "Pheromone levels should change after iteration"


def test_aco_pheromone_evaporation(aco_config, simple_graph, edge_costs):
    """Test pheromone evaporation works"""
    aco = ACOAlgorithm(aco_config)
    aco.initialize(simple_graph, edge_costs)
    
    # Set high pheromone on one edge
    test_edge = (0, 1)
    aco.pheromone[test_edge] = 10.0
    
    # Create a path that doesn't use this edge
    paths = [Path(nodes=[2, 3, 4], cost=2.0, reliability=0.9)]
    
    # Update pheromones (should evaporate the unused edge)
    aco.update_pheromones(paths)
    
    # Pheromone should be less than 10.0 due to evaporation
    assert aco.pheromone[test_edge] < 10.0


def test_aco_convergence(aco_config, simple_graph, edge_costs):
    """Test ACO converges to good solution"""
    aco = ACOAlgorithm(aco_config)
    aco.initialize(simple_graph, edge_costs)
    
    observation = {"source": 0, "destination": 4}
    
    # Run iterations and track best cost
    costs = []
    for _ in range(15):
        result = aco.step(observation)
        costs.append(result["best_cost"])
    
    # Cost should stabilize (converge)
    final_costs = costs[-5:]
    assert max(final_costs) - min(final_costs) < 1.0  # Low variance


def test_aco_routing_table(aco_config, complex_graph, complex_edge_costs):
    """Test routing table generation"""
    aco = ACOAlgorithm(aco_config)
    aco.initialize(complex_graph, complex_edge_costs)
    
    # Run a few iterations to build pheromones
    observation = {"source": 0, "destination": 4}
    for _ in range(10):
        aco.step(observation)
    
    # Generate routing table from all nodes to node 4
    sources = [0, 1, 2, 3]
    routing_table = aco.get_routing_table(sources, destination=4)
    
    assert len(routing_table) == 4
    for source in sources:
        if routing_table[source]:  # If path exists
            assert routing_table[source][0] == source
            assert routing_table[source][-1] == 4


def test_aco_state_serialization(aco_config, simple_graph, edge_costs):
    """Test state save/load"""
    aco = ACOAlgorithm(aco_config)
    aco.initialize(simple_graph, edge_costs)
    
    # Run a few iterations
    observation = {"source": 0, "destination": 4}
    for _ in range(5):
        aco.step(observation)
    
    # Save state
    state = aco.state_dict()
    
    # Create new instance and load
    aco2 = ACOAlgorithm(aco_config)
    aco2.graph = simple_graph
    aco2.heuristic = aco.heuristic
    aco2.load_state_dict(state)
    
    assert aco2.iteration == aco.iteration
    assert aco2.best_path.cost == aco.best_path.cost
    assert aco2.best_path.nodes == aco.best_path.nodes


def test_aco_invalid_path_handling(aco_config):
    """Test ACO handles disconnected graphs gracefully"""
    # Disconnected graph
    graph = {
        0: [1],
        1: [0],
        2: [3],
        3: [2]
    }
    costs = {
        (0, 1): 1.0, (1, 0): 1.0,
        (2, 3): 1.0, (3, 2): 1.0
    }
    
    aco = ACOAlgorithm(aco_config)
    aco.initialize(graph, costs)
    
    # Try to find path between disconnected components
    path = aco.construct_solution(source=0, destination=3, max_hops=10)
    
    # Should return invalid path
    assert path.cost == float('inf')
