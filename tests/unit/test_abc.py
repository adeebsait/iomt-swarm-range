"""Unit tests for ABC algorithm"""
import pytest
import numpy as np
from swarm.abc.algorithm import ABCAlgorithm, SensorPlacement


@pytest.fixture
def abc_config():
    """Basic ABC configuration"""
    return {
        "alpha": 1.0,
        "beta": 0.3,
        "gamma": 0.1,
        "iters": 10,
        "stagnation_limit": 5,
        "n_bees": 20
    }


@pytest.fixture
def simple_topology():
    """Simple linear topology: 0-1-2-3-4"""
    return {
        0: [1],
        1: [0, 2],
        2: [1, 3],
        3: [2, 4],
        4: [3]
    }


def test_abc_initialization(abc_config, simple_topology):
    """Test ABC initializes correctly"""
    abc = ABCAlgorithm(abc_config)
    abc.initialize_population(
        n_positions=5,
        max_sensors=3,
        topology_graph=simple_topology
    )
    
    assert len(abc.population) == 20
    assert abc.best_solution is not None
    assert abc.best_solution.fitness >= 0
    assert len(abc.best_solution.positions) <= 3


def test_abc_fitness_evaluation(abc_config, simple_topology):
    """Test fitness function computes correctly"""
    abc = ABCAlgorithm(abc_config)
    abc.topology = simple_topology
    abc.n_positions = 5
    abc.max_sensors = 3
    
    # Perfect coverage with 3 sensors at positions 0, 2, 4
    placement = SensorPlacement(positions=[0, 2, 4])
    fitness = abc._evaluate_fitness(placement)
    
    assert fitness > 0
    assert fitness <= 1.0  # Should be normalized


def test_abc_convergence(abc_config, simple_topology):
    """Test ABC improves fitness over iterations"""
    abc = ABCAlgorithm(abc_config)
    abc.initialize_population(
        n_positions=5,
        max_sensors=3,
        topology_graph=simple_topology
    )
    
    initial_fitness = abc.best_solution.fitness
    
    # Run 10 iterations
    for _ in range(10):
        result = abc.step({})
    
    final_fitness = result["fitness"]
    
    # Fitness should improve or stay same
    assert final_fitness >= initial_fitness
    assert result["converged"]
    assert result["coverage"] > 0


def test_abc_neighbor_generation(abc_config, simple_topology):
    """Test neighbor generation creates valid solutions"""
    abc = ABCAlgorithm(abc_config)
    abc.n_positions = 5
    abc.max_sensors = 3
    abc.topology = simple_topology
    
    original = SensorPlacement(positions=[0, 2])
    
    # Generate 10 neighbors
    for _ in range(10):
        neighbor = abc._generate_neighbor(original)
        
        # Check validity
        assert len(neighbor.positions) >= 1
        assert len(neighbor.positions) <= abc.max_sensors
        assert all(0 <= pos < abc.n_positions for pos in neighbor.positions)


def test_abc_state_serialization(abc_config, simple_topology):
    """Test state save/load"""
    abc = ABCAlgorithm(abc_config)
    abc.initialize_population(
        n_positions=5,
        max_sensors=3,
        topology_graph=simple_topology
    )
    
    # Run a few iterations
    for _ in range(5):
        abc.step({})
    
    # Save state
    state = abc.state_dict()
    
    # Create new instance and load
    abc2 = ABCAlgorithm(abc_config)
    abc2.load_state_dict(state)
    
    assert abc2.iteration == abc.iteration
    assert abc2.best_solution.fitness == abc.best_solution.fitness
    assert abc2.best_solution.positions == abc.best_solution.positions


def test_abc_scout_bee_phase(abc_config, simple_topology):
    """Test scout bees abandon stagnant solutions"""
    abc = ABCAlgorithm(abc_config)
    abc.initialize_population(
        n_positions=5,
        max_sensors=3,
        topology_graph=simple_topology
    )
    
    # Force stagnation
    abc.population[0].stagnation_count = abc.stagnation_limit
    original_fitness = abc.population[0].fitness
    
    # Run scout phase
    abc._scout_bee_phase()
    
    # Position should have changed (new random solution)
    assert abc.population[0].stagnation_count == 0
