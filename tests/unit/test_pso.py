"""Unit tests for PSO algorithm"""
import pytest
import numpy as np
from swarm.pso.algorithm import PSOAlgorithm, Particle


@pytest.fixture
def pso_config():
    """Basic PSO configuration"""
    return {
        "particles": 20,
        "omega": 0.72,
        "c1": 1.49,
        "c2": 1.49,
        "iters_per_window": 5,
        "resource_bounds": {
            "cpu": [0.5, 4.0],
            "mem": [1.0, 8.0],
            "bw": [10.0, 1000.0]
        }
    }


@pytest.fixture
def sample_metrics():
    """Sample agent metrics"""
    return {
        0: {
            "cpu_usage": 0.7,
            "mem_usage": 0.5,
            "queue_length": 15.0,
            "packet_rate": 120.0
        },
        1: {
            "cpu_usage": 0.6,
            "mem_usage": 0.4,
            "queue_length": 10.0,
            "packet_rate": 100.0
        }
    }


def test_pso_initialization(pso_config):
    """Test PSO initializes correctly"""
    pso = PSOAlgorithm(pso_config)
    pso.initialize_swarm(n_agents=3)
    
    assert len(pso.particles) == 20
    assert pso.n_agents == 3
    assert pso.dim == 9  # 3 agents * 3 resources
    assert pso.global_best_position is None  # Not evaluated yet


def test_pso_position_bounds(pso_config):
    """Test positions stay within bounds"""
    pso = PSOAlgorithm(pso_config)
    pso.initialize_swarm(n_agents=2)
    
    for particle in pso.particles:
        for i in range(pso.n_agents):
            idx = i * 3
            cpu = particle.position[idx]
            mem = particle.position[idx + 1]
            bw = particle.position[idx + 2]
            
            assert pso.cpu_bounds[0] <= cpu <= pso.cpu_bounds[1]
            assert pso.mem_bounds[0] <= mem <= pso.mem_bounds[1]
            assert pso.bw_bounds[0] <= bw <= pso.bw_bounds[1]


def test_pso_fitness_evaluation(pso_config, sample_metrics):
    """Test fitness function computes correctly"""
    pso = PSOAlgorithm(pso_config)
    pso.initialize_swarm(n_agents=2)
    
    position = pso.particles[0].position
    fitness = pso._evaluate_fitness(position, sample_metrics)
    
    assert isinstance(fitness, float)
    assert fitness >= 0  # Should be non-negative


def test_pso_optimization_reduces_fitness(pso_config, sample_metrics):
    """Test PSO reduces fitness over iterations"""
    pso = PSOAlgorithm(pso_config)
    pso.initialize_swarm(n_agents=2)
    
    # Initial evaluation
    observation = {"agent_metrics": sample_metrics}
    result1 = pso.step(observation)
    initial_fitness = result1["fitness"]
    
    # Run several iterations
    for _ in range(10):
        result = pso.step(observation)
    
    final_fitness = result["fitness"]
    
    # Fitness should decrease (better optimization)
    assert final_fitness <= initial_fitness
    # After 11 iterations total, should be converged (max_iters is 5)
    assert result["iteration"] == 11
    assert result["converged"] is True


def test_pso_allocation_parsing(pso_config, sample_metrics):
    """Test allocation dictionary is correctly formatted"""
    pso = PSOAlgorithm(pso_config)
    pso.initialize_swarm(n_agents=3)
    
    observation = {"agent_metrics": sample_metrics}
    result = pso.step(observation)
    
    allocation = result["resource_allocation"]
    
    assert len(allocation) == 3  # 3 agents
    for agent_id, resources in allocation.items():
        assert "cpu_cores" in resources
        assert "memory_gb" in resources
        assert "bandwidth_mbps" in resources
        
        # Check bounds
        assert pso.cpu_bounds[0] <= resources["cpu_cores"] <= pso.cpu_bounds[1]
        assert pso.mem_bounds[0] <= resources["memory_gb"] <= pso.mem_bounds[1]
        assert pso.bw_bounds[0] <= resources["bandwidth_mbps"] <= pso.bw_bounds[1]


def test_pso_velocity_update(pso_config):
    """Test velocity updates follow PSO equations"""
    pso = PSOAlgorithm(pso_config)
    pso.initialize_swarm(n_agents=2)
    
    particle = pso.particles[0]
    old_velocity = particle.velocity.copy()
    old_position = particle.position.copy()
    
    # Run one step
    observation = {"agent_metrics": {}}
    pso.step(observation)
    
    # Velocity should have changed
    assert not np.allclose(particle.velocity, old_velocity)
    
    # Position should have changed
    assert not np.allclose(particle.position, old_position)


def test_pso_state_serialization(pso_config, sample_metrics):
    """Test state save/load"""
    pso = PSOAlgorithm(pso_config)
    pso.initialize_swarm(n_agents=2)
    
    # Run a few iterations
    observation = {"agent_metrics": sample_metrics}
    for _ in range(5):
        pso.step(observation)
    
    # Save state
    state = pso.state_dict()
    
    # Create new instance and load
    pso2 = PSOAlgorithm(pso_config)
    pso2.load_state_dict(state)
    
    assert pso2.iteration == pso.iteration
    assert pso2.global_best_fitness == pso.global_best_fitness
    assert np.allclose(pso2.global_best_position, pso.global_best_position)


def test_pso_convergence_flag(pso_config, sample_metrics):
    """Test convergence flag is set correctly"""
    pso_config["iters_per_window"] = 3
    pso = PSOAlgorithm(pso_config)
    pso.initialize_swarm(n_agents=2)
    
    observation = {"agent_metrics": sample_metrics}
    
    # Run iterations and check convergence
    for i in range(5):
        result = pso.step(observation)
        # Converged when iteration >= max_iters (3)
        # iteration starts at 0, increments after step
        # So: iter 1,2 -> not converged; iter 3,4,5 -> converged
        if i < 2:
            assert result["converged"] is False, f"Should not be converged at iteration {i+1}"
        else:
            assert result["converged"] is True, f"Should be converged at iteration {i+1}"
