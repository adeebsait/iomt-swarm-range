"""
Attack Scenario Runner

Orchestrates complex attack scenarios with timing and coordination
"""
import time
import logging
from datetime import datetime
from typing import List, Dict
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AttackScenario:
    """Defines an attack scenario with timing"""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.timeline = []
        self.results = []
    
    def add_event(self, time_offset: int, attack_type: str, params: Dict):
        """
        Add attack event to timeline
        
        Args:
            time_offset: Seconds from scenario start
            attack_type: Type of attack
            params: Attack parameters
        """
        self.timeline.append({
            'time_offset': time_offset,
            'attack_type': attack_type,
            'params': params
        })
    
    def run(self):
        """Execute the scenario"""
        logger.info(f"Starting scenario: {self.name}")
        logger.info(f"Description: {self.description}")
        
        start_time = time.time()
        
        for event in sorted(self.timeline, key=lambda x: x['time_offset']):
            # Wait until event time
            elapsed = time.time() - start_time
            wait_time = event['time_offset'] - elapsed
            
            if wait_time > 0:
                logger.info(f"Waiting {wait_time:.1f}s for next event...")
                time.sleep(wait_time)
            
            # Execute attack
            logger.info(f"Executing: {event['attack_type']}")
            self.results.append({
                'timestamp': datetime.now().isoformat(),
                'event': event
            })
        
        logger.info(f"Scenario '{self.name}' completed")
        return self.results


# Define research scenarios
def create_baseline_scenario() -> AttackScenario:
    """Normal operation baseline (no attacks)"""
    scenario = AttackScenario(
        name="Baseline - Normal Operation",
        description="5 minutes of normal device operation with no attacks"
    )
    return scenario


def create_dos_scenario() -> AttackScenario:
    """DoS attack scenario"""
    scenario = AttackScenario(
        name="DoS Attack Scenario",
        description="Message flooding attack on MQTT broker"
    )
    
    # Normal operation for 60s
    # DoS attack for 60s
    scenario.add_event(60, "dos_message_flood", {"rate": 1000, "duration": 60})
    # Recovery period 60s
    
    return scenario


def create_data_injection_scenario() -> AttackScenario:
    """Data injection attack scenario"""
    scenario = AttackScenario(
        name="Data Injection Attack",
        description="False medical data injection"
    )
    
    # Normal 30s
    # Inject critical values for 60s
    scenario.add_event(30, "data_injection_critical", {"duration": 60})
    # Recovery 30s
    
    return scenario


def create_multi_vector_scenario() -> AttackScenario:
    """Complex multi-vector attack"""
    scenario = AttackScenario(
        name="Multi-Vector Attack",
        description="Coordinated DoS + Data Injection attack"
    )
    
    # Normal 30s
    # Start DoS
    scenario.add_event(30, "dos_message_flood", {"rate": 500, "duration": 90})
    # Add data injection during DoS
    scenario.add_event(60, "data_injection_critical", {"duration": 60})
    # Recovery 30s
    
    return scenario


if __name__ == "__main__":
    # Test scenario runner
    scenario = create_dos_scenario()
    print(f"\nScenario: {scenario.name}")
    print(f"Events: {len(scenario.timeline)}")
    print("\nTimeline:")
    for event in scenario.timeline:
        print(f"  T+{event['time_offset']}s: {event['attack_type']}")
