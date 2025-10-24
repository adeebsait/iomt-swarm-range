"""
IoMT Cyber Range - Attack Simulation Framework
"""
from enum import Enum

class AttackType(Enum):
    """Types of attacks that can be simulated"""
    DOS = "denial_of_service"
    DATA_INJECTION = "data_injection"
    REPLAY = "replay_attack"
    MITM = "man_in_the_middle"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    PROTOCOL_VIOLATION = "protocol_violation"

class AttackSeverity(Enum):
    """Attack severity levels"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4
