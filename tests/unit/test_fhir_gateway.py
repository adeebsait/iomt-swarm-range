"""Unit tests for FHIR gateway simulator"""
import pytest
from devices.fhir_gateway_sim.simulator import FHIRGatewaySimulator, FHIRResourceType


def test_fhir_gateway_initialization():
    """Test FHIR gateway initializes correctly"""
    gateway = FHIRGatewaySimulator(device_id="test_fhir_001")
    
    assert gateway.state.device_id == "test_fhir_001"
    assert gateway.state.is_online is True
    assert gateway.state.requests_today == 0


def test_create_fhir_observation():
    """Test FHIR Observation creation"""
    gateway = FHIRGatewaySimulator(device_id="test_fhir_002")
    
    observation = gateway.create_fhir_observation(
        patient_id="PATIENT_123",
        device_id="pump_001",
        code="8867-4",  # Heart rate
        value=75.0,
        unit="bpm"
    )
    
    assert observation["resourceType"] == "Observation"
    assert observation["status"] == "final"
    assert "subject" in observation
    assert observation["subject"]["reference"] == "Patient/PATIENT_123"
    assert observation["valueQuantity"]["value"] == 75.0
    assert gateway.state.observations_created == 1


def test_simulate_fhir_request():
    """Test FHIR request simulation"""
    gateway = FHIRGatewaySimulator(device_id="test_fhir_003")
    
    request = gateway.simulate_fhir_request()
    
    assert "method" in request
    assert "resource_type" in request
    assert "response_time_ms" in request
    assert "status_code" in request
    assert gateway.state.requests_today == 1


def test_telemetry_generation():
    """Test telemetry data is generated correctly"""
    gateway = FHIRGatewaySimulator(device_id="test_fhir_004", ward="Hospital")
    
    telemetry = gateway.generate_telemetry()
    
    assert telemetry["device_id"] == "test_fhir_004"
    assert telemetry["device_type"] == "fhir_gateway"
    assert telemetry["ward"] == "Hospital"
    assert "status" in telemetry
    assert "metrics" in telemetry


def test_error_tracking():
    """Test error rate tracking"""
    gateway = FHIRGatewaySimulator(device_id="test_fhir_005")
    
    # Simulate multiple requests
    for _ in range(10):
        gateway.simulate_fhir_request()
    
    # Should have tracked requests and possibly some errors
    assert gateway.state.requests_today == 10
    assert gateway.state.errors_today >= 0  # May or may not have errors
