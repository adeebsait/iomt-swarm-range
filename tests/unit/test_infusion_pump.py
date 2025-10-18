"""Unit tests for infusion pump simulator"""
import pytest
import time
from devices.infusion_pump_sim.simulator import InfusionPumpSimulator, PumpState


def test_pump_initialization():
    """Test pump initializes with correct defaults"""
    pump = InfusionPumpSimulator(device_id="test_pump_001")
    
    assert pump.state.device_id == "test_pump_001"
    assert pump.state.is_running is False
    assert pump.state.battery_percent == 100.0
    assert pump.state.infusion_rate_ml_hr == 0.0


def test_start_infusion():
    """Test starting an infusion"""
    pump = InfusionPumpSimulator(device_id="test_pump_002")
    
    pump.start_infusion(
        rate_ml_hr=100.0,
        volume_ml=500.0,
        medication="Normal Saline",
        patient_id="PATIENT_123"
    )
    
    assert pump.state.is_running is True
    assert pump.state.infusion_rate_ml_hr == 100.0
    assert pump.state.volume_to_infuse_ml == 500.0
    assert pump.state.medication_name == "Normal Saline"
    assert pump.state.patient_id == "PATIENT_123"


def test_stop_infusion():
    """Test stopping an infusion"""
    pump = InfusionPumpSimulator(device_id="test_pump_003")
    
    pump.start_infusion(rate_ml_hr=100.0, volume_ml=500.0)
    assert pump.state.is_running is True
    
    pump.stop_infusion()
    assert pump.state.is_running is False


def test_state_update():
    """Test state updates correctly over time"""
    pump = InfusionPumpSimulator(device_id="test_pump_004")
    
    # Use a large volume so it doesn't complete during test
    pump.start_infusion(rate_ml_hr=100.0, volume_ml=500.0)
    
    # Simulate 30 minutes (1800 seconds) at 100 mL/hr = 50 mL
    pump.update_state(delta_t=1800.0)
    
    # Should still be running (not complete)
    assert pump.state.is_running is True
    
    # Should have infused approximately 50 mL
    assert 45.0 < pump.state.volume_infused_ml < 55.0
    
    # Battery should have drained (1800 seconds * 0.002% per second = 3.6%)
    assert pump.state.battery_percent < 97.0


def test_infusion_completion():
    """Test infusion stops when volume is reached"""
    pump = InfusionPumpSimulator(device_id="test_pump_005")
    
    pump.start_infusion(rate_ml_hr=100.0, volume_ml=10.0)  # Small volume
    
    # Simulate time to complete (10 mL at 100 mL/hr = 6 minutes = 360 seconds)
    pump.update_state(delta_t=400.0)  # More than enough time
    
    # Pump should have stopped
    assert pump.state.is_running is False
    assert pump.state.volume_infused_ml == pytest.approx(10.0, abs=0.1)


def test_telemetry_generation():
    """Test telemetry data is generated correctly"""
    pump = InfusionPumpSimulator(device_id="test_pump_006", ward="ICU")
    
    pump.start_infusion(rate_ml_hr=150.0, volume_ml=300.0)
    
    telemetry = pump.generate_telemetry()
    
    assert telemetry["device_id"] == "test_pump_006"
    assert telemetry["device_type"] == "infusion_pump"
    assert telemetry["ward"] == "ICU"
    assert telemetry["status"] == "running"
    assert telemetry["infusion_rate_ml_hr"] == 150.0
    assert "timestamp" in telemetry
    assert "battery_percent" in telemetry
    assert "pressure_psi" in telemetry


def test_battery_drain():
    """Test battery drains during operation"""
    pump = InfusionPumpSimulator(device_id="test_pump_007")
    
    initial_battery = pump.state.battery_percent
    
    # Large volume to ensure pump keeps running
    pump.start_infusion(rate_ml_hr=100.0, volume_ml=1000.0)
    pump.update_state(delta_t=1000.0)  # 1000 seconds
    
    # Battery should have drained (1000 * 0.002 = 2%)
    assert pump.state.battery_percent < initial_battery - 1.5
    assert pump.state.is_running is True  # Should still be running
