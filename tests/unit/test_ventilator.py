"""Unit tests for ventilator simulator"""
import pytest
from devices.vent_sim.simulator import VentilatorSimulator, VentilationMode


def test_ventilator_initialization():
    """Test ventilator initializes with correct defaults"""
    vent = VentilatorSimulator(device_id="test_vent_001")
    
    assert vent.state.device_id == "test_vent_001"
    assert vent.state.is_ventilating is True
    assert vent.settings.mode == VentilationMode.AC
    assert vent.settings.tidal_volume_ml == 500


def test_mode_change():
    """Test ventilation mode can be changed"""
    vent = VentilatorSimulator(device_id="test_vent_002")
    
    initial_mode = vent.settings.mode
    vent.set_ventilation_mode(VentilationMode.CPAP)
    
    assert vent.settings.mode == VentilationMode.CPAP
    assert vent.settings.mode != initial_mode


def test_respiratory_data_update():
    """Test respiratory data updates correctly"""
    vent = VentilatorSimulator(device_id="test_vent_003")
    
    initial_hours = vent.state.ventilator_hours
    
    vent.update_respiratory_data(delta_t=60.0)  # 1 minute
    
    # Ventilator hours should increase
    assert vent.state.ventilator_hours > initial_hours
    
    # Minute volume should be calculated
    assert vent.respiratory.minute_volume_l > 0


def test_minute_volume_calculation():
    """Test minute volume is calculated correctly"""
    vent = VentilatorSimulator(device_id="test_vent_004")
    
    vent.settings.tidal_volume_ml = 500
    vent.settings.respiratory_rate = 12
    
    vent.update_respiratory_data(delta_t=1.0)
    
    # Minute volume should be approximately 500 * 12 / 1000 = 6.0 L
    assert 5.0 < vent.respiratory.minute_volume_l < 7.0


def test_high_pressure_alarm():
    """Test alarm triggers for high pressure"""
    vent = VentilatorSimulator(device_id="test_vent_005")
    
    # Force high pressure
    vent.respiratory.peak_pressure_cmh2o = 40
    
    vent._check_alarms()
    
    assert vent.state.alarm_active is True
    assert vent.state.alarm_type == "high_pressure"


def test_low_minute_volume_alarm():
    """Test alarm triggers for low minute volume"""
    vent = VentilatorSimulator(device_id="test_vent_006")
    
    # Force low minute volume
    vent.respiratory.minute_volume_l = 3.0
    
    vent._check_alarms()
    
    assert vent.state.alarm_active is True
    assert vent.state.alarm_type == "low_minute_volume"


def test_telemetry_generation():
    """Test telemetry data is generated correctly"""
    vent = VentilatorSimulator(device_id="test_vent_007", ward="ICU")
    
    telemetry = vent.generate_telemetry()
    
    assert telemetry["device_id"] == "test_vent_007"
    assert telemetry["device_type"] == "ventilator"
    assert telemetry["ward"] == "ICU"
    assert "settings" in telemetry
    assert "measurements" in telemetry
    assert "status" in telemetry
    
    # Check settings structure
    settings = telemetry["settings"]
    assert "mode" in settings
    assert "tidal_volume_ml" in settings
    assert "fio2_percent" in settings
    
    # Check measurements structure
    measurements = telemetry["measurements"]
    assert "minute_volume_l" in measurements
    assert "peak_pressure_cmh2o" in measurements
    assert "spo2_percent" in measurements
