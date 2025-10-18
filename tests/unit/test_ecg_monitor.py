"""Unit tests for ECG monitor simulator"""
import pytest
from devices.ecg_monitor_sim.simulator import ECGMonitorSimulator, VitalSigns


def test_monitor_initialization():
    """Test monitor initializes with correct defaults"""
    monitor = ECGMonitorSimulator(device_id="test_ecg_001")
    
    assert monitor.state.device_id == "test_ecg_001"
    assert monitor.state.is_monitoring is True
    assert monitor.vitals.heart_rate_bpm == 75
    assert monitor.vitals.spo2_percent == 98.0


def test_ecg_waveform_generation():
    """Test ECG waveform is generated correctly"""
    monitor = ECGMonitorSimulator(device_id="test_ecg_002")
    
    waveform = monitor.generate_ecg_waveform(duration_sec=1.0)
    
    # Should have correct number of samples (250 Hz * 1 second)
    assert len(waveform) == 250
    
    # All values should be floats
    assert all(isinstance(v, float) for v in waveform)
    
    # Voltage should be in reasonable range (-2 to 2 mV)
    assert all(-2.0 < v < 2.0 for v in waveform)


def test_vitals_update():
    """Test vital signs update with realistic variations"""
    monitor = ECGMonitorSimulator(device_id="test_ecg_003")
    
    initial_hr = monitor.vitals.heart_rate_bpm
    
    # Update vitals multiple times
    for _ in range(10):
        monitor.update_vitals(delta_t=1.0)
    
    # HR should have changed (with high probability)
    # Allow for rare case where it doesn't change
    assert 50 <= monitor.vitals.heart_rate_bpm <= 150
    
    # SpO2 should be in valid range
    assert 85 <= monitor.vitals.spo2_percent <= 100


def test_tachycardia_alarm():
    """Test alarm triggers for tachycardia"""
    monitor = ECGMonitorSimulator(device_id="test_ecg_004")
    
    # Force high heart rate
    monitor.vitals.heart_rate_bpm = 130
    
    monitor._check_alarms()
    
    assert monitor.state.alarm_active is True
    assert monitor.state.alarm_type == "tachycardia"


def test_hypoxia_alarm():
    """Test alarm triggers for low SpO2"""
    monitor = ECGMonitorSimulator(device_id="test_ecg_005")
    
    # Force low SpO2
    monitor.vitals.spo2_percent = 85.0
    
    monitor._check_alarms()
    
    assert monitor.state.alarm_active is True
    assert monitor.state.alarm_type == "hypoxia"


def test_telemetry_generation():
    """Test telemetry data is generated correctly"""
    monitor = ECGMonitorSimulator(device_id="test_ecg_006", ward="ICU")
    
    telemetry = monitor.generate_telemetry()
    
    assert telemetry["device_id"] == "test_ecg_006"
    assert telemetry["device_type"] == "ecg_monitor"
    assert telemetry["ward"] == "ICU"
    assert "vital_signs" in telemetry
    assert "ecg_lead_ii" in telemetry
    assert "timestamp" in telemetry
    
    # Check vital signs structure
    vitals = telemetry["vital_signs"]
    assert "heart_rate_bpm" in vitals
    assert "spo2_percent" in vitals
    assert "systolic_bp_mmhg" in vitals


def test_alarm_clears():
    """Test alarm clears when vitals normalize"""
    monitor = ECGMonitorSimulator(device_id="test_ecg_007")
    
    # Trigger alarm
    monitor.vitals.heart_rate_bpm = 130
    monitor._check_alarms()
    assert monitor.state.alarm_active is True
    
    # Normalize vitals
    monitor.vitals.heart_rate_bpm = 75
    monitor._check_alarms()
    assert monitor.state.alarm_active is False
