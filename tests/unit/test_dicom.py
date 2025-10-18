"""Unit tests for DICOM simulator"""
import pytest
from devices.dicom_sim.simulator import DICOMSimulator, DICOMOperation, ModalityType


def test_dicom_initialization():
    """Test DICOM node initializes correctly"""
    dicom = DICOMSimulator(device_id="test_dicom_001")
    
    assert dicom.state.device_id == "test_dicom_001"
    assert dicom.state.is_online is True
    assert dicom.state.storage_total_gb == 10000.0


def test_simulate_operation():
    """Test DICOM operation simulation"""
    dicom = DICOMSimulator(device_id="test_dicom_002")
    
    operation = dicom.simulate_operation()
    
    assert "operation" in operation
    assert "modality" in operation
    assert "duration_ms" in operation
    assert "data_size_mb" in operation
    assert dicom.state.operations_today == 1


def test_telemetry_generation():
    """Test telemetry data is generated correctly"""
    dicom = DICOMSimulator(device_id="test_dicom_003", ward="Radiology")
    
    telemetry = dicom.generate_telemetry()
    
    assert telemetry["device_id"] == "test_dicom_003"
    assert telemetry["device_type"] == "dicom_pacs"
    assert telemetry["ward"] == "Radiology"
    assert "storage" in telemetry
    assert "status" in telemetry


def test_storage_tracking():
    """Test storage usage is tracked"""
    dicom = DICOMSimulator(device_id="test_dicom_004")
    
    initial_storage = dicom.state.storage_used_gb
    
    # Simulate several store operations
    for _ in range(10):
        dicom.simulate_operation()
    
    # Storage might have increased if C_STORE operations occurred
    assert dicom.state.storage_used_gb >= initial_storage
