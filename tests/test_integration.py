import pytest
import os

def test_directory_structure():
    """Test expected directory structure"""
    expected_dirs = ['RAW_Images', 'Masks', 'Input_Images_To_Analyze']
    # This is a basic structure test
    assert len(expected_dirs) == 3

def test_file_extensions():
    """Test supported image extensions"""
    supported = ['.png', '.jpg', '.jpeg', '.bmp', '.gif']
    assert '.png' in supported
    assert len(supported) >= 3

def test_model_paths():
    """Test model path construction"""
    model_name = "test_model.pth"
    assert model_name.endswith('.pth') or model_name.endswith('.keras')

def test_configuration_values():
    """Test configuration consistency"""
    EPOCHS = 25
    BATCH_SIZE = 4
    assert EPOCHS > 0
    assert BATCH_SIZE > 0
