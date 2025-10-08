"""
Tests for predict_pytorch.py
"""
import pytest
import torch
from predict_pytorch import get_device


class TestGetDevice:
    """Tests for the get_device function"""
    
    def test_get_device_returns_valid_device(self):
        """Test that get_device returns a valid torch device"""
        device = get_device()
        
        assert isinstance(device, torch.device)
        assert device.type in ['cuda', 'cpu']
    
    def test_get_device_returns_cpu_when_no_cuda(self):
        """Test that get_device returns CPU when CUDA is not available"""
        device = get_device()
        
        # On most CI systems, CUDA won't be available
        # This test verifies the function works in that scenario
        if not torch.cuda.is_available():
            assert device.type == 'cpu'
