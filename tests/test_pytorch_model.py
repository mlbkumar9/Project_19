import pytest
import torch
import numpy as np

def test_torch_available():
    """Test that PyTorch is available"""
    assert torch is not None
    
def test_device_detection():
    """Test CUDA/CPU device detection"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert device.type in ['cuda', 'cpu']


def test_tensor_creation():
    """Test basic tensor operations"""
    tensor = torch.zeros((1, 3, 512, 512))
    assert tensor.shape == (1, 3, 512, 512)
    assert tensor.dtype == torch.float32

def test_model_config():
    """Test model configuration parameters"""
    IMG_SIZE = 512
    BATCH_SIZE = 4
    LEARNING_RATE = 1e-4
    
    assert IMG_SIZE == 512
    assert BATCH_SIZE > 0
    assert LEARNING_RATE > 0
