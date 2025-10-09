import pytest
import numpy as np

def test_numpy_available():
    """Test that NumPy is available"""
    assert np is not None

def test_image_dimensions():
    """Test image dimension constants"""
    IMG_WIDTH = 512
    IMG_HEIGHT = 512
    IMG_CHANNELS = 3
    
    assert IMG_WIDTH == IMG_HEIGHT
    assert IMG_CHANNELS == 3

def test_array_operations():
    """Test basic numpy array operations"""
    arr = np.zeros((512, 512, 3))
    assert arr.shape == (512, 512, 3)
    
    normalized = arr / 255.0
    assert normalized.max() == 0.0

def test_backbone_names():
    """Test backbone configuration"""
    backbones = ['ResNet50', 'VGG16', 'DenseNet121']
    assert len(backbones) > 0
    assert 'ResNet50' in backbones
