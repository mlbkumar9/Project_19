"""
Tests for damage_analyzer.py
"""
import pytest
import numpy as np
import cv2
import os
import tempfile
from damage_analyzer import analyze_damage_area


class TestAnalyzeDamageArea:
    """Tests for the analyze_damage_area function"""
    
    def test_analyze_damage_area_with_white_pixels(self):
        """Test that damage area is correctly calculated for an image with white pixels"""
        # Create a temporary image with known white pixel count
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            # Create a 100x100 image with a 50x50 white square in the center
            image = np.zeros((100, 100, 3), dtype=np.uint8)
            image[25:75, 25:75] = [255, 255, 255]
            cv2.imwrite(tmp.name, image)
            
            damage_area, damage_mask, original_image = analyze_damage_area(tmp.name)
            
            # Clean up
            os.unlink(tmp.name)
            
            # Verify results
            assert damage_area is not None
            assert damage_mask is not None
            assert original_image is not None
            assert damage_area == 2500  # 50x50 = 2500 white pixels
            assert damage_mask.shape == (100, 100)
    
    def test_analyze_damage_area_no_damage(self):
        """Test that damage area is zero for an image with no white pixels"""
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            # Create a 100x100 black image
            image = np.zeros((100, 100, 3), dtype=np.uint8)
            cv2.imwrite(tmp.name, image)
            
            damage_area, damage_mask, original_image = analyze_damage_area(tmp.name)
            
            # Clean up
            os.unlink(tmp.name)
            
            # Verify results
            assert damage_area == 0
            assert damage_mask is not None
            assert original_image is not None
    
    def test_analyze_damage_area_invalid_path(self):
        """Test that function handles invalid file paths gracefully"""
        damage_area, damage_mask, original_image = analyze_damage_area('/nonexistent/path/image.png')
        
        assert damage_area is None
        assert damage_mask is None
        assert original_image is None
    
    def test_analyze_damage_area_threshold_boundary(self):
        """Test damage detection at threshold boundary (240)"""
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            # Create image with pixels at threshold boundary
            image = np.zeros((100, 100, 3), dtype=np.uint8)
            # Pixels with value 240 should NOT be detected
            image[0:25, 0:25] = [240, 240, 240]
            # Pixels with value 241 should be detected
            image[25:50, 25:50] = [241, 241, 241]
            cv2.imwrite(tmp.name, image)
            
            damage_area, damage_mask, original_image = analyze_damage_area(tmp.name)
            
            # Clean up
            os.unlink(tmp.name)
            
            # Verify that only pixels > 240 are counted
            assert damage_area == 625  # 25x25 = 625 pixels with value 241
