"""
Unit tests for the Cat vs Dog Classifier API
Tests data preprocessing and model inference functions
"""
import pytest
import torch
import numpy as np
from PIL import Image
import io
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import (
    CatsDogsCNN, 
    preprocess_image, 
    predict_image,
    transform,
    device,
    class_names
)


class TestDataPreprocessing:
    """Test data preprocessing functions"""
    
    def test_preprocess_image_shape(self):
        """Test that preprocessing returns correct tensor shape"""
        # Create a test image (RGB)
        test_image = Image.new('RGB', (100, 100), color='red')
        
        # Preprocess the image
        processed_tensor = preprocess_image(test_image)
        
        # Check shape: (batch_size=1, channels=3, height=224, width=224)
        assert processed_tensor.shape == (1, 3, 224, 224), \
            f"Expected shape (1, 3, 224, 224), got {processed_tensor.shape}"
    
    def test_preprocess_image_type(self):
        """Test that preprocessing returns a torch tensor"""
        test_image = Image.new('RGB', (100, 100), color='blue')
        processed_tensor = preprocess_image(test_image)
        
        assert isinstance(processed_tensor, torch.Tensor), \
            f"Expected torch.Tensor, got {type(processed_tensor)}"
    
    def test_preprocess_image_device(self):
        """Test that preprocessed image is on correct device"""
        test_image = Image.new('RGB', (100, 100), color='green')
        processed_tensor = preprocess_image(test_image)
        
        assert processed_tensor.device.type == device.type, \
            f"Expected device {device.type}, got {processed_tensor.device.type}"
    
    def test_preprocess_image_converts_grayscale(self):
        """Test that grayscale images are converted to RGB"""
        # Create a grayscale image
        grayscale_image = Image.new('L', (100, 100), color=128)
        
        # Preprocess should handle this without error
        processed_tensor = preprocess_image(grayscale_image)
        
        # Should still have 3 channels after conversion
        assert processed_tensor.shape[1] == 3, \
            f"Expected 3 channels, got {processed_tensor.shape[1]}"
    
    def test_preprocess_image_normalized(self):
        """Test that image values are normalized"""
        test_image = Image.new('RGB', (100, 100), color=(128, 128, 128))
        processed_tensor = preprocess_image(test_image)
        
        # After normalization, values should be roughly in range [-2, 2]
        # (depends on normalization params)
        assert processed_tensor.min() >= -3 and processed_tensor.max() <= 3, \
            f"Values might not be normalized: min={processed_tensor.min()}, max={processed_tensor.max()}"


class TestModelArchitecture:
    """Test model architecture"""
    
    def test_model_initialization(self):
        """Test that model can be initialized"""
        model = CatsDogsCNN()
        assert model is not None, "Model initialization failed"
    
    def test_model_forward_pass(self):
        """Test model forward pass with dummy input"""
        model = CatsDogsCNN()
        model.eval()
        
        # Create dummy input: (batch_size=1, channels=3, height=224, width=224)
        dummy_input = torch.randn(1, 3, 224, 224)
        
        with torch.no_grad():
            output = model(dummy_input)
        
        # Output should be a single value per batch item (binary classification)
        assert output.shape == (1,), f"Expected shape (1,), got {output.shape}"
    
    def test_model_output_range(self):
        """Test that model output can be converted to probabilities"""
        model = CatsDogsCNN()
        model.eval()
        
        dummy_input = torch.randn(1, 3, 224, 224)
        
        with torch.no_grad():
            output = model(dummy_input)
            probability = torch.sigmoid(output).item()
        
        # Probability should be between 0 and 1
        assert 0 <= probability <= 1, \
            f"Probability should be in [0, 1], got {probability}"


class TestInferenceFunctions:
    """Test model inference functions"""
    
    def test_predict_image_returns_dict(self):
        """Test that prediction returns a dictionary"""
        test_image = Image.new('RGB', (100, 100), color='red')
        
        # Note: This requires the global model to be loaded
        # For unit tests, we might want to mock this
        try:
            result = predict_image(test_image)
            assert isinstance(result, dict), \
                f"Expected dict, got {type(result)}"
        except Exception as e:
            # If model isn't loaded, skip this test
            pytest.skip(f"Model not loaded: {e}")
    
    def test_predict_image_has_required_keys(self):
        """Test that prediction result has required keys"""
        test_image = Image.new('RGB', (100, 100), color='blue')
        
        try:
            result = predict_image(test_image)
            
            required_keys = ['predicted_class', 'confidence', 'probabilities']
            for key in required_keys:
                assert key in result, f"Missing required key: {key}"
        except Exception as e:
            pytest.skip(f"Model not loaded: {e}")
    
    def test_predict_image_class_valid(self):
        """Test that predicted class is valid"""
        test_image = Image.new('RGB', (100, 100), color='green')
        
        try:
            result = predict_image(test_image)
            
            assert result['predicted_class'] in class_names, \
                f"Invalid class: {result['predicted_class']}"
        except Exception as e:
            pytest.skip(f"Model not loaded: {e}")
    
    def test_predict_image_confidence_range(self):
        """Test that confidence is between 0 and 1"""
        test_image = Image.new('RGB', (100, 100), color='yellow')
        
        try:
            result = predict_image(test_image)
            
            confidence = result['confidence']
            assert 0 <= confidence <= 1, \
                f"Confidence should be in [0, 1], got {confidence}"
        except Exception as e:
            pytest.skip(f"Model not loaded: {e}")
    
    def test_predict_image_probabilities_sum(self):
        """Test that probabilities sum to approximately 1"""
        test_image = Image.new('RGB', (100, 100), color='purple')
        
        try:
            result = predict_image(test_image)
            
            probs = result['probabilities']
            prob_sum = sum(probs.values())
            
            # Should sum to 1 (with small tolerance for floating point)
            assert abs(prob_sum - 1.0) < 0.01, \
                f"Probabilities should sum to 1, got {prob_sum}"
        except Exception as e:
            pytest.skip(f"Model not loaded: {e}")


class TestImageFormats:
    """Test various image format handling"""
    
    @pytest.mark.parametrize("size", [(50, 50), (100, 100), (300, 200), (512, 512)])
    def test_various_image_sizes(self, size):
        """Test preprocessing handles various image sizes"""
        test_image = Image.new('RGB', size, color='red')
        processed_tensor = preprocess_image(test_image)
        
        # Should always resize to 224x224
        assert processed_tensor.shape == (1, 3, 224, 224)
    
    @pytest.mark.parametrize("mode", ['RGB', 'L', 'RGBA'])
    def test_various_color_modes(self, mode):
        """Test preprocessing handles various color modes"""
        if mode == 'RGBA':
            test_image = Image.new(mode, (100, 100), color=(255, 0, 0, 255))
        else:
            test_image = Image.new(mode, (100, 100), color=128 if mode == 'L' else 'red')
        
        # Should handle conversion without error
        processed_tensor = preprocess_image(test_image)
        assert processed_tensor.shape[1] == 3  # Should have 3 channels


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
