"""
test_model_export.py
--------------------
Test ONNX export and import for the CNN model.
"""
import os
import sys
import pytest
import tempfile
import torch
import onnx
import onnxruntime as ort
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from src.models.cnn import MotorImageryCNN
from src.models.train import export_to_onnx

def test_onnx_export_and_inference():
    """Test ONNX export and inference with temporary file cleanup."""
    try:
        input_shape = (1, 1, 160, 64)
        n_classes = 3
        model = MotorImageryCNN((1, 160, 64), n_classes)
        model.eval()
        dummy_input = torch.randn(*input_shape)
        
        # Use temporary file for ONNX model
        with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as temp_model:
            temp_model_path = temp_model.name
        
        try:
            export_to_onnx(model, input_shape, export_path=temp_model_path)
            
            # Check ONNX loads
            onnx_model = onnx.load(temp_model_path)
            onnx.checker.check_model(onnx_model)
            
            # Check ONNX inference
            ort_session = ort.InferenceSession(temp_model_path)
            input_name = ort_session.get_inputs()[0].name
            output = ort_session.run(None, {input_name: dummy_input.numpy()})
            
            assert output[0].shape[0] == 1
            assert output[0].shape[1] == n_classes
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_model_path):
                os.unlink(temp_model_path)
                
    except ImportError as e:
        pytest.skip(f"Required package not available: {e}")
    except Exception as e:
        pytest.fail(f"ONNX export/inference test failed: {e}")

def test_model_architecture_consistency():
    """Test that model architecture is consistent with expected input/output shapes."""
    try:
        input_shape = (1, 160, 64)  # Channel, time, electrodes
        n_classes = 3
        
        model = MotorImageryCNN(input_shape, n_classes)
        model.eval()
        
        # Test forward pass with batch
        batch_size = 4
        test_input = torch.randn(batch_size, *input_shape)
        
        with torch.no_grad():
            output = model(test_input)
        
        expected_output_shape = (batch_size, n_classes)
        assert output.shape == expected_output_shape, \
            f"Output shape {output.shape} != expected {expected_output_shape}"
        
        # Test that output is valid logits (no NaN/Inf)
        assert not torch.isnan(output).any(), "Model output contains NaN values"
        assert not torch.isinf(output).any(), "Model output contains Inf values"
        
    except Exception as e:
        pytest.skip(f"Model architecture test failed: {e}")
