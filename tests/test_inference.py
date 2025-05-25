"""
test_inference.py
-----------------
Test ONNX inference pipeline.
"""
import numpy as np
from src.inference.predict import run_onnx_inference

def test_run_onnx_inference():
    onnx_path = "test_model.onnx"  # Should be created by test_model_export.py
    dummy_input = np.random.randn(1, 1, 160, 64).astype(np.float32)
    output = run_onnx_inference(onnx_path, dummy_input)
    assert output.shape[0] == 1
