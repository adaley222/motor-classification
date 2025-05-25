"""
test_end_to_end.py
------------------
End-to-end test: preprocess, export, and ONNX inference.
"""
import os
import numpy as np
import torch
from src.data.preprocess import preprocess_edf
from src.models.cnn import MotorImageryCNN
from src.models.train import export_to_onnx
from src.inference.predict import run_onnx_inference

def test_end_to_end():
    test_edf = "test_data/test.edf"  # Replace with a real test file path
    if not os.path.exists(test_edf):
        import pytest; pytest.skip("Test EDF file not found.")
    X, y = preprocess_edf(test_edf)
    X = X[:1]  # Use a single sample for speed
    y = y[:1]
    # Add channel dim if needed
    if X.ndim == 3:
        X = X[:, np.newaxis, :, :]
    model = MotorImageryCNN((1, X.shape[2], X.shape[3]), n_classes=3)
    model.eval()
    dummy_input = torch.from_numpy(X)
    export_to_onnx(model, dummy_input.shape, export_path="test_model.onnx")
    output = run_onnx_inference("test_model.onnx", X)
    assert output.shape[0] == 1
