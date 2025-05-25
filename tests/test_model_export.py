"""
test_model_export.py
--------------------
Test ONNX export and import for the CNN model.
"""
import torch
import onnx
import onnxruntime as ort
import numpy as np
from src.models.cnn import MotorImageryCNN
from src.models.train import export_to_onnx

def test_onnx_export_and_inference():
    input_shape = (1, 1, 160, 64)
    n_classes = 3
    model = MotorImageryCNN((1, 160, 64), n_classes)
    model.eval()
    dummy_input = torch.randn(*input_shape)
    export_to_onnx(model, input_shape, export_path="test_model.onnx")
    # Check ONNX loads
    onnx_model = onnx.load("test_model.onnx")
    onnx.checker.check_model(onnx_model)
    # Check ONNX inference
    ort_session = ort.InferenceSession("test_model.onnx")
    input_name = ort_session.get_inputs()[0].name
    output = ort_session.run(None, {input_name: dummy_input.numpy()})
    assert output[0].shape[0] == 1
    assert output[0].shape[1] == n_classes
