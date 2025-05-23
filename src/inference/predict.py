"""
predict.py
----------
Loads an ONNX model and runs inference on preprocessed EEG data. Designed for use in a Tauri app with BCI devices (e.g., OpenBCI).
"""

import onnxruntime as ort
import numpy as np

def run_onnx_inference(onnx_path, input_data):
    """
    Run inference on EEG data using an ONNX model.

    Args:
        onnx_path (str): Path to the ONNX model file.
        input_data (np.ndarray): Preprocessed EEG data of shape (batch, channels, height, width).

    Returns:
        np.ndarray: Model output logits or probabilities.
    """
    session = ort.InferenceSession(onnx_path)
    input_name = session.get_inputs()[0].name
    # Ensure input_data is float32 and contiguous
    input_data = np.ascontiguousarray(input_data.astype(np.float32))
    outputs = session.run(None, {input_name: input_data})
    return outputs[0]
