"""
test_preprocessing.py
--------------------
Unit tests for the EEG preprocessing pipeline.
"""
import os
import numpy as np
import pytest
from src.data.preprocess import preprocess_edf

def test_preprocess_edf_shape():
    # Use a small test EDF file path or mock if available
    test_edf = "test_data/test.edf"  # Replace with a real test file path
    if not os.path.exists(test_edf):
        pytest.skip("Test EDF file not found.")
    X, y = preprocess_edf(test_edf)
    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert X.shape[0] == y.shape[0]
    assert X.ndim == 3
    assert y.ndim == 1
