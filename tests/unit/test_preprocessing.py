"""
test_preprocessing.py
--------------------
Unit tests for the EEG electrode pairs preprocessing pipeline.
"""
import os
import numpy as np
import pytest
import tempfile
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from src.data.preprocess import preprocess_edf_electrode_pairs, process_subjects_electrode_pairs, ELECTRODE_PAIRS
from tests.fixtures.mock_data import create_mock_physionet_files, create_mock_eeg_data

def test_mock_electrode_pairs_data():
    """Test that mock electrode pairs data has correct format."""
    X, y = create_mock_eeg_data(n_samples=50, n_timepoints=40, n_classes=4)
    
    # Test shapes
    assert X.shape == (50, 40, 2), f"Expected (50, 40, 2), got {X.shape}"
    assert y.shape == (50,), f"Expected (50,), got {y.shape}"
    
    # Test data types
    assert X.dtype == np.float32, f"Expected float32, got {X.dtype}"
    assert y.dtype in [np.int32, np.int64], f"Expected int type, got {y.dtype}"
    
    # Test label values (T1=2, T2=3, T3=4, T4=5)
    unique_labels = np.unique(y)
    assert all(label in [2, 3, 4, 5] for label in unique_labels), f"Invalid labels: {unique_labels}"

def test_electrode_pairs_constants():
    """Test that electrode pairs are properly defined."""
    assert len(ELECTRODE_PAIRS) == 9, f"Expected 9 electrode pairs, got {len(ELECTRODE_PAIRS)}"
    
    # Test that all pairs are tuples with 2 elements
    for pair in ELECTRODE_PAIRS:
        assert isinstance(pair, tuple), f"Pair should be tuple, got {type(pair)}"
        assert len(pair) == 2, f"Pair should have 2 elements, got {len(pair)}"
        assert all(isinstance(electrode, str) for electrode in pair), "Electrode names should be strings"
    
    # Test some expected pairs
    assert ('FC3', 'FC4') in ELECTRODE_PAIRS, "FC3-FC4 pair should be included"
    assert ('C3', 'C4') in ELECTRODE_PAIRS, "C3-C4 pair should be included"

def test_preprocess_edf_electrode_pairs_interface():
    """Test electrode pairs preprocessing function interface without MNE dependency."""
    # Test that function can be imported and has correct signature
    assert callable(preprocess_edf_electrode_pairs), "Function should be callable"
    
    # Test default parameters
    import inspect
    sig = inspect.signature(preprocess_edf_electrode_pairs)
    
    expected_params = ['edf_path', 'event_id', 'tmin', 'tmax', 'l_freq', 'h_freq', 'montage']
    actual_params = list(sig.parameters.keys())
    
    assert all(param in actual_params for param in expected_params), f"Missing parameters: {set(expected_params) - set(actual_params)}"
    
    # Test default values
    assert sig.parameters['tmin'].default == 0, "Default tmin should be 0"
    assert sig.parameters['tmax'].default == 0.25, "Default tmax should be 0.25"
    assert sig.parameters['l_freq'].default == 1.0, "Default l_freq should be 1.0"
    assert sig.parameters['h_freq'].default == 40.0, "Default h_freq should be 40.0"
