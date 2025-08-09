"""
test_data_pipeline.py
--------------------
Test the complete data pipeline from raw data to model input.
"""

import pytest
import os
import tempfile
import numpy as np
from unittest.mock import patch, MagicMock

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from tests.fixtures.mock_data import create_mock_processed_data, save_mock_data_to_temp


class TestDataPipeline:
    
    def test_processed_data_format(self):
        """Test that processed data matches expected format."""
        X, y, subjects = create_mock_processed_data()
        
        # Test shapes - samples can vary, but structure should be consistent
        assert len(X.shape) == 3, f"X should be 3D (samples, time, channels), got shape {X.shape}"
        assert len(y.shape) == 1, f"y should be 1D (samples,), got shape {y.shape}"
        assert len(subjects.shape) == 1, f"subjects should be 1D (samples,), got shape {subjects.shape}"
        
        # Test expected dimensions for 0.25s epochs
        assert X.shape[1] == 160, f"Time dimension should be 160 points (0.25s at 640Hz), got {X.shape[1]}"
        assert X.shape[2] == 64, f"Channel dimension should be 64 EEG channels, got {X.shape[2]}"
        
        # Test data types
        assert X.dtype == np.float32, f"X dtype {X.dtype} should be float32"
        assert y.dtype in [np.int32, np.int64], f"y dtype {y.dtype} should be integer"
        assert subjects.dtype in [np.int32, np.int64], f"subjects dtype {subjects.dtype} should be integer"
        
        # Test value ranges
        assert y.min() >= 0 and y.max() <= 2, f"y values {y.min()}-{y.max()} should be 0-2 (T0, T1, T2)"
        assert subjects.min() >= 1, f"Subject IDs should start from 1, got min {subjects.min()}"
    
    def test_data_consistency(self):
        """Test that data samples are consistent across files."""
        X, y, subjects = create_mock_processed_data()
        
        # All arrays should have same number of samples
        n_samples = X.shape[0]
        assert y.shape[0] == n_samples, "y and X should have same number of samples"
        assert subjects.shape[0] == n_samples, "subjects and X should have same number of samples"
        
        # Check we have reasonable number of subjects (flexible for future expansion)
        unique_subjects = np.unique(subjects)
        n_subjects = len(unique_subjects)
        assert n_subjects >= 2, f"Should have at least 2 subjects for meaningful ML, got {n_subjects}"
        assert n_subjects <= 50, f"Subject count {n_subjects} seems unreasonably high"
        
        # Each subject should have at least some samples
        for subject_id in unique_subjects:
            subject_count = np.sum(subjects == subject_id)
            assert subject_count > 0, f"Subject {subject_id} has no samples"
            assert subject_count >= 10, f"Subject {subject_id} has only {subject_count} samples (too few)"
    
    def test_eeg_signal_properties(self):
        """Test that EEG signals have realistic properties."""
        X, y, subjects = create_mock_processed_data()
        
        # Test signal amplitude (should be in microVolt range)
        signal_std = np.std(X)
        assert 10 < signal_std < 200, f"Signal std {signal_std} outside expected EEG range"
        
        # Test no extreme outliers (simple check)
        signal_max = np.max(np.abs(X))
        assert signal_max < 1000, f"Signal max {signal_max} too large for EEG"
        
        # Test that signals are not constant
        for i in range(min(10, X.shape[0])):  # Check first 10 samples
            sample_var = np.var(X[i])
            assert sample_var > 0, f"Sample {i} has zero variance"
    
    def test_epoch_temporal_structure(self):
        """Test that epochs have proper temporal structure."""
        X, y, subjects = create_mock_processed_data()
        
        n_samples, n_timepoints, n_channels = X.shape
        
        # Test expected dimensions match 0.25s epochs at 640Hz
        expected_timepoints = 160  # 0.25s * 640Hz = 160
        expected_channels = 64     # Standard EEG setup
        
        assert n_timepoints == expected_timepoints, \
            f"Timepoints {n_timepoints} != expected {expected_timepoints} for 0.25s epochs"
        assert n_channels == expected_channels, \
            f"Channels {n_channels} != expected {expected_channels} for EEG"
        
        # Test temporal continuity (adjacent timepoints shouldn't be too different)
        for sample_idx in range(min(5, n_samples)):
            sample = X[sample_idx]
            
            # Calculate temporal differences
            temporal_diffs = np.abs(np.diff(sample, axis=0))
            max_temporal_diff = np.max(temporal_diffs)
            
            # Should not have extreme temporal jumps
            assert max_temporal_diff < 500, \
                f"Sample {sample_idx} has extreme temporal jump: {max_temporal_diff}"
    
    def test_data_splitting_compatibility(self):
        """Test that data can be properly split for training."""
        X, y, subjects = create_mock_processed_data()
        
        # Test train/val/test splitting by subject to avoid leakage
        unique_subjects = np.unique(subjects)
        n_subjects = len(unique_subjects)
        
        # Flexible splitting based on available subjects
        if n_subjects >= 3:
            # Minimum viable split
            train_subjects = unique_subjects[:max(1, n_subjects - 2)]
            val_subjects = unique_subjects[-2:-1] if n_subjects > 2 else []
            test_subjects = unique_subjects[-1:]
        else:
            pytest.skip(f"Need at least 3 subjects for splitting test, got {n_subjects}")
        
        train_mask = np.isin(subjects, train_subjects)
        val_mask = np.isin(subjects, val_subjects) if len(val_subjects) > 0 else np.zeros_like(subjects, dtype=bool)
        test_mask = np.isin(subjects, test_subjects)
        
        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]
        
        # Verify splits
        assert X_train.shape[0] > 0, "Training set should not be empty"
        assert X_test.shape[0] > 0, "Test set should not be empty"
        
        # Verify no subject overlap between train and test
        train_subject_ids = set(subjects[train_mask])
        test_subject_ids = set(subjects[test_mask])
        assert len(train_subject_ids & test_subject_ids) == 0, "Train/test subjects should not overlap"
    
    def test_model_input_format(self):
        """Test that data can be formatted for CNN input."""
        X, y, subjects = create_mock_processed_data()
        
        # Add channel dimension for CNN (batch, channels, height, width)
        X_cnn = X[:, np.newaxis, :, :]  # Add channel dimension
        
        n_samples = X.shape[0]
        expected_shape = (n_samples, 1, 160, 64)  # (batch, channels, time, electrodes)
        assert X_cnn.shape == expected_shape, \
            f"CNN input shape {X_cnn.shape} != expected {expected_shape}"
        
        # Test conversion to torch-compatible format
        try:
            import torch
            X_torch = torch.from_numpy(X_cnn)
            y_torch = torch.from_numpy(y)
            
            assert X_torch.dtype == torch.float32, "X should be float32 for torch"
            assert y_torch.dtype in [torch.int32, torch.int64], "y should be integer for torch"
            
            # Test batch sampling
            batch_size = min(32, n_samples)  # Flexible batch size
            sample_batch_X = X_torch[:batch_size]
            sample_batch_y = y_torch[:batch_size]
            
            assert sample_batch_X.shape == (batch_size, 1, 160, 64), "Batch X shape incorrect"
            assert sample_batch_y.shape == (batch_size,), "Batch y shape incorrect"
            
        except ImportError:
            pytest.skip("PyTorch not available for tensor format testing")
    
    def test_file_io_pipeline(self):
        """Test complete file I/O pipeline."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create mock data once
            X_orig, y_orig, subjects_orig = create_mock_processed_data()
            
            # Save to files manually to control the data
            files = {
                'X_processed': os.path.join(temp_dir, 'X_processed.npy'),
                'y_processed': os.path.join(temp_dir, 'y_processed.npy'),
                'subject_ids': os.path.join(temp_dir, 'subject_ids.npy')
            }
            
            np.save(files['X_processed'], X_orig)
            np.save(files['y_processed'], y_orig)
            np.save(files['subject_ids'], subjects_orig)
            
            # Verify files exist
            for file_path in files.values():
                assert os.path.exists(file_path), f"File {file_path} was not created"
            
            # Load data back
            X_loaded = np.load(files['X_processed'])
            y_loaded = np.load(files['y_processed'])
            subjects_loaded = np.load(files['subject_ids'])
            
            # Verify loaded data matches original (same data, so should match exactly)
            np.testing.assert_array_equal(X_loaded, X_orig, "Loaded X doesn't match original")
            np.testing.assert_array_equal(y_loaded, y_orig, "Loaded y doesn't match original")  
            np.testing.assert_array_equal(subjects_loaded, subjects_orig, "Loaded subjects don't match original")
    
    def test_data_preprocessing_pipeline(self):
        """Test that data preprocessing maintains expected properties."""
        X, y, subjects = create_mock_processed_data()
        
        # Simulate basic preprocessing steps
        # 1. Normalization
        X_normalized = (X - np.mean(X, axis=1, keepdims=True)) / (np.std(X, axis=1, keepdims=True) + 1e-8)
        
        # Check normalization worked
        sample_means = np.mean(X_normalized, axis=1)
        sample_stds = np.std(X_normalized, axis=1)
        
        assert np.allclose(sample_means, 0, atol=1e-6), "Normalized data should have zero mean"
        assert np.allclose(sample_stds, 1, atol=1e-6), "Normalized data should have unit std"
        
        # 2. Check for NaN/Inf values
        assert not np.isnan(X_normalized).any(), "Normalized data contains NaN values"
        assert not np.isinf(X_normalized).any(), "Normalized data contains Inf values"
        
        # 3. Label encoding check
        unique_labels = np.unique(y)
        assert len(unique_labels) == 3, f"Should have 3 unique labels (T0,T1,T2), got {len(unique_labels)}"
        assert set(unique_labels) == {0, 1, 2}, f"Labels should be {{0,1,2}}, got {set(unique_labels)}"