"""
mock_data.py
------------
Utilities for creating mock EEG data for testing.
"""

import numpy as np
import tempfile
import os
from typing import Tuple

def create_mock_eeg_data(n_samples: int = 100, n_channels: int = 64, 
                        n_timepoints: int = 160, n_classes: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create mock EEG data for testing.
    
    Args:
        n_samples: Number of samples/epochs
        n_channels: Number of EEG channels
        n_timepoints: Number of time points per epoch (160 for 0.25s at 640Hz)
        n_classes: Number of classes (T0, T1, T2)
        
    Returns:
        Tuple of (X, y) where X is EEG data and y are labels
    """
    # Generate realistic-looking EEG data with some structure
    X = np.random.randn(n_samples, n_timepoints, n_channels).astype(np.float32)
    
    # Add some signal-like patterns for 0.25 second epochs
    for i in range(n_samples):
        # Add some oscillatory components
        t = np.linspace(0, 0.25, n_timepoints)  # 0.25 second epochs
        for freq in [8, 12, 20]:  # Alpha, beta frequencies
            X[i] += 0.1 * np.sin(2 * np.pi * freq * t)[:, np.newaxis]
    
    # Scale to typical EEG amplitude range (microvolts)
    X *= 50
    
    # Generate balanced labels
    y = np.random.randint(0, n_classes, size=n_samples)
    
    return X, y

def create_mock_physionet_files(temp_dir: str, n_subjects: int = 2, 
                               n_runs: int = 2) -> list:
    """
    Create mock PhysioNet-style files for testing.
    
    Args:
        temp_dir: Temporary directory to create files in
        n_subjects: Number of subjects to simulate
        n_runs: Number of runs per subject
        
    Returns:
        List of created file paths
    """
    created_files = []
    
    for subject in range(1, n_subjects + 1):
        for run in [3, 7, 11][:n_runs]:  # Use actual PhysioNet run numbers
            # Create mock .edf filename
            filename = f"S{subject:03d}R{run:02d}.edf"
            filepath = os.path.join(temp_dir, filename)
            
            # Create a minimal mock file (not actual EDF format, just for path testing)
            with open(filepath, 'wb') as f:
                f.write(b"MOCK_EDF_DATA" + b"\x00" * 1000)
            
            created_files.append(filepath)
    
    return created_files

def create_mock_processed_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create mock processed data matching expected SageMaker format.
    
    Returns:
        Tuple of (X_processed, y_processed, subject_ids)
    """
    n_samples = 240  # ~2400 total epochs / 10 for testing
    X_processed, y_processed = create_mock_eeg_data(n_samples)
    
    # Create subject IDs (5 subjects, balanced)
    subject_ids = np.repeat(np.arange(1, 6), n_samples // 5)
    
    return X_processed, y_processed, subject_ids

def save_mock_data_to_temp(temp_dir: str) -> dict:
    """
    Save mock processed data to temporary files.
    
    Args:
        temp_dir: Directory to save files in
        
    Returns:
        Dictionary with file paths
    """
    X, y, subjects = create_mock_processed_data()
    
    files = {
        'X_processed': os.path.join(temp_dir, 'X_processed.npy'),
        'y_processed': os.path.join(temp_dir, 'y_processed.npy'),
        'subject_ids': os.path.join(temp_dir, 'subject_ids.npy')
    }
    
    np.save(files['X_processed'], X)
    np.save(files['y_processed'], y)
    np.save(files['subject_ids'], subjects)
    
    return files