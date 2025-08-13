"""
mock_data.py
------------
Utilities for creating mock EEG data for testing.
"""

import numpy as np
import tempfile
import os
from typing import Tuple

def create_mock_eeg_data(n_samples: int = 100, n_timepoints: int = 40, 
                        n_classes: int = 4) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create mock EEG electrode pairs data for testing.
    
    Args:
        n_samples: Number of samples/epochs
        n_timepoints: Number of time points per epoch (40 for 0.25s at 160Hz)
        n_classes: Number of classes (T1, T2, T3, T4)
        
    Returns:
        Tuple of (X, y) where X is electrode pairs data (n_samples, 40, 2) and y are labels
    """
    # Generate realistic-looking EEG electrode pairs data
    X = np.random.randn(n_samples, n_timepoints, 2).astype(np.float32)
    
    # Add some signal-like patterns for 0.25 second epochs
    for i in range(n_samples):
        # Add some oscillatory components
        t = np.linspace(0, 0.25, n_timepoints)  # 0.25 second epochs
        for freq in [8, 12, 20]:  # Alpha, beta frequencies
            # Add correlated signals to both electrodes in the pair
            signal = 0.1 * np.sin(2 * np.pi * freq * t)
            X[i, :, 0] += signal
            X[i, :, 1] += signal * 0.8  # Slightly different amplitude for realism
    
    # Scale to typical EEG amplitude range (microvolts)
    X *= 50
    
    # Generate balanced labels (T1=2, T2=3, T3=4, T4=5)
    y = np.random.choice([2, 3, 4, 5], size=n_samples)
    
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
    Create mock processed electrode pairs data matching expected SageMaker format.
    
    Returns:
        Tuple of (X_processed, y_processed, subject_ids)
    """
    n_samples = 240  # Test dataset size
    X_processed, y_processed = create_mock_eeg_data(n_samples)
    
    # Create subject IDs (5 subjects, balanced)
    subject_ids = np.repeat([f"S{i:03d}" for i in range(1, 6)], n_samples // 5)
    
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