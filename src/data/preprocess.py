"""
preprocess.py
-------------
Loads raw EEG data, applies filtering, sets montage, extracts events, and epochs the data for model training.
"""

import mne
import numpy as np
import os
from pathlib import Path

def preprocess_edf(edf_path, event_id=None, tmin=0, tmax=0.25, l_freq=1.0, h_freq=40.0, montage='standard_1005'):
    """
    Loads and preprocesses a single EDF file using MNE.

    Args:
        edf_path (str): Path to the EDF file.
        event_id (dict, optional): Mapping of event labels to integer codes. Defaults to {'T0': 1, 'T1': 2, 'T2': 3}.
        tmin (float, optional): Start time before event (in seconds). Defaults to 0.
        tmax (float, optional): End time after event (in seconds). Defaults to 0.25.
        l_freq (float, optional): Low cutoff frequency for bandpass filter. Defaults to 1.0.
        h_freq (float, optional): High cutoff frequency for bandpass filter. Defaults to 40.0.
        montage (str, optional): Name of the electrode montage to use. Defaults to 'standard_1005'.

    Returns:
        tuple: (X, y)
            X (np.ndarray): Array of shape (n_epochs, n_channels, n_times) containing the epoched EEG data.
            y (np.ndarray): Array of shape (n_epochs,) containing the event labels for each epoch.
    """
    # Load raw EDF file
    raw = mne.io.read_raw_edf(edf_path, preload=True)
    
    # Apply bandpass filter
    raw.filter(l_freq, h_freq, fir_design='firwin')
    
    # Set montage for electrode positions, allowing unknown channel positions
    raw.set_montage(montage, on_missing='warn')
    
    # Set up event mapping including T0 (resting state)
    if event_id is None:
        event_id = dict(T0=1, T1=2, T2=3)
    
    # Extract events from annotations
    events, _ = mne.events_from_annotations(raw, event_id=event_id)
    
    # Select EEG channels, excluding any marked as bad
    picks = mne.pick_types(raw.info, eeg=True, exclude='bads')
    
    # Create epochs of 0.25 seconds
    epochs = mne.Epochs(raw, events, event_id, tmin, tmax, 
                       proj=True, picks=picks, 
                       baseline=None, preload=True)
    
    # Get data and labels
    X = epochs.get_data()
    y = epochs.events[:, -1]
    
    return X, y

def process_all_subjects(raw_dir, processed_dir, event_id=None, tmin=0, tmax=0.25, 
                        l_freq=1.0, h_freq=40.0, montage='standard_1005'):
    """
    Process all EDF files for all subjects in the raw directory.
    
    Args:
        raw_dir (str): Path to directory containing subject folders with EDF files
        processed_dir (str): Path to directory to save processed data
        event_id (dict, optional): Mapping of event labels to integer codes
        tmin (float, optional): Start time before event (in seconds)
        tmax (float, optional): End time after event (in seconds)
        l_freq (float, optional): Low cutoff frequency for bandpass filter
        h_freq (float, optional): High cutoff frequency for bandpass filter
        montage (str, optional): Name of the electrode montage to use
    
    Returns:
        tuple: (X_all, y_all, subject_ids)
            X_all (np.ndarray): Combined feature array for all subjects
            y_all (np.ndarray): Combined label array for all subjects
            subject_ids (np.ndarray): Array indicating which subject each epoch belongs to
    """
    # Create processed directory if it doesn't exist
    Path(processed_dir).mkdir(parents=True, exist_ok=True)
    
    X_all = []
    y_all = []
    subject_ids = []  # To keep track of which subject each epoch belongs to
    
    # Get all subject directories
    subject_dirs = [d for d in Path(raw_dir).iterdir() if d.is_dir()]
    
    for subject_dir in sorted(subject_dirs):
        subject_id = subject_dir.name
        print(f"Processing subject {subject_id}...")
        
        # Process all EDF files for this subject
        for edf_file in subject_dir.glob('*.edf'):
            print(f"  Processing run: {edf_file.name}")
            
            try:
                # Process the file
                X, y = preprocess_edf(
                    str(edf_file),
                    event_id=event_id,
                    tmin=tmin,
                    tmax=tmax,
                    l_freq=l_freq,
                    h_freq=h_freq,
                    montage=montage
                )
                
                X_all.append(X)
                y_all.append(y)
                # Add subject ID for each epoch
                subject_ids.extend([subject_id] * len(y))
                
            except Exception as e:
                print(f"Error processing {edf_file}: {str(e)}")
                continue
    
    # Combine all processed data
    if X_all:
        X_combined = np.concatenate(X_all, axis=0)
        y_combined = np.concatenate(y_all, axis=0)
        subject_ids = np.array(subject_ids)
        
        # Save processed data
        np.save(os.path.join(processed_dir, 'X_processed.npy'), X_combined)
        np.save(os.path.join(processed_dir, 'y_processed.npy'), y_combined)
        np.save(os.path.join(processed_dir, 'subject_ids.npy'), subject_ids)
        
        print(f"\nProcessing complete!")
        print(f"Total epochs: {len(y_combined)}")
        print(f"Data shape: {X_combined.shape}")
        print(f"Unique labels: {np.unique(y_combined)}")
        print(f"Number of subjects: {len(np.unique(subject_ids))}")
        
        return X_combined, y_combined, subject_ids
    else:
        print("No data was processed successfully!")
        return None, None, None
