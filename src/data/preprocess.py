"""
preprocess.py
-------------
Loads raw EEG data and processes it using electrode pairs as described in 
"A Simplified CNN Classification Method for MI-EEG via the Electrode Pairs Signals"
by Lun et al. (2020).
"""

import mne
import numpy as np
import os
from pathlib import Path

# Define the 9 pairs of symmetric electrodes over motor cortex region 
# as used in Lun et al. (2020) research paper
ELECTRODE_PAIRS = [
    ('FC5', 'FC6'),
    ('FC3', 'FC4'), 
    ('FC1', 'FC2'),
    ('C5', 'C6'),
    ('C3', 'C4'),
    ('C1', 'C2'),
    ('CP5', 'CP6'),
    ('CP3', 'CP4'),
    ('CP1', 'CP2')
]

def preprocess_edf_electrode_pairs(edf_path, event_id=None, tmin=0, tmax=0.25, 
                                  l_freq=1.0, h_freq=40.0, montage='standard_1005'):
    """
    Loads and preprocesses a single EDF file using electrode pairs approach from Lun et al. (2020).

    Args:
        edf_path (str): Path to the EDF file.
        event_id (dict, optional): Mapping of event labels to integer codes. 
                                  Defaults to {'T1': 2, 'T2': 3, 'T3': 4, 'T4': 5}.
        tmin (float, optional): Start time before event (in seconds). Defaults to 0.
        tmax (float, optional): End time after event (in seconds). Defaults to 0.25.
        l_freq (float, optional): Low cutoff frequency for bandpass filter. Defaults to 1.0.
        h_freq (float, optional): High cutoff frequency for bandpass filter. Defaults to 40.0.
        montage (str, optional): Name of the electrode montage to use. Defaults to 'standard_1005'.

    Returns:
        tuple: (X_pairs, y, pair_names)
            X_pairs (np.ndarray): Array of shape (n_samples, 40, 2) containing electrode pair data.
                                 Each sample contains one electrode pair from one epoch.
            y (np.ndarray): Array of shape (n_samples,) containing the event labels.
            pair_names (list): List of electrode pair names corresponding to each sample.
    """
    # Load raw EDF file
    raw = mne.io.read_raw_edf(edf_path, preload=True)
    
    # Apply bandpass filter (1-40 Hz as specified in paper)
    raw.filter(l_freq, h_freq, fir_design='firwin')
    
    # Set montage for electrode positions, allowing unknown channel positions
    raw.set_montage(montage, on_missing='warn')
    
    # Set up event mapping for T1, T2, T3, T4 (left fist, right fist, both fists, both feet)
    if event_id is None:
        event_id = dict(T1=2, T2=3, T3=4, T4=5)  # PhysioNet event codes
    
    # Extract events from annotations
    events, _ = mne.events_from_annotations(raw, event_id=event_id)
    
    # Create epochs of 0.25 seconds 
    epochs = mne.Epochs(raw, events, event_id, tmin, tmax, 
                       proj=True, baseline=None, preload=True)
    
    # Get channel names
    ch_names = epochs.ch_names
    
    # Create samples for each electrode pair
    X_pairs = []
    y_labels = []
    pair_names = []
    
    for epoch_idx in range(len(epochs)):
        epoch_data = epochs[epoch_idx].get_data()[0]  # Shape: (n_channels, n_times)
        epoch_label = epochs.events[epoch_idx, -1]
        
        # For each electrode pair, create a sample
        for pair_name in ELECTRODE_PAIRS:
            electrode1, electrode2 = pair_name
            
            # Check if both electrodes exist in the data
            if electrode1 in ch_names and electrode2 in ch_names:
                idx1 = ch_names.index(electrode1)
                idx2 = ch_names.index(electrode2)
                
                # Extract data for both electrodes
                signal1 = epoch_data[idx1, :]  # Shape: (n_times,)
                signal2 = epoch_data[idx2, :]  # Shape: (n_times,)
                
                # Combine into pair format: (n_times, 2)
                pair_data = np.column_stack([signal1, signal2])
                
                # Ensure we have exactly 40 time points (160 Hz * 0.25 seconds)
                if pair_data.shape[0] == 40:
                    X_pairs.append(pair_data)
                    y_labels.append(epoch_label)
                    pair_names.append(f"{electrode1}-{electrode2}")
                else:
                    print(f"Warning: Expected 40 time points, got {pair_data.shape[0]} for {pair_name}")
    
    if X_pairs:
        X_pairs = np.array(X_pairs)  # Shape: (n_samples, 40, 2)
        y_labels = np.array(y_labels)
        
        print(f"Created {len(X_pairs)} electrode pair samples")
        print(f"Data shape: {X_pairs.shape}")
        print(f"Available electrode pairs: {len(set(pair_names))}")
        
        return X_pairs, y_labels, pair_names
    else:
        print("No valid electrode pair data found!")
        return None, None, None

def process_subjects_electrode_pairs(raw_dir, processed_dir, subjects=None, runs=None, 
                                    event_id=None, tmin=0, tmax=0.25, 
                                    l_freq=1.0, h_freq=40.0, montage='standard_1005'):
    """
    Process specified EDF files for specified subjects using electrode pairs approach from Lun et al. (2020).
    
    Args:
        raw_dir (str): Path to directory containing subject folders with EDF files
        processed_dir (str): Path to directory to save processed data
        subjects (list, optional): List of subject IDs to process (e.g., [1, 2, 3] or ['S001', 'S002']). 
                                  If None, processes all subjects found.
        runs (list, optional): List of run numbers to process (e.g., [3, 7, 11] for motor imagery runs).
                              If None, defaults to [3, 7, 11] as used in Lun et al. (2020).
        event_id (dict, optional): Mapping of event labels to integer codes.
                                  Defaults to {'T1': 2, 'T2': 3, 'T3': 4, 'T4': 5}.
        tmin (float, optional): Start time before event (in seconds). Defaults to 0.
        tmax (float, optional): End time after event (in seconds). Defaults to 4.0.
        l_freq (float, optional): Low cutoff frequency for bandpass filter. Defaults to 1.0.
        h_freq (float, optional): High cutoff frequency for bandpass filter. Defaults to 40.0.
        montage (str, optional): Name of the electrode montage to use. Defaults to 'standard_1005'.
    
    Returns:
        tuple: (X_all, y_all, subject_ids, pair_names_all)
            X_all (np.ndarray): Combined electrode pair data for all subjects
            y_all (np.ndarray): Combined label array for all subjects  
            subject_ids (np.ndarray): Array indicating which subject each sample belongs to
            pair_names_all (list): List of electrode pair names for each sample
    """
    # Create processed directory if it doesn't exist
    Path(processed_dir).mkdir(parents=True, exist_ok=True)
    
    # Default to motor imagery runs as used in Lun et al. (2020)
    if runs is None:
        runs = [3, 7, 11]
    
    # Set up event mapping for T1, T2, T3, T4 if not provided
    if event_id is None:
        event_id = dict(T1=2, T2=3, T3=4, T4=5)
    
    X_all = []
    y_all = []
    subject_ids = []
    pair_names_all = []
    
    # Get all subject directories
    all_subject_dirs = [d for d in Path(raw_dir).iterdir() if d.is_dir()]
    
    # Filter subjects if specified
    if subjects is not None:
        # Convert subject IDs to directory name format
        target_subjects = []
        for subj in subjects:
            if isinstance(subj, int):
                # Convert integer to S001 format
                target_subjects.append(f"S{subj:03d}")
            else:
                # Assume it's already in correct format
                target_subjects.append(str(subj))
        
        subject_dirs = [d for d in all_subject_dirs if d.name in target_subjects]
        print(f"Processing {len(subject_dirs)} specified subjects: {[d.name for d in subject_dirs]}")
    else:
        subject_dirs = all_subject_dirs
        print(f"Processing all {len(subject_dirs)} subjects found")
    
    print(f"Target runs: {runs}")
    print(f"Event mapping: {event_id}")
    
    for subject_dir in sorted(subject_dirs):
        subject_id = subject_dir.name
        print(f"\nProcessing subject {subject_id}...")
        
        # Process specified run files
        for run_num in runs:
            run_file = f"R{run_num:02d}.edf"  # Format as R03.edf, R07.edf, etc.
            edf_path = subject_dir / run_file
            
            if edf_path.exists():
                print(f"  Processing run: {run_file}")
                
                try:
                    # Process the file
                    X_pairs, y, pair_names = preprocess_edf_electrode_pairs(
                        str(edf_path),
                        event_id=event_id,
                        tmin=tmin,
                        tmax=tmax,
                        l_freq=l_freq,
                        h_freq=h_freq,
                        montage=montage
                    )
                    
                    if X_pairs is not None:
                        X_all.append(X_pairs)
                        y_all.append(y)
                        # Add subject ID for each sample
                        subject_ids.extend([subject_id] * len(y))
                        pair_names_all.extend(pair_names)
                        print(f"    Added {len(X_pairs)} electrode pair samples")
                    
                except Exception as e:
                    print(f"    Error processing {edf_path}: {str(e)}")
                    continue
            else:
                print(f"    File not found: {run_file}")
    
    # Combine all processed data
    if X_all:
        X_combined = np.concatenate(X_all, axis=0)
        y_combined = np.concatenate(y_all, axis=0)
        subject_ids = np.array(subject_ids)
        
        # Save processed data
        np.save(os.path.join(processed_dir, 'X_processed.npy'), X_combined)
        np.save(os.path.join(processed_dir, 'y_processed.npy'), y_combined)
        np.save(os.path.join(processed_dir, 'subject_ids.npy'), subject_ids)
        
        print(f"\n" + "="*50)
        print("PROCESSING COMPLETE!")
        print(f"Total samples: {len(y_combined)}")
        print(f"Data shape: {X_combined.shape}")
        print(f"Label distribution: {dict(zip(*np.unique(y_combined, return_counts=True)))}")
        print(f"Number of subjects: {len(np.unique(subject_ids))}")
        print(f"Electrode pairs used: {len(set(pair_names_all))}")
        print(f"Samples per subject: {len(y_combined) // len(np.unique(subject_ids)):.1f} average")
        print("="*50)
        
        return X_combined, y_combined, subject_ids, pair_names_all
    else:
        print("No data was processed successfully!")
        return None, None, None, None
