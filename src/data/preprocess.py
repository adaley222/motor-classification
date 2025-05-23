"""
preprocess.py
-------------
Loads raw EEG data, applies filtering, sets montage, extracts events, and epochs the data for model training.
"""

import mne
import numpy as np
import os

def preprocess_edf(edf_path, event_id=None, tmin=0, tmax=4, l_freq=1.0, h_freq=40.0, montage='standard_1005'):
    """
    Loads and preprocesses a single EDF file using MNE.

    Args:
        edf_path (str): Path to the EDF file.
        event_id (dict, optional): Mapping of event labels to integer codes. Defaults to {'T1': 2, 'T2': 3}.
        tmin (float, optional): Start time before event (in seconds). Defaults to 0.
        tmax (float, optional): End time after event (in seconds). Defaults to 4.
        l_freq (float, optional): Low cutoff frequency for bandpass filter. Defaults to 1.0.
        h_freq (float, optional): High cutoff frequency for bandpass filter. Defaults to 40.0.
        montage (str, optional): Name of the electrode montage to use. Defaults to 'standard_1005'.

    Returns:
        tuple: (X, y)
            X (np.ndarray): Array of shape (n_epochs, n_channels, n_times) containing the epoched EEG data.
            y (np.ndarray): Array of shape (n_epochs,) containing the event labels for each epoch.
    """
    raw = mne.io.read_raw_edf(edf_path, preload=True)
    raw.filter(l_freq, h_freq, fir_design='firwin')
    raw.set_montage(montage)
    if event_id is None:
        event_id = dict(T1=2, T2=3)
    events, _ = mne.events_from_annotations(raw, event_id=event_id)
    picks = mne.pick_types(raw.info, eeg=True, exclude='bads')
    epochs = mne.Epochs(raw, events, event_id, tmin, tmax, proj=True, picks=picks, baseline=None, preload=True)
    X = epochs.get_data()
    y = epochs.events[:, -1]
    return X, y
