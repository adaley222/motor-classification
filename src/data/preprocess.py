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
    - Bandpass filters
    - Sets montage
    - Extracts events
    - Epochs the data
    Returns epochs and labels.
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
