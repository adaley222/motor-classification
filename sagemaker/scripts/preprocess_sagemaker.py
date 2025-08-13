"""
preprocess_sagemaker.py
-----------------------
SageMaker-compatible preprocessing script for EEG motor imagery data.
Runs as a SageMaker Processing Job with S3 input/output.
"""

import os
import boto3
import numpy as np
import mne
from pathlib import Path
import argparse
import logging
from urllib.error import HTTPError
from mne.datasets import eegbci

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define the 9 pairs of symmetric electrodes over motor cortex region 
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
    Loads and preprocesses a single EDF file using electrode pairs approach for 0.25-second epochs.
    """
    logger.info(f"Processing EDF file with electrode pairs: {edf_path}")
    
    # Load raw EDF file
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    
    # Apply bandpass filter (1-40 Hz)
    raw.filter(l_freq, h_freq, fir_design='firwin', verbose=False)
    
    # Set montage for electrode positions
    raw.set_montage(montage, on_missing='warn', verbose=False)
    
    # Set up event mapping for T1, T2, T3, T4
    if event_id is None:
        event_id = dict(T1=2, T2=3, T3=4, T4=5)  # PhysioNet event codes
    
    # Extract events from annotations
    events, _ = mne.events_from_annotations(raw, event_id=event_id, verbose=False)
    
    # Create epochs of 0.25 seconds
    epochs = mne.Epochs(raw, events, event_id, tmin, tmax, 
                       proj=True, baseline=None, preload=True, verbose=False)
    
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
    
    if X_pairs:
        X_pairs = np.array(X_pairs)  # Shape: (n_samples, 40, 2)
        y_labels = np.array(y_labels)
        
        logger.info(f"Created {len(X_pairs)} electrode pair samples from {edf_path}")
        return X_pairs, y_labels, pair_names
    else:
        logger.warning(f"No valid electrode pair data found in {edf_path}")
        return None, None, None

def download_and_process_physionet(subjects, runs, processing_dir, s3_bucket=None, s3_prefix='processed-data'):
    """
    Download PhysioNet data and process it using electrode pairs approach for SageMaker training.
    
    Args:
        subjects (list): List of subject IDs
        runs (list): List of run numbers (default: [3, 7, 11] for motor imagery)
        processing_dir (str): Local processing directory (/opt/ml/processing/)
        s3_bucket (str): S3 bucket name for output
        s3_prefix (str): S3 prefix for processed data
    """
    # Create directories
    input_dir = os.path.join(processing_dir, 'input')
    output_dir = os.path.join(processing_dir, 'output')
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    X_all = []
    y_all = []
    subject_ids = []
    pair_names_all = []
    
    for subject in subjects:
        logger.info(f"Processing subject {subject}")
        try:
            # Download data for this subject
            raw_fnames = eegbci.load_data(subject, runs, path=input_dir)
            
            for fname in raw_fnames:
                logger.info(f"Processing file: {fname}")
                try:
                    X_pairs, y, pair_names = preprocess_edf_electrode_pairs(fname)
                    if X_pairs is not None:
                        X_all.append(X_pairs)
                        y_all.append(y)
                        subject_ids.extend([subject] * len(y))
                        pair_names_all.extend(pair_names)
                        logger.info(f"Added {len(X_pairs)} electrode pair samples")
                except Exception as e:
                    logger.error(f"Error processing {fname}: {str(e)}")
                    continue
                    
        except HTTPError as e:
            logger.error(f"Error downloading subject {subject}: {e}")
            continue
    
    if X_all:
        # Combine all data
        X_combined = np.concatenate(X_all, axis=0)
        y_combined = np.concatenate(y_all, axis=0)
        subject_ids = np.array(subject_ids)
        
        # Save locally first
        np.save(os.path.join(output_dir, 'X_processed.npy'), X_combined)
        np.save(os.path.join(output_dir, 'y_processed.npy'), y_combined)
        np.save(os.path.join(output_dir, 'subject_ids.npy'), subject_ids)
        
        logger.info(f"Processing complete! Total samples: {len(y_combined)}")
        logger.info(f"Data shape: {X_combined.shape}")
        logger.info(f"Unique labels: {np.unique(y_combined)}")
        logger.info(f"Label distribution: {dict(zip(*np.unique(y_combined, return_counts=True)))}")
        logger.info(f"Electrode pairs used: {len(set(pair_names_all))}")
        
        # Upload to S3 if bucket specified
        if s3_bucket:
            upload_to_s3(output_dir, s3_bucket, s3_prefix)
            
        return X_combined, y_combined, subject_ids
    else:
        logger.error("No data was processed successfully!")
        return None, None, None

def upload_to_s3(local_dir, bucket, prefix):
    """Upload processed files to S3."""
    s3_client = boto3.client('s3')
    
    for file_name in os.listdir(local_dir):
        if file_name.endswith('.npy'):
            local_path = os.path.join(local_dir, file_name)
            s3_key = f"{prefix}/{file_name}"
            
            logger.info(f"Uploading {file_name} to s3://{bucket}/{s3_key}")
            s3_client.upload_file(local_path, bucket, s3_key)

def main():
    parser = argparse.ArgumentParser(description='SageMaker EEG Preprocessing')
    parser.add_argument('--subjects', nargs='+', type=int, default=[1, 2, 3, 4, 5],
                        help='Subject IDs to process')
    parser.add_argument('--runs', nargs='+', type=int, default=[3, 7, 11],
                        help='Run numbers to process')
    parser.add_argument('--s3-bucket', type=str, 
                        help='S3 bucket for output')
    parser.add_argument('--s3-prefix', type=str, default='processed-data',
                        help='S3 prefix for processed data')
    parser.add_argument('--processing-dir', type=str, default='/opt/ml/processing',
                        help='Processing directory')
    
    args = parser.parse_args()
    
    # Run preprocessing
    download_and_process_physionet(
        subjects=args.subjects,
        runs=args.runs,
        processing_dir=args.processing_dir,
        s3_bucket=args.s3_bucket,
        s3_prefix=args.s3_prefix
    )

if __name__ == "__main__":
    main()