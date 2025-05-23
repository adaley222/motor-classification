"""
download.py
------------
Downloads EEG motor imagery data from PhysioNet using MNE. Handles subject/run selection and saves raw EDF files to a specified directory.
"""

import os
from mne.datasets import eegbci
from mne.io import read_raw_edf
from mne import concatenate_raws
from urllib.error import HTTPError

def download_physionet(subjects, runs, out_dir):
    """
    Download EEG motor imagery data for given subjects and runs using MNE's eegbci API.

    Args:
        subjects (list of int): List of subject numbers to download.
        runs (list of int): List of run numbers to download for each subject.
        out_dir (str): Directory to save downloaded EDF files.

    Returns:
        list: List of file paths to downloaded EDF files.
    """
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    raw_fnames = []
    for subject in subjects:
        try:
            raw_fnames += eegbci.load_data(subject, runs, path=out_dir)
        except HTTPError as err:
            print(f"Error downloading subject {subject}: {err}")
    return raw_fnames
