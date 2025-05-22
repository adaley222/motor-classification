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
    Downloads EEG motor imagery data for given subjects and runs using MNE's eegbci API.
    Saves EDF files to out_dir.
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
