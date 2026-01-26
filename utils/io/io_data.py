"""
utils/io.py
------------
Helper functions for loading and saving subject data.
"""

import os
import glob
import h5py
from scipy import signal
from scipy.stats import zscore

def load_subject_list(path):
    """Load subject IDs from text file (one per line)."""
    with open(path, 'r') as f:
        subjects = [line.strip() for line in f if line.strip()]
    print(f"[IO] Loaded {len(subjects)} subjects from {path}")
    return subjects


def find_bold_files(subjects, base_path):
    """Find all BOLD .h5 files for given subjects."""
    data = {}
    for subj in subjects:
        subj_dir = os.path.join(base_path, subj)
        bold_files = glob.glob(
            os.path.join(subj_dir, "task-rest*_run-*_bold_Atlas_hp2000_clean_GSR_parcellated.h5")
        )
        if bold_files:
            data[subj] = bold_files
    print(f"[IO] Found BOLD data for {len(data)} subjects")
    return data


def load_bold_matrix(h5_file_path):
    """Load BOLD data (ROIs × T) from .h5 file, detrend, and z-score."""
    with h5py.File(h5_file_path, "r") as f:
        data = f["dataset"][:]  # (ROIs, T)
    data = signal.detrend(data, axis=1)
    data = zscore(data, axis=1)
    return data

def load_roi_timeseries_runs(h5_path):
    """Load all ROI×T runs from one subject's parcellated file."""
    runs = {}
    with h5py.File(h5_path, "r") as f:
        for run_name in f.keys():
            runs[run_name] = f[run_name][:]
    return runs



