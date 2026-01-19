import os
import glob

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import signal
from scipy.stats import zscore

def normalize_subject_id(subj_id: str) -> str:
    """Convert NDAR_INVxxxxx → sub-NDARINVxxxxx."""
    subj_id_clean = subj_id.replace("NDAR_INV", "NDARINV")
    return f"sub-{subj_id_clean}"

def revert_to_meta_id(fmriprep_id: str) -> str:
    """Convert sub-NDARINVxxxxx → NDAR_INVxxxxx."""
    return fmriprep_id.replace("sub-NDARINV", "NDAR_INV")

def load_subject_list(path: str) -> list[str]:
    """Load and normalize subject IDs from text file."""
    with open(path, "r") as f:
        subjects = [normalize_subject_id(line.strip()) for line in f if line.strip()]
    print(f"[IO] Loaded {len(subjects)} subjects from {path}")
    return subjects

def find_fmriprep_rest_bold(subjects: list[str], base_path: str) -> dict[str, list[str]]:
    """Find resting-state preprocessed BOLD files in fMRIPrep output."""
    data = {}
    for subj in subjects:
        func_dir = os.path.join(base_path, subj, "func")
        if not os.path.exists(func_dir):
            print(f"Missing func directory for {subj}")
            continue

        pattern = os.path.join(
            func_dir,
            f"{subj}_task-rest*_space-MNI152NLin2009cAsym*_desc-preproc_bold.nii.gz"
        )
        bold_files = sorted(glob.glob(pattern))
        if bold_files:
            data[subj] = bold_files
        else:
            print(f"No resting-state BOLD found for {subj}")

    print(f"Found resting-state data for {len(data)} / {len(subjects)} subjects")
    return data

def load_nifti_bold(nifti_path):
    """
    Load preprocessed BOLD NIfTI file from fMRIPrep.
    Returns data as (X, Y, Z, T) numpy array and TR.
    """
    img = nib.load(nifti_path)
    data = img.get_fdata()  # shape: (x, y, z, time)
    header = img.header
    tr = header.get_zooms()[-1]
    return data, tr

def load_confounds(bold_path):
    """
    Given a preprocessed BOLD NIfTI path, find and load the corresponding
    fMRIPrep confounds .tsv file as a pandas DataFrame.
    """
    func_dir = os.path.dirname(bold_path)
    base = os.path.basename(bold_path)

    # Remove the space and res info (fMRIPrep confounds never include these)
    pattern_core = base.split("_space-")[0]
    pattern = os.path.join(func_dir, f"{pattern_core}_desc-confounds_timeseries.tsv")

    matches = glob.glob(pattern)
    if len(matches) == 0:
        raise FileNotFoundError(f"No confounds file found for {bold_path}")
    if len(matches) > 1:
        print(f"Multiple confounds files found for {bold_path}, using first.")

    confound_path = matches[0]
    df = pd.read_csv(confound_path, sep="\t")
    return df
