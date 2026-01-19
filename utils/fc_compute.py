
"""
utils/fc_compute.py
----------------
Connectivity computation.
"""

import numpy as np
from scipy.stats import zscore
from utils.filters import bandpass_filter

def compute_fc_per_mode(imfs):
    """
    Compute Fisher z-transformed FC for each IMF.
    Args:
        imfs: (K, T, R)
    Returns:
        fc_mats: (K, R, R)
    """
    K, T, R = imfs.shape
    FCs = []
    for k in range(K):
        Xk = zscore(imfs[k].T, axis=1)
        fc = np.corrcoef(Xk)
        fc_z = np.arctanh(np.clip(fc, -0.999999, 0.999999))
        FCs.append(fc_z)
    return np.stack(FCs)


def compute_fc_whole_band(bold, tr, lowcut=0.01, highcut=0.1):
    """Compute Fisher z-FC for the full BOLD signal within a band."""
    fs = 1.0 / tr
    bold_filt = bandpass_filter(bold, fs, lowcut, highcut)
    bold_filt = zscore(bold_filt, axis=0)
    fc = np.corrcoef(bold_filt)
    return np.arctanh(np.clip(fc, -0.999999, 0.999999))