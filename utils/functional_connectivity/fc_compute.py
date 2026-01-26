
"""
utils/fc_compute.py
----------------
Connectivity computation.
"""
import numpy as np
from scipy.stats import zscore
from collections import defaultdict
from utils.preparation.filters import bandpass_filter

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

def bin_fcs_by_freq(subjects_combined, freq_bands):
    binned = {band: defaultdict(list) for band in freq_bands}
    
    for subj, entries in subjects_combined.items():
        e = entries[0]
        # Assign each IMF FC to its corresponding frequency band
        for f, fc in zip(e["freqs"], e["fc_modes"]):
            for band, (fmin, fmax) in freq_bands.items():
                if fmin <= f < fmax:
                    binned[band][subj].append(fc)
                    


    # Average FCs per subject for each band
    binned_mean = {
        band: {s: np.nanmean(fcs, axis=0) for s, fcs in subj_dict.items()}
        for band, subj_dict in binned.items()
    }

    return binned_mean