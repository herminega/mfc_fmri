
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
    K, T, R = imfs.shape
    FCs = []
    for k in range(K):
        X = zscore(imfs[k], axis=0, nan_policy="omit")   # (T, R), per ROI over time
        print(f"Mode {k}: After z-scoring, X shape: {X.shape}, mean: {np.nanmean(X):.4f}, std: {np.nanstd(X):.4f}")
        fc = np.corrcoef(X, rowvar=False)    # (R, R)
        print(f"Mode {k}: FC shape: {fc.shape}, mean: {np.nanmean(fc):.4f}, std: {np.nanstd(fc):.4f}")
        FCs.append(np.arctanh(np.clip(fc, -0.999999, 0.999999)))
    return np.stack(FCs)

def compute_fc_whole_band(bold_rt, tr, lowcut=0.01, highcut=0.1):
    fs = 1.0 / tr

    # Per-ROI temporal demeaning (consistent with decomposition)
    # bold_rt = bold_rt - bold_rt.mean(axis=1, keepdims=True)

    # Dimension (R, T) for filtering and FC computation
    bold_tr = bold_rt
    print("Before filtering, bold_tr shape:", bold_tr.shape)

    bold_filt = bandpass_filter(bold_tr, fs, lowcut, highcut, axis=1)
    bold_filt = zscore(bold_filt, axis=1, nan_policy="omit")
    print("After filtering, bold_filt shape:", bold_filt.shape)
    print("bold_rt shape:", bold_rt.shape)

    fc = np.corrcoef(bold_filt, rowvar=True)
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