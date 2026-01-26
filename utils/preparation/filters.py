"""
utils/filters.py
----------------
Filtering utilities for BOLD data and IMF selection.
"""
from scipy.signal import butter, filtfilt


def bandpass_filter(data, fs, lowcut=0.01, highcut=0.1, order=4):
    """Apply Butterworth bandpass filter to BOLD signals."""
    nyq = 0.5 * fs
    low, high = lowcut / nyq, highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data, axis=0)


def filter_imfs_by_freq(imfs, freqs, fmin=0.005, fmax=0.25, verbose=True):
    """Keep only IMFs with central frequencies within [fmin, fmax]."""
    mask = (freqs >= fmin) & (freqs <= fmax)
    filtered_imfs = imfs[mask, :, :]
    filtered_freqs = freqs[mask]
    if verbose:
        print(f"[Filter] Retained {filtered_imfs.shape[0]}/{len(freqs)} IMFs ({fmin}-{fmax} Hz)")
    return filtered_imfs, filtered_freqs
