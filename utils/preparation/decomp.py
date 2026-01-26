"""
utils/decomp.py
----------------
Core decomposition utilities for VLMD and MVMD, plus decomposition evaluation.
"""
import sys
sys.path.append("/cluster/home/herminea/mental_health_project/vlmd")  
### Need to download vlmd from GitHub: VLMD, Morante, M., & Rehman, N. (2025).
from vlmd import vlmd 

import time
import numpy as np
from scipy.signal import hilbert
from pysdkit import MVMD
from vlmd import vlmd
from scipy.ndimage import gaussian_filter1d

def compute_mean_frequency_per_imf(imfs, fs=1.25):
    """Compute mean frequency per IMF using Hilbert transform."""
    K, T, R = imfs.shape
    mean_freqs = np.zeros(K)
    for k in range(K):
        inst_freqs_roi = []
        for r in range(R):
            analytic = hilbert(imfs[k, :, r])
            phase = np.unwrap(np.angle(analytic))
            inst_freq = np.diff(phase) * fs / (2 * np.pi)
            inst_freq = inst_freq[np.isfinite(inst_freq)]  # drop NaN/inf
            inst_freqs_roi.append(np.nanmean(inst_freq))
        mean_freqs[k] = np.nanmean(inst_freqs_roi)
    return mean_freqs



def vlmd_to_roi_imfs(latent_modes, latent_coefs):
    """
    Convert VLMD latent modes into ROI-space IMFs.
    Args:
        latent_modes: (K, T, L) or (K, L, T)
        latent_coefs: (L, R) or (R, L)
    Returns:
        imfs: (K, T, R)
    """
    lm = np.asarray(latent_modes)
    lc = np.asarray(latent_coefs)

    if lm.ndim != 3:
        raise ValueError(f"latent_modes must be 3D, got {lm.shape}")

    # Ensure (K, L, T)
    if lm.shape[1] != lc.shape[0] and lm.shape[2] == lc.shape[0]:
        lm = np.transpose(lm, (0, 2, 1))

    # Project to ROI space
    if lc.shape[0] == lm.shape[1]:      # (L, R)
        roi_modes = np.einsum('lc,klt->kct', lc, lm)
    elif lc.shape[1] == lm.shape[1]:    # (R, L)
        roi_modes = np.einsum('cl,klt->kct', lc, lm)
    else:
        raise ValueError(f"Incompatible shapes: lm={lm.shape}, lc={lc.shape}")

    return np.transpose(roi_modes, (0, 2, 1))  # (K, T, R)


def decompose_mvmd(X, fs=1.25, K=5, alpha=1500, verbose=True):
    """
    Multivariate Variational Mode Decomposition (MVMD)
    Args:
        X: (R, T)
    Returns:
        imfs: (K, T, R), freqs: (K,)
    """
    if verbose:
        print(f"[MVMD] Decomposing with K={K}, alpha={alpha}, fs={fs}")
    mvmd = MVMD(alpha=alpha, K=K, tau=0, tol=1e-7)
    imfs = mvmd.fit_transform(X)
    
    freqs = compute_mean_frequency_per_imf(imfs, fs=fs)

    if verbose:
        print(f"[MVMD] Produced {imfs.shape[0]} modes with mean freqs: {np.round(freqs, 4)}")

    return imfs, freqs


def decompose_vlmd(X, fs=1.25, K=5, alpha=1500, num_latents=48, verbose=True):
    """
    Variational Latent Mode Decomposition (VLMD)
    Args:
        X: (R, T)
    Returns:
        imfs_roi: (K, T, R), freqs: (K,)
    """
    start = time.time()
    if verbose:
        print(f"[VLMD] Decomposition started (K={K}, alpha={alpha}, L={num_latents}, fs={fs})")

    lm, lc, omegas, _ = vlmd(
        X,
        num_modes=K,
        num_latents=num_latents,
        alpha=alpha,
        reg_lambda=0.1,
        reg_rho=1.0,
        sampling_rate=fs
    )

    imfs_roi = vlmd_to_roi_imfs(lm, lc)
    freqs = omegas[-1]

    if verbose:
        print(f"[VLMD] Done in {time.time() - start:.2f}s, produced {K} modes")
        
    return imfs_roi, freqs


def run_decomposition_pipeline(X, method="vlmd", n_modes=5, fs=1.25, alpha=1500, num_latents=48):
    if method.lower() == "mvmd":
        return decompose_mvmd(X, fs, K=n_modes, alpha=alpha)
    elif method.lower() == "vlmd":
        return decompose_vlmd(X, fs, K=n_modes, alpha=alpha, num_latents=num_latents)
    else:
        raise ValueError(f"Unknown method '{method}'")


def evaluate_decomposition(X, imfs, fs=1.25):
    """
    Compute quantitative quality metrics for decomposition.
    Returns dict with reconstruction error, orthogonality index,
    mean IMF frequencies, and variance explained.
    """
    K, T, R = imfs.shape
    recon = np.sum(imfs, axis=0)
    err = np.linalg.norm(recon - X.T) / np.linalg.norm(X.T)

    num = np.sum([np.sum(imfs[i]*imfs[j]) for i in range(K) for j in range(K) if i != j])
    den = np.sum([np.sum(imfs[i]**2) for i in range(K)])
    OI = np.abs(num) / den

    mean_freqs = compute_mean_frequency_per_imf(imfs, fs)
    total_var = np.var(X)
    var_per_imf = [np.var(imfs[k]) / total_var for k in range(K)]

    return {
        "reconstruction_error": err,
        "orthogonality_index": OI,
        "mean_imf_freqs": mean_freqs,
        "var_per_imf": var_per_imf,
    }


def compute_hht(imfs, fs=1.25, smooth_sigma=1):
    """
    Compute instantaneous amplitude and frequency for IMFs via Hilbert transform.

    Args:
        imfs: np.ndarray, shape (K, T, R)
            Intrinsic Mode Functions for each ROI.
        fs: float, default=1.25
            Sampling frequency (Hz).
        smooth_sigma: float, default=1
            Gaussian smoothing (in samples) for instantaneous frequency.

    Returns:
        inst_amp, inst_freq : np.ndarray, shape (K, T, R)
            Instantaneous amplitude and frequency for each IMF and ROI.
    """
    analytic = hilbert(imfs, axis=1)
    inst_amp = np.abs(analytic)
    phase = np.unwrap(np.angle(analytic), axis=1)
    inst_freq = np.diff(phase, prepend=phase[:, 0:1, :], axis=1) * (fs / (2 * np.pi))
    if smooth_sigma > 0:
        inst_freq = gaussian_filter1d(inst_freq, sigma=smooth_sigma, axis=1)
    return inst_amp, inst_freq