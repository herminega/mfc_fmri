"""
utils/spectral/marginal_spectrum.py
------------------------------------
Marginal Hilbert-Huang spectrum computation and spectral feature extraction.
"""

import numpy as np
import pandas as pd

from utils.config import FS, FMAX, NBINS
from utils.preparation.decomp import compute_hht


# ---------------------------------------------------------------------------
# Low-level spectral utilities
# ---------------------------------------------------------------------------

def band_mask(freqs, fmin, fmax):
    """Boolean mask selecting frequency bins within [fmin, fmax)."""
    return (freqs >= fmin) & (freqs < fmax)


def shannon_entropy(p, base=2):
    """Shannon entropy of a probability distribution."""
    p = np.asarray(p, float)
    p = p[p > 0]
    if p.size == 0:
        return np.nan
    return (
        -np.sum(p * np.log2(p)) if base == 2
        else -np.sum(p * np.log(p) / np.log(base))
    )


def spectral_entropy(H, eps=1e-12, base=2, normalize=True):
    """Normalised spectral entropy of a power/amplitude spectrum H."""
    H = np.clip(np.asarray(H, float), 0, None)
    s = H.sum()
    if s <= 0:
        return np.nan
    p   = (H + eps) / (s + eps * H.size)
    ent = shannon_entropy(p, base=base)
    if normalize:
        ent /= np.log2(len(p)) if base == 2 else np.log(len(p)) / np.log(base)
    return ent


def spectral_centroid(freqs, H, eps=1e-12):
    """Centre-of-mass frequency of a spectrum H."""
    H   = np.clip(np.asarray(H, float), 0, None)
    den = H.sum()
    if den <= 0:
        return np.nan
    return np.sum(freqs * H) / (den + eps)


# ---------------------------------------------------------------------------
# Marginal spectrum computation
# ---------------------------------------------------------------------------

def compute_network_marginal_spectrum(inst_amp, inst_freq, roi_to_net,
                                      net_idx, fmax=FMAX, nbins=NBINS):
    """
    Compute marginal Hilbert spectrum for one network.

    Parameters
    ----------
    inst_amp, inst_freq : (K, T, R) arrays
    roi_to_net          : (R,) int array mapping each ROI to a network ID
    net_idx             : int, which network to compute
    fmax, nbins         : frequency axis parameters

    Returns
    -------
    freqs : (nbins,) frequency bin centres, or None if network has no ROIs
    H     : (nbins,) marginal spectrum averaged over time and ROIs, or None
    """
    freq_bins = np.linspace(0, fmax, nbins + 1)
    mask_roi  = (roi_to_net == net_idx)
    if not np.any(mask_roi):
        return None, None

    K, T, _ = inst_amp.shape
    amp  = inst_amp[:, :, mask_roi]
    frq  = inst_freq[:, :, mask_roi]
    inds = np.digitize(frq, freq_bins) - 1
    inds = inds.clip(0, len(freq_bins) - 2)

    H = np.zeros(len(freq_bins) - 1)
    for k in range(K):
        np.add.at(H, inds[k].ravel(), amp[k].ravel())
    H /= (T * amp.shape[2])

    freqs = 0.5 * (freq_bins[:-1] + freq_bins[1:])
    return freqs, H


def build_net_marginals(subjects_dict, groups, roi_to_net, n_networks,
                        fs=FS, fmax=FMAX, nbins=NBINS):
    """
    Compute marginal Hilbert spectra for all subjects and all networks.

    Returns
    -------
    net_marginals : dict  net_idx -> {"MDD": [(subj, H), ...], "HC": [...]}
    freqs_ref     : (nbins,) shared frequency axis
    """
    net_marginals = {i: {"MDD": [], "HC": []} for i in range(n_networks)}
    freqs_ref     = None

    for subj, entries in subjects_dict.items():
        imfs  = entries[0]["imfs"]
        group = groups[subj]
        inst_amp, inst_freq = compute_hht(imfs, fs=fs, smooth_sigma=1)

        for net_idx in range(n_networks):
            freqs, H = compute_network_marginal_spectrum(
                inst_amp, inst_freq, roi_to_net, net_idx, fmax=fmax, nbins=nbins
            )
            if freqs is None:
                continue
            if freqs_ref is None:
                freqs_ref = freqs
            net_marginals[net_idx][group].append((subj, H))

    return net_marginals, freqs_ref


# ---------------------------------------------------------------------------
# Spectral feature table
# ---------------------------------------------------------------------------

def build_spectral_features_table(net_marginals, freqs_ref, freq_bands,
                                   net_names, total_range=(0.01, 0.25), eps=1e-12):
    """
    Build a long-format DataFrame with spectral features per
    subject × network × band.

    Columns: subject, group, network, network_id, band, band_amp,
             rel_band_amp, band_entropy, band_centroid, total_amp,
             total_entropy, total_centroid
    """
    rows       = []
    total_mask = band_mask(freqs_ref, *total_range)

    for net_idx, by_group in net_marginals.items():
        net_name = net_names[net_idx]
        for group, subj_list in by_group.items():
            for subj, H in subj_list:
                H         = np.asarray(H, float)
                H_total   = H[total_mask]
                total_amp = np.nansum(H_total)

                for band, (fmin, fmax) in freq_bands.items():
                    m = band_mask(freqs_ref, fmin, fmax)
                    if not np.any(m):
                        continue
                    H_band   = H[m]
                    band_amp = np.nansum(H_band)

                    rows.append({
                        "subject":        subj,
                        "group":          group,
                        "network":        net_name,
                        "network_id":     net_idx,
                        "band":           band,
                        "band_amp":       band_amp,
                        "rel_band_amp":   band_amp / total_amp if total_amp > 0 else np.nan,
                        "band_entropy":   spectral_entropy(H_band, eps=eps, normalize=True),
                        "band_centroid":  spectral_centroid(freqs_ref[m], H_band, eps=eps),
                        "total_amp":      total_amp,
                        "total_entropy":  spectral_entropy(H_total, eps=eps),
                        "total_centroid": spectral_centroid(
                            freqs_ref[total_mask], H_total, eps=eps
                        ),
                    })

    return pd.DataFrame(rows)
