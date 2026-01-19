"""
utils/plot_decomp.py
---------------------
Visualization utilities for VLMD/MVMD decomposition results.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
from scipy.ndimage import gaussian_filter1d
from utils.plot.plot_base import add_freq_bands
from utils.decomp import compute_hht
from utils.io_data import load_roi_timeseries_runs

# ---------------------------------------------------------------
# IMF and frequency plots
# ---------------------------------------------------------------
def plot_imfs(X, imfs, roi_idx=0, freqs=None, subj_label="", method_name="VLMD", save_path=None):
    """
    Plot original signal and its IMFs for a single ROI.
    Args:
        X: (R, T)
        imfs: (K, T, R)
    """
    K, T, R = imfs.shape
    t = np.arange(T)
    fig, axs = plt.subplots(K + 1, 1, figsize=(12, 1.6 * (K + 1)), sharex=True)

    axs[0].plot(t, X[roi_idx], color='k', lw=1.2)
    axs[0].set_ylabel("Original", rotation=0, labelpad=25)
    axs[0].set_title(f"{method_name} – ROI {roi_idx} – {subj_label}")
    axs[0].grid(alpha=0.3)

    for k in range(K):
        axs[k + 1].plot(t, imfs[k, :, roi_idx], lw=0.9)
        label = f"IMF {k+1}"
        if freqs is not None and len(freqs) > k:
            label += f"\n{freqs[k]:.3f} Hz"
        axs[k + 1].set_ylabel(label, rotation=0, labelpad=30)
        axs[k + 1].grid(alpha=0.2)

    axs[-1].set_xlabel("Time (samples)")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_imfs_with_spectrum(X, imfs, fs=1.25, roi_idx=0, freqs=None,
                            subj_label="", method_name="VLMD", fmax=0.3, save_path=None):
    """
    Plot the original signal and each IMF with its corresponding PSD side by side,
    preserving the vertical structure of the pure IMF plot.
    
    Args:
        X: (R, T) array of signals
        imfs: (K, T, R) array of IMFs
        fs: sampling frequency
        roi_idx: ROI index
    """
    K, T, R = imfs.shape
    t = np.arange(T) / fs

    # +1 row for original signal
    fig, axs = plt.subplots(K + 1, 2, figsize=(12, 1.6 * (K + 1)), sharex='col')

    # --- Row 0: Original signal ---
    axs[0, 0].plot(t, X[roi_idx], color='k', lw=1.2)
    axs[0, 0].set_ylabel("Original", rotation=0, labelpad=25)
    axs[0, 0].grid(alpha=0.3)
    axs[0, 0].set_title(f"{method_name} – ROI {roi_idx} – {subj_label}")

    # Power spectrum of original signal
    f, Pxx = welch(X[roi_idx], fs=fs, nperseg=min(256, T))
    axs[0, 1].semilogy(f, Pxx, color='darkorange', lw=0.8)
    axs[0, 1].set_xlim(0, fmax)
    axs[0, 1].set_ylabel("PSD", rotation=0, labelpad=20)
    axs[0, 1].grid(alpha=0.3)

    # --- Each IMF row ---
    for k in range(K):
        # Time-domain IMF
        axs[k + 1, 0].plot(t, imfs[k, :, roi_idx], lw=0.9)
        label = f"IMF {k+1}"
        if freqs is not None and len(freqs) > k:
            label += f"\n{freqs[k]:.3f} Hz"
        axs[k + 1, 0].set_ylabel(label, rotation=0, labelpad=30)
        axs[k + 1, 0].grid(alpha=0.3)

        # Frequency-domain IMF
        f, Pxx = welch(imfs[k, :, roi_idx], fs=fs, nperseg=min(256, T))
        axs[k + 1, 1].semilogy(f, Pxx, color='darkorange', lw=0.8)
        axs[k + 1, 1].set_xlim(0, fmax)
        axs[k + 1, 1].grid(alpha=0.3)
    
    # shade all PSD axes, but legend only on the top one
    add_freq_bands(axs[0, 1], alpha=0.10, add_legend=True, legend_outside=True)
    for k in range(K):
        add_freq_bands(axs[k + 1, 1], alpha=0.10, add_legend=False)

    # Axis labels
    axs[-1, 0].set_xlabel("Time (s)")
    axs[-1, 1].set_xlabel("Frequency (Hz)")

    # Clean layout
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()



# ---------------------------------------------------------------------
# HILBERT–HUANG TRANSFORMS (IMPROVED)
# ---------------------------------------------------------------------
def plot_imfs_with_hht(imfs, inst_amp, inst_freq, fs=1.25, roi_idx=0,
                       freqs=None, subj_label="", method_name="VLMD",
                       fmax=0.3, smooth_sigma=1, cmap="turbo", save_path=None):
    """
    Plot each IMF (time-domain) next to its Hilbert–Huang spectrum.
    """
    K, T, R = imfs.shape
    t = np.arange(T) / fs
    freq_bins = np.linspace(0, fmax, 200)

    fig, axs = plt.subplots(K, 2, figsize=(12, 1.8 * K), gridspec_kw={'width_ratios': [1.2, 2]}, sharex='col')
    axs = np.atleast_2d(axs)

    for k in range(K):
        # --- IMF signal ---
        axs[k, 0].plot(t, imfs[k, :, roi_idx], lw=0.8, color="navy")
        label = f"IMF {k+1}"
        if freqs is not None and len(freqs) > k:
            label += f"\n{freqs[k]:.3f} Hz"
        axs[k, 0].set_ylabel(label, rotation=0, labelpad=30)
        axs[k, 0].grid(alpha=0.2)

        # --- HHT Spectrum ---
        f = inst_freq[k, :, roi_idx]
        a = inst_amp[k, :, roi_idx]
        inds = np.digitize(f, freq_bins) - 1
        H = np.zeros((len(freq_bins)-1, T))
        np.add.at(H, (inds.clip(0, len(freq_bins)-2), np.arange(T)), a)
        if smooth_sigma > 0:
            H = gaussian_filter1d(H, sigma=smooth_sigma, axis=1)

        pcm = axs[k, 1].pcolormesh(t, freq_bins[:-1], H, shading="auto", cmap=cmap)
        axs[k, 1].set_ylim(0, fmax)
        axs[k, 1].grid(False)

        if k == 0:
            axs[k, 0].set_title("IMF Signal", fontsize=10)
            axs[k, 1].set_title("Hilbert–Huang Spectrum", fontsize=10)

    # --- Axis labels ---
    axs[-1, 0].set_xlabel("Time (s)")
    axs[-1, 1].set_xlabel("Time (s)")
    for ax in axs[:, 1]:
        ax.set_ylabel("Frequency (Hz)")

    # --- Shared colorbar to the right ---
    fig.subplots_adjust(right=0.85, wspace=0.25, hspace=0.3)
    cbar_ax = fig.add_axes([0.88, 0.12, 0.02, 0.75])  # [left, bottom, width, height]
    cbar = fig.colorbar(pcm, cax=cbar_ax)
    cbar.set_label("Amplitude", rotation=270, labelpad=15)

    # --- Title ---
    fig.suptitle(f"{method_name} – ROI {roi_idx} ({subj_label})", fontsize=13, y=0.98)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

def plot_combined_hht(inst_amp, inst_freq, fs, roi_idx=0,
                      method_name="VLMD", subj_label="", fmax=0.3, smooth_sigma=1,
                      save_path=None):
    """
    Combined Hilbert–Huang spectrum:
    Sum amplitude contributions from all IMFs.
    """
    K, T, R = inst_amp.shape
    t = np.arange(T) / fs
    freq_bins = np.linspace(0, fmax, 200)
    H = np.zeros((len(freq_bins)-1, T))

    for k in range(K):
        f = inst_freq[k, :, roi_idx]
        a = inst_amp[k, :, roi_idx]
        inds = np.digitize(f, freq_bins) - 1
        np.add.at(H, (inds.clip(0, len(freq_bins)-2), np.arange(T)), a)

    if smooth_sigma > 0:
        H = gaussian_filter1d(H, sigma=smooth_sigma, axis=1)

    plt.figure(figsize=(10, 5))
    plt.pcolormesh(t, freq_bins[:-1], H, shading='auto', cmap='turbo')
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.title(f"Combined Hilbert–Huang Spectrum – ROI {roi_idx} ({method_name}, {subj_label})")
    plt.colorbar(label="Amplitude (summed across IMFs)")
    plt.ylim(0, fmax)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

def plot_marginal_spectrum(inst_amp, inst_freq, roi_idx=0, fmax=0.3, bins=200, save_path=None):
    """
    Marginal Hilbert amplitude spectrum (integrated over time).
    """
    K, T, R = inst_amp.shape
    freq_bins = np.linspace(0, fmax, bins)
    spec = np.zeros(bins - 1)
    for k in range(K):
        f = inst_freq[k, :, roi_idx]
        a = inst_amp[k, :, roi_idx]
        inds = np.digitize(f, freq_bins) - 1
        np.add.at(spec, inds.clip(0, bins - 2), a)

    plt.figure(figsize=(8, 4))
    plt.plot(freq_bins[:-1], spec, color="steelblue")
    add_freq_bands(plt.gca(), alpha=0.1)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude (a.u.)")
    plt.title(f"Marginal Hilbert Spectrum – ROI {roi_idx}")
    plt.xlim(0, fmax)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

def plot_band_hht(inst_amp, inst_freq, fs, freq_bands, roi_idx=0,
                  method_name="VLMD", subj_label="", smooth_sigma=1,
                  save_path=None):
    """
    Hilbert–Huang spectrum summarized into predefined frequency bands (Slow-5, Slow-4, ...).

    inst_amp:  (K, T, R) instantaneous amplitudes
    inst_freq: (K, T, R) instantaneous frequencies in Hz
    freq_bands: dict, e.g. {"Slow-5": (0.01, 0.027), ...}
    """
    K, T, R = inst_amp.shape
    t = np.arange(T) / fs

    band_names = list(freq_bands.keys())
    band_edges = np.array([freq_bands[b] for b in band_names])  # shape (B, 2)
    B = len(band_names)

    # H_band[b, t] = summed amplitude across all IMFs whose inst. freq lies in band b at time t
    H_band = np.zeros((B, T))

    for k in range(K):
        f = inst_freq[k, :, roi_idx]  # (T,)
        a = inst_amp[k, :, roi_idx]   # (T,)

        # For each band, add amplitude where f is within band
        for b in range(B):
            f_lo, f_hi = band_edges[b]
            mask = (f >= f_lo) & (f < f_hi)
            H_band[b, mask] += a[mask]

    if smooth_sigma > 0:
        H_band = gaussian_filter1d(H_band, sigma=smooth_sigma, axis=1)

    # ---- Plot as time × band "heatmap" ----
    plt.figure(figsize=(10, 4))
    extent = [t[0], t[-1], 0, B]  # x_min, x_max, y_min, y_max

    plt.imshow(H_band[::-1, :],  # flip so first band is at top
               aspect="auto",
               extent=extent,
               interpolation="nearest",
               cmap="turbo")

    plt.colorbar(label="Amplitude (summed across IMFs)")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency band")

    # y-ticks at band centers with labels
    y_pos = np.arange(B) + 0.5
    plt.yticks(y_pos, band_names[::-1])

    plt.title(f"Banded Hilbert–Huang Spectrum – ROI {roi_idx} "
              f"({method_name}, {subj_label})")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()



def plot_example_subjects(results, plot_hht=True, plot_psd=True, method_name="VLMD", freq_bands=None, save_figs=False):
    """
    Plot example decompositions (IMFs, spectra, and HHTs) for one MDD and one HC subject.
    Automatically computes Hilbert–Huang transforms before plotting.

    Args:
        results: list of subject results dicts (with keys like 'group', 'subject', 'imfs', 'freqs', etc.)
        plot_hht: bool, whether to plot Hilbert–Huang spectra
        plot_psd: bool, whether to include PSD plots
        save_figs: bool, whether to save figures to file
    """

    mdd_data = next((r for r in results if r["group"] == "MDD"), None)
    hc_data  = next((r for r in results if r["group"] == "HC"), None)

    for group, data in [("MDD", mdd_data), ("HC", hc_data)]:
        if data is None:
            print(f"No data found for group {group}")
            continue

        subj_label = f"{data['subject']} ({group}) run {data['run_idx']}"
        #X = load_bold_matrix(data["run_file"])
        runs = load_roi_timeseries_runs(data["run_file"])
        X = runs[data["run_name"]]  # same run that was decomposed

        fs = 1 / 0.8
        
        # --- Compute HHT on the fly ---
        inst_amp, inst_freq = compute_hht(data["imfs"], fs=fs, smooth_sigma=1)

        # --- 1. Time-domain IMFs ---
        plot_imfs(
            X=X,
            imfs=data["imfs"],
            roi_idx=0,
            freqs=data["freqs"],
            subj_label=subj_label,
            method_name=method_name,
            save_path=f"{data['subject']}_imfs.png" if save_figs else None,
        )

        # --- 2. IMFs with spectra (Welch PSDs) ---
        if plot_psd:
            plot_imfs_with_spectrum(
                X=X,
                imfs=data["imfs"],
                fs=fs,
                roi_idx=0,
                freqs=data["freqs"],
                subj_label=subj_label,
                method_name=method_name,
                fmax=0.3,
                save_path=f"{data['subject']}_imfs_psd.png" if save_figs else None,
            )

        # --- 3. Hilbert–Huang per IMF ---
        if plot_hht:
            plot_imfs_with_hht(
                imfs=data["imfs"],
                inst_amp=inst_amp,
                inst_freq=inst_freq,
                fs=fs,
                roi_idx=0,
                freqs=data["freqs"],
                subj_label=subj_label,
                method_name=method_name,
                fmax=0.3,
                smooth_sigma=1,
                save_path=f"{data['subject']}_hht_per_imf.png" if save_figs else None,
            )

            # --- 4. Combined HHT ---
            plot_combined_hht(
                inst_amp=inst_amp,
                inst_freq=inst_freq,
                fs=fs,
                roi_idx=0,
                method_name=method_name,
                subj_label=subj_label,
                fmax=0.3,
                smooth_sigma=1,
                save_path=None,
            )
            
            plot_band_hht(
                inst_amp=inst_amp,
                inst_freq=inst_freq,
                fs=fs,
                freq_bands=freq_bands,
                roi_idx=0,
                method_name=method_name,
                subj_label=subj_label,
                smooth_sigma=1,
                save_path=None,
            )
