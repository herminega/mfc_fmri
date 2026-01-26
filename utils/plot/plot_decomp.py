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
from utils.preparation.decomp import compute_hht
from utils.io.io_data import load_roi_timeseries_runs

# ---------------------------------------------------------------
# IMF and frequency plots
# ---------------------------------------------------------------
def plot_imfs(
    X, imfs, roi_idx=0, freqs=None, fs=1.25,
    subj_id="Subject 01", group="MDD", method_name="VLMD",
    save_path=None
):
    K, T, R = imfs.shape
    t = np.arange(T) / fs

    fig, axs = plt.subplots(
        K + 1, 1,
        figsize=(6.5, 1.0 * (K + 1)),
        sharex=True,
        layout="constrained"
    )

    fig.suptitle(f"{subj_id} ({group}) – {method_name}", y=1.02)

    # Original
    axs[0].plot(t, X[roi_idx], color="black", lw=1)
    axs[0].set_ylabel("BOLD", rotation=0, ha="right", va="center")
    axs[0].yaxis.set_label_coords(-0.05, 0.5)
    axs[0].set_yticks([])

    for k in range(K):
        axs[k + 1].plot(t, imfs[k, :, roi_idx], lw=0.9)
        if freqs is not None:
            label = f"IMF {k+1} ({freqs[k]:.3f} Hz)"
        else:
            label = f"IMF {k+1}"
        axs[k + 1].set_ylabel(label, rotation=0, ha="right", va="center")
        axs[k + 1].yaxis.set_label_coords(-0.05, 0.5)
        axs[k + 1].set_yticks([])

    axs[-1].set_xlabel("Time (s)")

    if save_path:
        fig.savefig(save_path.replace(".png", ".pdf"), bbox_inches="tight")
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def plot_imfs_with_spectrum(
    X, imfs, fs=1.25, roi_idx=0, freqs=None,
    subj_id="Subject 01", group="MDD", method_name="VLMD",
    fmax=0.3, save_path=None
):
    K, T, R = imfs.shape
    t = np.arange(T) / fs

    fig, axs = plt.subplots(
        K + 1, 2,
        figsize=(6.8, 1.0 * (K + 1)),
        sharex="col",
        gridspec_kw={"width_ratios": [1.4, 1.0]},
        layout="constrained"
    )

    fig.suptitle(f"{subj_id} ({group}) – {method_name}", y=1.02)

    axs[0, 0].set_title("Time series")
    axs[0, 1].set_title("Power spectrum")

    # Original
    axs[0, 0].plot(t, X[roi_idx], color="black", lw=1)
    axs[0, 0].set_ylabel("BOLD", rotation=0, ha="right")
    axs[0, 0].yaxis.set_label_coords(-0.05, 0.5)
    axs[0, 0].set_yticks([])

    f, Pxx = welch(X[roi_idx], fs=fs, nperseg=min(256, T))
    axs[0, 1].semilogy(f, Pxx, lw=1)
    axs[0, 1].set_xlim(0, fmax)
    axs[0, 1].set_yticks([])

    add_freq_bands(axs[0, 1], alpha=0.12, add_legend=False)

    for k in range(K):
        axs[k + 1, 0].plot(t, imfs[k, :, roi_idx], lw=0.9)
        axs[k + 1, 0].set_ylabel(
            f"IMF {k+1}\n{freqs[k]:.3f} Hz", rotation=0, ha="right"
        )
        axs[k + 1, 0].yaxis.set_label_coords(-0.05, 0.5)
        axs[k + 1, 0].set_yticks([])

        f, Pxx = welch(imfs[k, :, roi_idx], fs=fs, nperseg=min(256, T))
        axs[k + 1, 1].semilogy(f, Pxx, lw=1)
        axs[k + 1, 1].set_xlim(0, fmax)
        axs[k + 1, 1].set_yticks([])
        add_freq_bands(axs[k + 1, 1], alpha=0.12, add_legend=False)

    axs[-1, 0].set_xlabel("Time (s)")
    axs[-1, 1].set_xlabel("Frequency (Hz)")

    if save_path:
        fig.savefig(save_path.replace(".png", ".pdf"), bbox_inches="tight")
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()



# ---------------------------------------------------------------------
# HILBERT–HUANG TRANSFORMS (IMPROVED)
# ---------------------------------------------------------------------
def plot_combined_hht(
    inst_amp, inst_freq, fs, roi_idx=0,
    subj_id="Subject 01", group="MDD", method_name="VLMD",
    fmax=0.3, smooth_sigma=1, save_path=None
):
    K, T, R = inst_amp.shape
    t = np.arange(T) / fs
    freq_bins = np.linspace(0, fmax, 200)
    H = np.zeros((len(freq_bins)-1, T))

    for k in range(K):
        inds = np.digitize(inst_freq[k, :, roi_idx], freq_bins) - 1
        np.add.at(H, (inds.clip(0, len(freq_bins)-2), np.arange(T)),
                  inst_amp[k, :, roi_idx])

    if smooth_sigma > 0:
        H = gaussian_filter1d(H, sigma=smooth_sigma, axis=1)

    fig, ax = plt.subplots(figsize=(6.5, 3.5), layout="constrained")
    pcm = ax.pcolormesh(t, freq_bins[:-1], H, shading="auto", cmap="turbo")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(f"Combined Hilbert–Huang spectrum\n{subj_id} ({group}, {method_name})")

    cbar = fig.colorbar(pcm, ax=ax, fraction=0.05, pad=0.02)
    cbar.set_label("Amplitude (summed across IMFs)", rotation=270, labelpad=14)

    if save_path:
        fig.savefig(save_path.replace(".png", ".pdf"), bbox_inches="tight")
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def plot_example_subjects(results, plot_hht=True, plot_psd=True,
                          method_name="VLMD", save_figs=False):
    mdd_data = next((r for r in results if r["group"] == "MDD"), None)
    hc_data  = next((r for r in results if r["group"] == "HC"), None)

    for i, (group, data) in enumerate([("MDD", mdd_data), ("HC", hc_data)], start=1):
        if data is None:
            print(f"No data found for group {group}")
            continue

        # Short, thesis-friendly label
        subj_id = f"Subject {i:02d}"

        runs = load_roi_timeseries_runs(data["run_file"])
        X = runs[data["run_name"]]

        fs = 1 / 0.8  # TR = 0.8 s

        inst_amp, inst_freq = compute_hht(data["imfs"], fs=fs, smooth_sigma=1)

        # --- 1) IMFs (time-domain) ---
        plot_imfs(
            X=X,
            imfs=data["imfs"],
            roi_idx=0,
            freqs=data["freqs"],
            fs=fs,
            subj_id=subj_id,
            group=group,
            method_name=method_name,
            save_path=f"{data['subject']}_imfs.png" if save_figs else None,
        )

        # --- 2) IMFs + PSD ---
        if plot_psd:
            plot_imfs_with_spectrum(
                X=X,
                imfs=data["imfs"],
                fs=fs,
                roi_idx=0,
                freqs=data["freqs"],
                subj_id=subj_id,
                group=group,
                method_name=method_name,
                fmax=0.3,
                save_path=f"{data['subject']}_imfs_psd.png" if save_figs else None,
            )

        # --- 3) Combined HHT only ---
        if plot_hht:
            plot_combined_hht(
                inst_amp=inst_amp,
                inst_freq=inst_freq,
                fs=fs,
                roi_idx=0,
                subj_id=subj_id,
                group=group,
                method_name=method_name,
                fmax=0.3,
                smooth_sigma=1,
                save_path=f"{data['subject']}_combined_hht.png" if save_figs else None,
            )

