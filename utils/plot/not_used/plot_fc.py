"""
utils/plot_fc.py
-----------------
Functional connectivity visualization utilities.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from utils.metrics import fc_similarity


# ---------------------------------------------------------------------
# Subject-level summary (multiple runs)
# ---------------------------------------------------------------------
def plot_fc_subject(subj_id, subjects_dict, save_dir="results/figures/fc_subjects",
                    include_wholeband=True, method_name="VLMD", save=False):
    """Plot FC matrices (IMFs + optional whole-band) for one subject."""
    if subj_id not in subjects_dict:
        print(f"Subject {subj_id} not found.")
        return

    runs = sorted(subjects_dict[subj_id], key=lambda r: r["run_idx"])
    group = runs[0]["group"]
    if save:
        os.makedirs(save_dir, exist_ok=True)

    n_runs = len(runs)
    K_max = max(r["fc_modes"].shape[0] for r in runs)
    n_cols = K_max + int(include_wholeband)

    fig = plt.figure(figsize=(3 * n_cols, 3 * n_runs))
    fig.suptitle(f"{subj_id} ({group}) – {method_name} FC matrices", fontsize=14, y=0.99)
    

    plot_idx = 1
    for run in runs:
        fc_modes, freqs, run_idx = run["fc_modes"], run["freqs"], run["run_idx"]
        fc_whole = run["fc_whole"]
        K = fc_modes.shape[0]

        for k in range(n_cols):
            ax = plt.subplot(n_runs, n_cols, plot_idx)
            plot_idx += 1

            # Whole-band FC
            if include_wholeband and k == 0:
                if fc_whole is not None:
                    vmax = np.nanmax(np.abs(fc_whole))
                    im = ax.imshow(fc_whole, cmap="RdBu_r", vmin=-1, vmax=1, origin="upper")
                    if run_idx == 1:
                        ax.set_title("Whole-band\n(0.01–0.1 Hz)", fontsize=8)
                else:
                    ax.text(0.5, 0.5, "Missing", ha="center", va="center", fontsize=8)
                    ax.axis("off")
                continue

            # IMF FCs
            mode_idx = k if not include_wholeband else k - 1
            if mode_idx < K:
                im = ax.imshow(fc_modes[mode_idx], cmap="RdBu_r", vmin=-1, vmax=1, origin="upper")
                if run_idx == 1:
                    ax.set_title(f"IMF {mode_idx+1}\n{freqs[mode_idx]:.3f} Hz", fontsize=8)
            else:
                ax.axis("off")

            if k == 0:
                ax.set_ylabel(f"Run {run_idx}", rotation=0, labelpad=20)
            ax.set_xticks([]); ax.set_yticks([])

    # Shared colorbar
    cbar_ax = fig.add_axes([0.93, 0.15, 0.015, 0.7])
    fig.colorbar(im, cax=cbar_ax).set_label("Fisher z-FC", rotation=270, labelpad=15)
    if save:
        save_path = os.path.join(save_dir, f"{subj_id}_{group}_summary.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {save_path}")
    else:
        plt.show()


# ---------------------------------------------------------------------
# FC similarity plots
# ---------------------------------------------------------------------
def plot_fc_whole_vs_imfs(subject_id, subjects_dict):
    """Plot similarity between each IMF FC and whole-band FC for one subject."""
    runs = sorted(subjects_dict[subject_id], key=lambda r: r["run_idx"])
    group = runs[0]["group"]

    all_sims, all_freqs = [], []
    for run in runs:
        fc_whole = run["fc_whole"]
        fc_modes, freqs = run["fc_modes"], run["freqs"]
        sims = [fc_similarity(fc_whole, fc_modes[k]) for k in range(fc_modes.shape[0])]
        all_sims.append(sims)
        all_freqs.append(freqs)

    max_modes = max(len(s) for s in all_sims)
    padded_sims = np.full((len(all_sims), max_modes), np.nan)
    padded_freqs = np.full_like(padded_sims, np.nan)
    for i, (sims, freqs) in enumerate(zip(all_sims, all_freqs)):
        padded_sims[i, :len(sims)] = sims
        padded_freqs[i, :len(freqs)] = freqs

    mean_sims = np.nanmean(padded_sims, axis=0)
    mean_freqs = np.nanmean(padded_freqs, axis=0)

    plt.figure(figsize=(7, 5))
    for i, (freqs, sims) in enumerate(zip(all_freqs, all_sims), start=1):
        plt.plot(freqs, sims, "o--", alpha=0.6, label=f"Run {i}")
    plt.plot(mean_freqs, mean_sims, "o-", color="teal", lw=2.5, label="Mean")

    plt.xlabel("IMF Frequency (Hz)")
    plt.ylabel("FC similarity to whole-band FC")
    plt.title(f"{subject_id} ({group}) – FC similarity per IMF")
    plt.grid(alpha=0.3)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.show()

    return mean_freqs, mean_sims


def plot_fc_whole_vs_imfs_combined(subject_id, subj_data):
    """
    Plot similarity between each IMF FC and whole-band FC for a single combined subject.
    """
    
    fc_whole = subj_data["fc_whole"]
    fc_modes = subj_data["fc_modes"]
    freqs = subj_data.get("freqs")
    group = subj_data.get("group", "Unknown")

    sims = [fc_similarity(fc_whole, fc_modes[k]) for k in range(fc_modes.shape[0])]
    
    plt.figure(figsize=(6,4))
    plt.plot(freqs, sims, "o-", lw=2, color="teal")
    plt.xlabel("IMF Frequency (Hz)")
    plt.ylabel("FC similarity to whole-band FC")
    plt.title(f"{subject_id} ({group}) – Combined runs")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    return np.array(freqs), np.array(sims)


#HELPER
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt

def add_network_group_boxes(ax, group_ranges, colors, labels, linewidth=2):
    """
    Draw colored rectangles along the diagonal blocks of a square FC matrix.

    group_ranges: list of (start_idx, end_idx) inclusive indices, e.g. [(0,16), (17,19), (20,20)]
    colors:       list of colors, e.g. ["#1f78b4", "#33a02c", "#ff7f00"]
    labels:       list of labels, e.g. ["Cortical (Schaefer-17)", "Subcortical (Tian)", "Cerebellum"]
    """
    handles = []

    for (start, end), color, label in zip(group_ranges, colors, labels):
        # imshow pixel centers are at 0..N-1, so edges are at -0.5..N-0.5
        x = start - 0.5
        y = start - 0.5
        size = end - start + 1

        rect = Rectangle(
            (x, y),
            width=size,
            height=size,
            fill=False,
            edgecolor=color,
            linewidth=linewidth
        )
        ax.add_patch(rect)

        # For legend
        handles.append(Rectangle((0, 0), 1, 1, fill=False, edgecolor=color,
                                 linewidth=linewidth, label=label))

    return handles



### Combined FC 
import os
import matplotlib.pyplot as plt

def plot_fc_subject_combined(
    subj_id,
    subjects_dict,
    save_dir="results/figures/fc_subjects",
    include_wholeband=True,
    method_name="VLMD",
    use_network=True,      # <--- default to network mode
    net_names=None,        # kept for backwards compatibility (unused when use_network=True)
    save=False
):
    """Plot subject-level FC matrices (IMFs + optional whole-band)."""
    if subj_id not in subjects_dict:
        print(f"Subject {subj_id} not found.")
        return

    subj_data = subjects_dict[subj_id]
    fc_modes = subj_data["fc_net"] if use_network else subj_data["fc_modes"]
    freqs    = subj_data.get("freqs")
    fc_whole = subj_data.get("fc_whole")
    group    = subj_data.get("group", "Unknown")

    K = fc_modes.shape[0]
    n_cols = K + int(include_wholeband)

    fig, axs = plt.subplots(1, n_cols, figsize=(3 * n_cols, 3))
    if n_cols == 1:
        axs = [axs]

    fig.suptitle(f"{subj_id} ({group}) – {method_name} FC matrices",
                 fontsize=14, y=1.05)

    # Define network group ranges for 21 networks:
    # 0–16: Schaefer cortical, 17–19: Tian subcortical, 20: Cerebellum
    group_ranges = [(0, 16), (17, 19), (20, 20)]
    group_colors = ["#1f78b4", "#33a02c", "#ff7f00"]  # blue, green, orange
    group_labels = ["Cortical (Schaefer-17)",
                    "Subcortical (Tian-3)",
                    "Cerebellum"]

    legend_handles = None  # we’ll fill this the first time we draw boxes
    im = None  # to keep last imshow for colorbar

    # --- Whole-band FC ---
    if include_wholeband:
        ax = axs[0]
        if fc_whole is not None:
            im = ax.imshow(fc_whole, cmap="RdBu_r", vmin=-1, vmax=1, origin="upper")
            ax.set_title("Whole-band\n(0.01–0.1 Hz)", fontsize=9)

            if use_network:
                legend_handles = add_network_group_boxes(
                    ax, group_ranges, group_colors, group_labels, linewidth=2
                )
                ax.set_xticks([])
                ax.set_yticks([])
        else:
            ax.text(0.5, 0.5, "Missing", ha="center", va="center", fontsize=9)
            ax.axis("off")

    # --- IMF FCs ---
    for k in range(K):
        ax = axs[k + int(include_wholeband)]
        im = ax.imshow(fc_modes[k], cmap="RdBu_r", vmin=-1, vmax=1, origin="upper")
        title = f"IMF {k+1}"
        if freqs is not None and len(freqs) > k:
            title += f"\n{freqs[k]:.3f} Hz"
        ax.set_title(title, fontsize=9)

        if use_network:
            # draw group boxes on each IMF panel
            legend_handles = add_network_group_boxes(
                ax, group_ranges, group_colors, group_labels, linewidth=2
            )
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            # ROI mode – you *could* keep net_names here if desired
            if net_names is not None:
                ax.set_xticks(range(len(net_names)))
                ax.set_xticklabels(net_names, rotation=90, fontsize=6)
                ax.set_yticks(range(len(net_names)))
                ax.set_yticklabels(net_names, fontsize=6)
            else:
                ax.set_xticks([])
                ax.set_yticks([])

    # Shared colorbar
    if im is not None:
        cbar = fig.colorbar(im, ax=axs, shrink=0.7)
        cbar.set_label("Fisher z-FC", rotation=270, labelpad=15)

    # Group legend (cortical / subcortical / cerebellum) – once per figure
    if use_network and legend_handles is not None:
        fig.legend(
            handles=legend_handles,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.02),
            ncols=len(legend_handles),
            frameon=False,
            fontsize=8
        )

    if save:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{subj_id}_{group}_summary.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved {save_path}")
        plt.close(fig)
    else:
        plt.show()
