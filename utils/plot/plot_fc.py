"""
utils/plot_fc.py
-----------------
Functional connectivity visualization utilities.
"""
import os
import numpy as np
import matplotlib.pyplot as plt

def short_subject_label(subj_id, idx=None):
    # idx is optional (e.g., Subject 01). fallback is a shortened ID.
    if idx is not None:
        return f"Subject {idx:02d}"
    return f"Subject {str(subj_id)[-6:]}"  # last 6 chars as fallback


# ---------------------------------------------------------------------
# Subject-level summary (4 runs)
# ---------------------------------------------------------------------
def plot_fc_subject(
    subj_id, subjects_dict, save_dir="results/figures/fc_subjects",
    include_wholeband=True, method_name="VLMD",
    subj_display=None, save=False, dpi=300
):
    if subj_id not in subjects_dict:
        print(f"Subject {subj_id} not found.")
        return

    runs = sorted(subjects_dict[subj_id], key=lambda r: r["run_idx"])
    group = runs[0]["group"]
    if subj_display is None:
        subj_display = short_subject_label(subj_id)

    n_runs = len(runs)
    K_max = max(r["fc_modes"].shape[0] for r in runs)
    n_cols = K_max + int(include_wholeband)

    # fixed width; height grows with runs
    fig_w = 6.8
    fig_h = max(2.2, 1.7 * n_runs)
    fig, axs = plt.subplots(
        n_runs, n_cols,
        figsize=(fig_w, fig_h),
        layout="constrained"
    )
    if n_runs == 1:
        axs = np.expand_dims(axs, axis=0)

    fig.suptitle(f"{subj_display} ({group}) – {method_name} FC matrices", y=1.01)

    im = None
    for r, run in enumerate(runs):
        fc_modes, freqs, run_idx = run["fc_modes"], run["freqs"], run["run_idx"]
        fc_whole = run["fc_whole"]
        K = fc_modes.shape[0]

        for c in range(n_cols):
            ax = axs[r, c]
            ax.set_xticks([]); ax.set_yticks([])

            if include_wholeband and c == 0:
                if fc_whole is None:
                    ax.axis("off")
                    ax.text(0.5, 0.5, "Missing", ha="center", va="center", fontsize=8)
                else:
                    im = ax.imshow(fc_whole, cmap="RdBu_r", vmin=-1, vmax=1, origin="upper")
                    if r == 0:
                        ax.set_title("Whole\n(0.01–0.10)", fontsize=8)
                ax.set_ylabel(f"Run {run_idx}", rotation=0, labelpad=18, va="center")
                continue

            mode_idx = c if not include_wholeband else c - 1
            if mode_idx < K:
                im = ax.imshow(fc_modes[mode_idx], cmap="RdBu_r", vmin=-1, vmax=1, origin="upper")
                if r == 0:
                    ax.set_title(f"IMF {mode_idx+1}\n{freqs[mode_idx]:.3f}", fontsize=8)
            else:
                ax.axis("off")

            if c == 0 and not include_wholeband:
                ax.set_ylabel(f"Run {run_idx}", rotation=0, labelpad=18, va="center")

    if im is not None:
        cbar = fig.colorbar(im, ax=axs, fraction=0.02, pad=0.01)
        cbar.set_label("Fisher z-FC", rotation=270, labelpad=14)

    if save:
        os.makedirs(save_dir, exist_ok=True)
        png_path = os.path.join(save_dir, f"{subj_id}_{group}_runs_fc.png")
        pdf_path = png_path.replace(".png", ".pdf")
        fig.savefig(pdf_path, bbox_inches="tight")
        fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {pdf_path}")
    else:
        plt.show()

# ---------------------------------------------------------------------
# Subject-level summary (averaged across runs)
# ---------------------------------------------------------------------
def style_matrix_axis(ax, ticks, N, show_ticks_here: bool):
    """
    Critical: lock axis limits to pixel edges so x/y ticks align to matrix indices.
    """
    ax.set_aspect("equal")  # square pixels (prevents stretching)
    ax.set_xlim(-0.5, N - 0.5)

    # origin="upper" means y runs downward; this keeps 0 at top, N-1 at bottom
    ax.set_ylim(N - 0.5, -0.5)

    if show_ticks_here and ticks:
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.tick_params(axis="both", labelsize=7, length=2, pad=1)
    else:
        ax.set_xticks([])
        ax.set_yticks([])

def plot_fc_subject_combined(
    subj_id,
    subjects_dict,
    save_dir="results/figures/fc_subjects",
    include_wholeband=True,
    method_name="VLMD",
    use_network=False,
    subj_display=None,
    show_sparse_ticks=True,
    ticks_on="first",          # "first" or "all" (first = cleaner multi-panel)
    save=False,
    dpi=300,
    vmax=1.0,
):
    if subj_id not in subjects_dict:
        print(f"Subject {subj_id} not found.")
        return

    subj_data = subjects_dict[subj_id]
    if isinstance(subj_data, list):
        subj_data = subj_data[0]

    group = subj_data.get("group", "Unknown")
    freqs = subj_data.get("freqs")

    # --- choose which FC stack to plot ---
    if use_network:
        if "fc_net" not in subj_data:
            raise KeyError(
                "use_network=True but 'fc_net' not found in subj_data. "
                "Compute subj_data['fc_net'] first or set use_network=False."
            )
        fc_modes = subj_data["fc_net"]
        fc_whole = subj_data.get("fc_whole_net", None)  # network whole-band if available
        if fc_whole is None:
            fc_whole = subj_data.get("fc_whole", None)  # fallback
    else:
        fc_modes = subj_data["fc_modes"]
        fc_whole = subj_data.get("fc_whole", None)

    if subj_display is None:
        subj_display = str(subj_id)

    K = fc_modes.shape[0]
    n_cols = K + int(include_wholeband)
    N = fc_modes.shape[-1]

    # --- sparse tick positions ---
    if show_sparse_ticks:
        if use_network:
            ticks = [0, 5, 10, 15, 20]
        else:
            ticks = [0, 100, 200, 300, 400]
        ticks = [t for t in ticks if 0 <= t < N]
    else:
        ticks = []

    # --- size: scale with number of panels (small but meaningful improvement) ---
    fig_w = 1.35 * n_cols + 1.2
    fig_h = 2.6
    fig, axs = plt.subplots(1, n_cols, figsize=(fig_w, fig_h), layout="constrained")
    if n_cols == 1:
        axs = [axs]

    fig.suptitle(f"{subj_display} ({group}) – {method_name}", y=1.02)

    im = None
    col = 0

    # Whole-band
    if include_wholeband:
        ax = axs[col]
        col += 1
        if fc_whole is None:
            ax.axis("off")
            ax.text(0.5, 0.5, "Missing", ha="center", va="center", fontsize=9)
        else:
            im = ax.imshow(
                fc_whole,
                cmap="RdBu_r",
                vmin=-vmax,
                vmax=vmax,
                origin="upper",
                interpolation="nearest"  # avoids smoothing artifacts
            )
            ax.set_title("Whole\n(0.01–0.10 Hz)", fontsize=9)

            show_ticks_here = (ticks_on == "all") or (ticks_on == "first")
            style_matrix_axis(ax, ticks, N, show_ticks_here)

    # IMFs
    for k in range(K):
        ax = axs[col + k]
        im = ax.imshow(
            fc_modes[k],
            cmap="RdBu_r",
            vmin=-vmax,
            vmax=vmax,
            origin="upper",
            interpolation="nearest"
        )

        if freqs is not None and len(freqs) > k:
            ax.set_title(f"IMF {k+1}\n{freqs[k]:.3f} Hz", fontsize=9)
        else:
            ax.set_title(f"IMF {k+1}", fontsize=9)

        show_ticks_here = (ticks_on == "all")  # only first panel by default
        style_matrix_axis(ax, ticks, N, show_ticks_here)

    # Shared colorbar
    if im is not None:
        cbar = fig.colorbar(im, ax=axs, shrink=0.5, aspect=25, pad=0.02)
        cbar.set_label("Fisher z-FC", rotation=270, labelpad=14)

    if save:
        os.makedirs(save_dir, exist_ok=True)
        base = os.path.join(save_dir, f"{subj_id}_{group}_{'net' if use_network else 'roi'}_combined_fc")
        fig.savefig(base + ".pdf", bbox_inches="tight")
        fig.savefig(base + ".png", dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {base}.pdf")
    else:
        plt.show()
