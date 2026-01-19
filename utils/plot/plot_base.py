"""
utils/plot/plot_base.py
--------------
Visualization utilities for VLMD/MVMD decomposition results.
Includes IMF plots, Hilbert–Huang transforms, frequency spectra, and FC visualizations.
"""

import numpy as np
from scipy.signal import hilbert
from scipy.ndimage import gaussian_filter1d
import matplotlib as mpl

def set_mpl_style():
    mpl.rcParams.update({
        # --- figure ---
        "figure.dpi": 120,
        "savefig.dpi": 300,

        # --- fonts ---
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 9,

        # --- axes ---
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "axes.linewidth": 0.8,
        "axes.titlepad": 6,

        # --- ticks ---
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,

        # --- legends ---
        "legend.fontsize": 8,
        "legend.frameon": False,

        # --- lines ---
        "lines.linewidth": 1.2,
        "lines.markersize": 5,

        # --- PDF / vector output ---
        "pdf.fonttype": 42,   # embeds text as editable text
        "ps.fonttype": 42,
    })



FREQ_BANDS = {
    "Slow-5": (0.01, 0.027),
    "Slow-4": (0.027, 0.073),
    "Slow-3": (0.073, 0.198),
    "Slow-2": (0.198, 0.25),
}

def add_freq_bands(ax, alpha=0.12, add_legend=False, legend_outside=True):
    colors = ["#a6cee3", "#b2df8a", "#fb9a99", "#fdbf6f"]
    handles = []
    labels = []

    for (band, (fmin, fmax)), c in zip(FREQ_BANDS.items(), colors):
        h = ax.axvspan(fmin, fmax, color=c, alpha=alpha, lw=0, zorder=0)
        handles.append(h)
        labels.append(band)

    if add_legend:
        if legend_outside:
            ax.legend(handles, labels,
                      frameon=False, ncols=2,
                      loc="upper left", bbox_to_anchor=(1.02, 1.0),
                      borderaxespad=0.0)
        else:
            ax.legend(handles, labels, frameon=False, ncols=2, loc="upper right")

    return handles, labels




