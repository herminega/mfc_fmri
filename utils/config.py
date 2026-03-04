"""
utils/config.py
---------------
Shared project-wide constants and path helpers.
Import from here instead of redefining in every notebook.
"""

import os

WORKSPACE  = "/cluster/home/herminea/mental_health_project/workspace"
ATLAS_PATH = os.path.join(WORKSPACE, "atlas/roi_to_net_434.csv")

# fMRI slow-oscillation frequency bands (Hz)
FREQ_BANDS = {
    "Slow-5": (0.01,  0.027),
    "Slow-4": (0.027, 0.073),
    "Slow-3": (0.073, 0.198),
    "Slow-2": (0.198, 0.25),
    "Whole":  (0.01,  0.1),
}

BANDS   = ["Slow-5", "Slow-4", "Slow-3", "Slow-2", "Whole"]
METRICS = ["band_amp", "rel_band_amp", "total_amp"]

FS    = 1 / 0.8   # fMRI sampling rate for TR = 0.8 s
FMAX  = 0.25      # Maximum frequency of interest (Hz)
NBINS = 200       # Frequency bins for marginal spectrum


def results_dir(method: str, kind: str = "fc") -> str:
    """Return the results directory for a given decomposition method."""
    return os.path.join(WORKSPACE, f"results/fmri_prep/{method}/{kind}")
