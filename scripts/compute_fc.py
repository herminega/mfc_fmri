#!/usr/bin/env python3
"""
scripts/compute_fc.py
-------------------------
Compute functional connectivity (FC) matrices for decomposed and original BOLD signals.
"""

import os, sys

# --- Add parent directory so we can import from utils ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pickle
import numpy as np
from utils.fc_compute import compute_fc_per_mode, compute_fc_whole_band
from utils.io_data import load_bold_matrix, load_roi_timeseries_runs


def compute_fc_matrices(
    src_dir=None,
    dest_dir=None,
    tr=0.8,
):
    """Compute FC matrices for decomposed modes and whole BOLD signal."""
    os.makedirs(dest_dir, exist_ok=True)

    print(f"\nComputing FC matrices for all decomposed runs in {src_dir}...\n")

    for fname in os.listdir(src_dir):
        if not fname.endswith(".pkl"):
            continue

        src_path = os.path.join(src_dir, fname)
        dest_path = os.path.join(dest_dir, fname.replace(".pkl", "_fc.pkl"))

        if os.path.exists(dest_path):
            print(f"Skipping {fname} (already computed)")
            continue

        with open(src_path, "rb") as f:
            data = pickle.load(f)

        subj = data["subject"]
        imfs = data["imfs"]
        freqs = data["freqs"]
        run_file = data.get("run_file")

        print(f"Subject: {subj} | Run file: {os.path.basename(run_file)}")
        
        # --- Filter IMFs based on frequency range ---
        low, high = 0.005, 0.25
        freqs = np.array(data["freqs"])  # ensure numpy array
        valid_idx = np.where((freqs >= low) & (freqs <= high))[0]

        if len(valid_idx) == 0:
            print(f"[Warning] No IMFs in range {low}-{high} Hz for {subj}")
        else:
            print(f"Keeping {len(valid_idx)}/{len(freqs)} IMFs in {low}-{high} Hz range")

        imfs = imfs[valid_idx]
        freqs = freqs[valid_idx]

        # --- Compute FC per mode (only filtered IMFs) ---
        fc_modes = compute_fc_per_mode(imfs)

        # --- Compute FC for the full BOLD signal (band 0.01–0.1 Hz) ---
        try:
            #X = load_bold_matrix(run_file)
            
            runs = load_roi_timeseries_runs(data["run_file"])            
            X = runs[data["run_name"]]  # same run that was decomposed
            
            fc_whole = compute_fc_whole_band(X, tr=tr, lowcut=0.01, highcut=0.1)
        
        except Exception as e:
            print(f"Warning: Could not compute whole-band FC: {e}")
            fc_whole = None

        result = {
            "subject": subj,
            "imfs": imfs,
            "freqs": freqs,
            "fc_modes": fc_modes,
            "fc_whole": fc_whole,
            "params": data["params"],
            "group": data.get("group"),
            "run_idx": data.get("run_idx"),
            "run_name": data.get("run_name"),
            "run_file": run_file,
        }


        with open(dest_path, "wb") as f:
            pickle.dump(result, f)
        print(f"Saved FCs to {dest_path}\n")

def main():
    """Main entrypoint for script execution."""
    compute_fc_matrices(
        src_dir="/cluster/home/herminea/mental_health_project/test/results/fmri_prep/vlmd/imfs",
        dest_dir="/cluster/home/herminea/mental_health_project/test/results/fmri_prep/vlmd/fc",
        tr=0.8,
    )
    compute_fc_matrices(
        src_dir="/cluster/home/herminea/mental_health_project/test/results/fmri_prep/mvmd/imfs",
        dest_dir="/cluster/home/herminea/mental_health_project/test/results/fmri_prep/mvmd/fc",
        tr=0.8,
    )
if __name__ == "__main__":
    main()
