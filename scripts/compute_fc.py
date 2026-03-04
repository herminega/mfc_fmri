#!/usr/bin/env python3
"""
scripts/compute_fc.py
-------------------------
Compute functional connectivity (FC) matrices for decomposed and original BOLD signals.
"""

import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pickle
import numpy as np
from utils.functional_connectivity.fc_compute import compute_fc_per_mode, compute_fc_whole_band
from utils.io.io_data import load_roi_timeseries_runs


def compute_fc_matrices(src_dir=None, dest_dir=None, tr=0.8, roi_ts_dir=None):
    os.makedirs(dest_dir, exist_ok=True)

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
        freqs = np.array(data["freqs"])
        run_file = data.get("run_file")

        # Patch run_file to new base dir if provided
        if roi_ts_dir is not None:
            if run_file is None:
                raise ValueError(f"{fname}: run_file missing")
            run_file = os.path.join(roi_ts_dir, os.path.basename(run_file))

        if run_file is None or not os.path.exists(run_file):
            raise FileNotFoundError(f"{fname}: ROI timeseries file not found: {run_file}")

        print(f"Subject: {subj} | Run file: {os.path.basename(run_file)}")

        # Filter IMFs
        low, high = 0.005, 0.25
        valid_idx = np.where((freqs >= low) & (freqs <= high))[0]
        imfs = imfs[valid_idx]
        freqs = freqs[valid_idx]

        fc_modes = compute_fc_per_mode(imfs)

        # Whole-band FC
        runs = load_roi_timeseries_runs(run_file)  # <-- use patched run_file

        run_name = data.get("run_name")
        if isinstance(runs, dict):
            if run_name not in runs:
                raise KeyError(f"{fname}: run_name '{run_name}' not found. Example keys: {list(runs.keys())[:10]}")
            X = runs[run_name]
        else:
            X = runs

        print("X shape used:", X.shape)
        fc_whole = compute_fc_whole_band(X, tr=tr, lowcut=0.01, highcut=0.1)

        result = {
            "subject": subj,
            "imfs": imfs,
            "freqs": freqs,
            "fc_modes": fc_modes,
            "fc_whole": fc_whole,
            "params": data["params"],
            "group": data.get("group"),
            "run_idx": data.get("run_idx"),
            "run_name": run_name,
            "run_file": run_file,
        }

        with open(dest_path, "wb") as f:
            pickle.dump(result, f)

        print(f"Saved FCs to {dest_path}\n")

def main():
    """Main entrypoint for script execution."""
    ROI_TS_DIR = "/cluster/home/herminea/mental_health_project/workspace/data/roi_timeseries"
    
    compute_fc_matrices(
        src_dir="/cluster/home/herminea/mental_health_project/workspace/results/fmri_prep/vlmd/imfs_new",
        dest_dir="/cluster/home/herminea/mental_health_project/workspace/results/fmri_prep/vlmd/fc",
        tr=0.8,
        roi_ts_dir=ROI_TS_DIR
    )
    compute_fc_matrices(
        src_dir="/cluster/home/herminea/mental_health_project/workspace/results/fmri_prep/mvmd/imfs_new",
        dest_dir="/cluster/home/herminea/mental_health_project/workspace/results/fmri_prep/mvmd/fc",
        tr=0.8,
        roi_ts_dir=ROI_TS_DIR
    )
if __name__ == "__main__":
    main()
    