#!/usr/bin/env python3
import os, sys, pickle
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.io.io_data import load_roi_timeseries_runs
from utils.functional_connectivity.fc_compute import compute_fc_whole_band

def recompute_fc_whole_only(fc_dir, tr=0.8, overwrite=True, out_dir=None):
    """
    Recompute ONLY fc_whole for each *_fc.pkl in fc_dir.
    If overwrite=True, updates files in place.
    If overwrite=False, writes updated pickles to out_dir.
    """
    if not overwrite:
        assert out_dir is not None, "Provide out_dir when overwrite=False"
        os.makedirs(out_dir, exist_ok=True)

    files = [f for f in os.listdir(fc_dir) if f.endswith("_fc.pkl")]
    print(f"Found {len(files)} FC files in {fc_dir}")

    for fname in sorted(files):
        path = os.path.join(fc_dir, fname)

        with open(path, "rb") as f:
            d = pickle.load(f)

        # Fix path if your environment changed (test -> workspace)
        run_file = d["run_file"].replace("/test/", "/workspace/")

        try:
            runs = load_roi_timeseries_runs(run_file)
            X = runs[d["run_name"]]   # (R,T)
            print("X shape used:", X.shape)
            fc_whole_new = compute_fc_whole_band(X, tr=tr, lowcut=0.01, highcut=0.1)

            # sanity check: must be ROI×ROI
            R = X.shape[0]
            assert fc_whole_new.shape == (R, R), f"Unexpected fc_whole shape {fc_whole_new.shape} (R={R})"

            d["fc_whole"] = fc_whole_new
            d["run_file"] = run_file  # optional: store corrected path

            out_path = path if overwrite else os.path.join(out_dir, fname)
            with open(out_path, "wb") as f:
                pickle.dump(d, f)

            print(f"[OK] {fname}: updated fc_whole -> {fc_whole_new.shape}")

        except Exception as e:
            print(f"[FAIL] {fname}: {e}")

def main():
    recompute_fc_whole_only(
        fc_dir="/cluster/home/herminea/mental_health_project/workspace/results/fmri_prep/vlmd/fc",
        tr=0.8,
        overwrite=True,
    )
    recompute_fc_whole_only(
        fc_dir="/cluster/home/herminea/mental_health_project/workspace/results/fmri_prep/mvmd/fc",
        tr=0.8,
        overwrite=True,
    )

if __name__ == "__main__":
    main()