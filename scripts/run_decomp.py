#!/usr/bin/env python3
"""
scripts/run_decomp.py
----------------------
Run VLMD or MVMD decomposition on multiple subjects and save the IMFs/frequencies.
"""

import os, sys

# --- Add parent directory so we can import from utils ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pickle
import numpy as np
from utils.helpers import setup_logger
from utils.io.io_data import load_subject_list, find_bold_files, load_bold_matrix, load_roi_timeseries_runs
from utils.preparation.decomp import run_decomposition_pipeline
from utils.preparation.filters import filter_imfs_by_freq

def normalize_id(s):
    s = s.replace("sub-", "").replace("_", "").upper()
    if not s.startswith("NDARINV"):
        s = s.replace("NDAR", "NDARINV")
    return s


def run_decomp_batch(
    data_path,
    mdd_list,
    hc_list,
    save_dir="results/decompositions/vlmd",
    method="vlmd",
    tr=0.8,
    n_modes=8,
    alpha=1800,
    num_latents=96,
    fmin=0.005,
    fmax=0.25,
    n_per_group=None,
    log_path="logs/vlmd_decomp.log",
):
    """Run decomposition for multiple subjects."""
    os.makedirs(save_dir, exist_ok=True)
    fs = 1 / tr
    logger = setup_logger("vlmd_decomp", log_path)

    # ----------------------------
    # Load subject lists
    # ----------------------------
    mdd_subjects = load_subject_list(mdd_list)
    hc_subjects = load_subject_list(hc_list)

    valid_mdd = [s for s in mdd_subjects]
    valid_hc = [s for s in hc_subjects]

    if n_per_group:
        subjects = valid_mdd[:n_per_group] + valid_hc[:n_per_group]
    else:
        subjects = valid_mdd + valid_hc

    # ----------------------------
    # Find parcellated H5 files
    # ----------------------------
    bold_data = {
        os.path.basename(f).replace("_roi_timeseries.h5", ""): os.path.join(data_path, f)
        for f in sorted(os.listdir(data_path))
        if f.endswith("_roi_timeseries.h5")
    }
    # Preprocess all .h5 keys to a normalized form
    bold_data_norm = {normalize_id(k): v for k, v in bold_data.items()}

    logger.info(f"Running {method.upper()} decomposition for {len(subjects)} subjects.")
    logger.info(f"MDD: {len(valid_mdd)} valid | HC: {len(valid_hc)} valid")

    # ----------------------------
    # Loop over subjects
    # ----------------------------
    for subj in subjects:
        label = "MDD" if subj in mdd_subjects else "HC"
        subj_norm = normalize_id(subj)

        if subj_norm not in bold_data_norm:
            logger.warning(f"No parcellated file found for {subj}, skipping.")
            continue

        subj_file = bold_data_norm[subj_norm]
        runs_dict = load_roi_timeseries_runs(subj_file)
        
        for run_idx, (run_name, X) in enumerate(runs_dict.items(), start=1):
            logger.info(f"[{subj}] Processing {run_name} with shape {X.shape}")
            save_path = os.path.join(save_dir, f"{subj}_run{run_idx}.pkl")

            # Demean each ROI time series
            X = X - X.mean(axis=1, keepdims=True)   # mean over time for each ROI
            print("mean over time per ROI (should be ~0):", np.abs(X.mean(axis=1)).mean())
            
            if os.path.exists(save_path):
                logger.info(f"Skipping existing {save_path}")
                continue

            try:
                logger.info(f"[{subj}] BEFORE decomp: X shape {X.shape}, mean(abs(mean over axis1))={np.mean(np.abs(X.mean(axis=1))):.4g}, mean(abs(mean over axis0))={np.mean(np.abs(X.mean(axis=0))):.4g}")
                # ----------------------------
                # Run decomposition
                # ----------------------------
                if method.lower() != "mvmd":
                    imfs, freqs = run_decomposition_pipeline(
                        X, method="vlmd", n_modes=n_modes, fs=fs,
                        alpha=alpha, num_latents=num_latents
                    )
                    imfs, freqs = filter_imfs_by_freq(imfs, freqs, fmin=fmin, fmax=fmax)
                else:
                    imfs, freqs = run_decomposition_pipeline(
                        X, method="mvmd", n_modes=n_modes, fs=fs,
                        alpha=alpha, num_latents=num_latents
                    )

                # ----------------------------
                # Save results
                # ----------------------------
                result = {
                    "subject": subj,
                    "group": label,
                    "run_idx": run_idx,
                    "run_name": run_name,
                    "run_file": subj_file,
                    "imfs": imfs,
                    "freqs": freqs,
                    "params": dict(TR=tr, FS=fs, N_MODES=n_modes,
                                   ALPHA=alpha, NUM_LATENTS=num_latents),
                }

                with open(save_path, "wb") as f:
                    pickle.dump(result, f)
                logger.info(f"Saved decomposition → {save_path}")

            except Exception as e:
                logger.exception(f"Error in {subj}, run {run_idx}: {e}")  # <-- logs full traceback



def main():
    """Run both VLMD and MVMD decompositions sequentially in one job."""

    BASE_DIR = "/cluster/home/herminea/mental_health_project/workspace"
    DATA_PATH = f"{BASE_DIR}/data/roi_timeseries/"
    MDD_LIST = f"{BASE_DIR}/data/subjects_lists/mdd_subjects_matched.txt"
    HC_LIST  = f"{BASE_DIR}/data/subjects_lists/hc_subjects_matched.txt"

    # --- Run VLMD ---
    # run_decomp_batch(
    #     data_path=DATA_PATH,
    #     mdd_list=MDD_LIST,
    #     hc_list=HC_LIST,
    #     save_dir=f"{BASE_DIR}/results/fmri_prep/vlmd/imfs_new",
    #     method="vlmd",
    #     n_per_group=None,
    #     log_path=f"{BASE_DIR}/jobs/logs/vlmd_decomp.log"
    # )

    # --- Run MVMD ---
    run_decomp_batch(
        data_path=DATA_PATH,
        mdd_list=MDD_LIST,
        hc_list=HC_LIST,
        save_dir=f"{BASE_DIR}/results/fmri_prep/mvmd/imfs_new",
        method="mvmd",
        n_per_group=None,
        log_path=f"{BASE_DIR}/jobs/logs/mvmd_decomp.log"
    )

if __name__ == "__main__":
    main()

