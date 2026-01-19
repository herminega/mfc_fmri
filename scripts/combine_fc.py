#!/usr/bin/env python3
"""
scripts/combine_fc_runs.py
--------------------------
Combine per-run FC matrices into a single subject-level FC (averaged across runs).
"""

import os, sys

# --- Add parent directory so we can import from utils ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pickle
import numpy as np
from utils.io_results import load_results, group_results_by_subject
from utils.metrics import fisher_z_mean

def combine_fc_matrices(
    src_dir=None,
    dest_dir=None,
):
    os.makedirs(dest_dir, exist_ok=True)

    print(f"\nCombining FC matrices across runs in {src_dir}...\n")

    # Load all .pkl results
    results = load_results(src_dir)
    subjects = group_results_by_subject(results)

    for subj, runs in subjects.items():
        print(f"Subject {subj}: {len(runs)} runs")

        fc_modes_list = []
        fc_whole_list = []
        freqs = runs[0]["freqs"]
        group = runs[0].get("group", "Unknown")

        for r in runs:
            fc_modes_list.append(r["fc_modes"])
            if r["fc_whole"] is not None:
                fc_whole_list.append(r["fc_whole"])

        # Ensure same number of IMFs across runs
        K = min(fc.shape[0] for fc in fc_modes_list)
        fc_modes_list = [fc[:K] for fc in fc_modes_list]

        # Average across runs (Fisher z-space)
        fc_modes_combined = np.stack([
            fisher_z_mean([run[k] for run in fc_modes_list])
            for k in range(K)
        ])

        fc_whole_combined = fisher_z_mean(fc_whole_list) if fc_whole_list else None

        result = {
            "subject": subj,
            "group": group,
            "freqs": freqs[:K],
            "fc_modes": fc_modes_combined,
            "fc_whole": fc_whole_combined,
            "n_runs": len(runs),
        }

        dest_path = os.path.join(dest_dir, f"{subj}_combined_fc.pkl")
        with open(dest_path, "wb") as f:
            pickle.dump(result, f)
        print(f"→ Saved combined FC to {dest_path}")

    print("\nAll subjects combined successfully.\n")


def main():
    """Main entrypoint for script execution."""
    combine_fc_matrices(
        src_dir="/cluster/home/herminea/mental_health_project/test/results/fmri_prep/vlmd/fc",
        dest_dir="/cluster/home/herminea/mental_health_project/test/results/fmri_prep/vlmd/combined_fc",
    )
    combine_fc_matrices(
        src_dir="/cluster/home/herminea/mental_health_project/test/results/fmri_prep/mvmd/fc",
        dest_dir="/cluster/home/herminea/mental_health_project/test/results/fmri_prep/mvmd/combined_fc",
    )

if __name__ == "__main__":
    main()
