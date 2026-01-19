# utils/fc_analysis.py
"""
High-level analyses for functional connectivity data.
Compute within-subject and between-group metrics, similarities, etc.
"""

import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils.metrics import fc_similarity
from scipy.stats import ttest_ind, ttest_ind
from scipy.spatial.distance import pdist, squareform



def compute_within_subject_similarity(fc_results):
    """Compute mean FC similarity across runs for each subject and each mode."""
    subjects = sorted(set(r["subject"] for r in fc_results))
    results = []

    for subj in subjects:
        subj_results = [r for r in fc_results if r["subject"] == subj]
        group = subj_results[0]["group"]
        n_runs = len(subj_results)
        if n_runs < 2:
            continue

        # find the minimum number of modes across runs for this subject
        K_common = min(r["fc_modes"].shape[0] for r in subj_results)
        sim_per_mode = np.zeros(K_common)

        for k in range(K_common):
            pairs = list(itertools.combinations(subj_results, 2))
            sims = []
            for a, b in pairs:
                # compare only up to common size
                if k < a["fc_modes"].shape[0] and k < b["fc_modes"].shape[0]:
                    sims.append(fc_similarity(a["fc_modes"][k], b["fc_modes"][k]))
            sim_per_mode[k] = np.nanmean(sims)

        results.append({
            "subject": subj,
            "group": group,
            "n_runs": n_runs,
            "run_similarity_per_mode": sim_per_mode,
            "mean_similarity": np.nanmean(sim_per_mode)
        })

    return pd.DataFrame(results)



