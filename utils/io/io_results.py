# utils/io_results.py

import os
import pickle
import pandas as pd
from collections import defaultdict

def load_fc_results(results_dir):
    """Load all FC .pkl files into a list of dicts."""
    results = []
    for fname in sorted(os.listdir(results_dir)):
        if not fname.endswith(".pkl"):
            continue
        with open(os.path.join(results_dir, fname), "rb") as f:
            results.append(pickle.load(f))
    print(f"[Load] Loaded {len(results)} FC result files from {results_dir}")
    return results


def group_results_by_subject(results):
    """Group result dicts by subject."""
    subjects_dict = defaultdict(list)
    for r in results:
        subjects_dict[r["subject"]].append(r)
    return subjects_dict


def load_fc_combined_data(res_dir):
    results = load_fc_results(res_dir)
    subjects_combined = group_results_by_subject(results)

    fcs_by_imf = defaultdict(dict)
    groups = {}
    freqs_by_imf = defaultdict(list)

    for subj, entries in subjects_combined.items():
        entry = entries[0]
        groups[subj] = entry["group"]
        fcs_by_imf["whole"][subj] = entry["fc_whole"]
        for i, fc in enumerate(entry["fc_modes"], start=1):
            fcs_by_imf[i][subj] = fc
            freqs_by_imf[i].append(entry["freqs"][i-1])

    return subjects_combined, fcs_by_imf, groups, freqs_by_imf

