# utils/io_results.py

import os
import pickle
import pandas as pd
from collections import defaultdict

def load_results(results_dir):
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
