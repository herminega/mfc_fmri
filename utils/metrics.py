"""
utils/metrics.py
----------------
Connectivity and decomposition metrics.
"""

import numpy as np


def fc_similarity(fc1, fc2):
    """Compute upper-triangular correlation similarity between two FC matrices."""
    iu = np.triu_indices_from(fc1, k=1)
    return np.corrcoef(fc1[iu], fc2[iu])[0, 1]


def fisher_z_to_r(z):
    """Convert Fisher z back to correlation."""
    return np.tanh(z)


def fisher_z_mean(fc_list):
    """Mean Fisher-z FC across runs."""
    z_stack = np.stack(fc_list)
    return np.nanmean(z_stack, axis=0)