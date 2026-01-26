"""
utils/metrics.py
----------------
Connectivity and decomposition metrics.
"""
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform


def flatten_fc(fc):
    iu = np.triu_indices(fc.shape[0], 1)
    return fc[iu]

def fc_similarity(fc1, fc2):
    v1, v2 = flatten_fc(fc1), flatten_fc(fc2)
    return 1 - pdist(np.vstack([v1, v2]), metric='correlation')[0]


def compute_similarity_matrix(fcs):
    subs = list(fcs.keys())
    X = np.array([flatten_fc(fcs[s]) for s in subs])
    sim = 1 - squareform(pdist(X, metric='correlation'))
    return sim, subs

def fisher_z_to_r(z):
    """Convert Fisher z back to correlation."""
    return np.tanh(z)


def fisher_z_mean(fc_list):
    """Mean Fisher-z FC across runs."""
    z_stack = np.stack(fc_list)
    return np.nanmean(z_stack, axis=0)


def group_similarity_means(sim, subs, groups):
    mdd_idx = [i for i, s in enumerate(subs) if groups[s] == "MDD"]
    hc_idx  = [i for i, s in enumerate(subs) if groups[s] == "HC"]

    MDD_MDD = np.nanmean(sim[np.ix_(mdd_idx, mdd_idx)][~np.eye(len(mdd_idx), dtype=bool)])
    HC_HC   = np.nanmean(sim[np.ix_(hc_idx, hc_idx)][~np.eye(len(hc_idx), dtype=bool)])
    MDD_HC  = np.nanmean(sim[np.ix_(mdd_idx, hc_idx)])
    return MDD_MDD, HC_HC, MDD_HC

def permutation_test_within_vs_between(sim, subs, groups, n_perm=5000, seed=42):
    rng = np.random.default_rng(seed)
    mdd_idx = [i for i, s in enumerate(subs) if groups[s] == "MDD"]
    hc_idx  = [i for i, s in enumerate(subs) if groups[s] == "HC"]

    within = np.concatenate([
        sim[np.ix_(mdd_idx, mdd_idx)][~np.eye(len(mdd_idx), dtype=bool)],
        sim[np.ix_(hc_idx, hc_idx)][~np.eye(len(hc_idx), dtype=bool)]
    ])
    between = sim[np.ix_(mdd_idx, hc_idx)].ravel()
    true_diff = np.mean(within) - np.mean(between)

    all_idx = np.arange(len(subs))
    n_mdd = len(mdd_idx)
    perm_diffs = np.empty(n_perm)
    for k in range(n_perm):
        rng.shuffle(all_idx)
        MDD_p, HC_p = all_idx[:n_mdd], all_idx[n_mdd:]
        within_p = np.concatenate([
            sim[np.ix_(MDD_p, MDD_p)][~np.eye(len(MDD_p), dtype=bool)],
            sim[np.ix_(HC_p, HC_p)][~np.eye(len(HC_p), dtype=bool)]
        ])
        between_p = sim[np.ix_(MDD_p, HC_p)].ravel()
        perm_diffs[k] = np.mean(within_p) - np.mean(between_p)

    pval = np.mean(np.abs(perm_diffs) >= np.abs(true_diff))
    return true_diff, pval

def analyze_similarity_by_imf(fcs_by_imf, groups):
    results = []
    for imf_idx, fcs in fcs_by_imf.items():
        sim, subs = compute_similarity_matrix(fcs)
        MDD_MDD, HC_HC, MDD_HC = group_similarity_means(sim, subs, groups)
        diff, pval = permutation_test_within_vs_between(sim, subs, groups)
        results.append((imf_idx, MDD_MDD, HC_HC, MDD_HC, diff, pval))
    return pd.DataFrame(results, columns=["IMF", "MDD–MDD", "HC–HC", "MDD–HC", "Δ", "p"])
