
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Effect sizes
# ---------------------------------------------------------------------------

def hedges_g(x1, x2):
    """Compute Hedges' g (bias-corrected Cohen's d) for two independent samples."""
    x1, x2 = np.asarray(x1), np.asarray(x2)
    x1, x2 = x1[~np.isnan(x1)], x2[~np.isnan(x2)]
    if len(x1) < 2 or len(x2) < 2:
        return np.nan
    
    n1, n2 = len(x1), len(x2)
    s1, s2 = np.nanstd(x1, ddof=1), np.nanstd(x2, ddof=1)
    s_pooled = np.sqrt(((n1 - 1)*s1**2 + (n2 - 1)*s2**2) / (n1 + n2 - 2))
    d = (np.nanmean(x1) - np.nanmean(x2)) / s_pooled

    # Correction for small sample bias
    correction = 1 - (3 / (4*(n1 + n2) - 9))
    g = d * correction
    return g


def cohen_d(x1, x2):
    """Compute Cohen's d for two independent samples (handles NaNs)."""
    x1, x2 = np.asarray(x1), np.asarray(x2)
    x1, x2 = x1[~np.isnan(x1)], x2[~np.isnan(x2)]
    if len(x1) < 2 or len(x2) < 2:
        return np.nan
    n1, n2 = len(x1), len(x2)
    s1, s2 = np.nanstd(x1, ddof=1), np.nanstd(x2, ddof=1)
    s_pooled = np.sqrt(((n1 - 1)*s1**2 + (n2 - 1)*s2**2) / (n1 + n2 - 2))
    return (np.nanmean(x1) - np.nanmean(x2)) / s_pooled

# ---------------------------------------------------------------------------
# Permutation test for group differences
# ---------------------------------------------------------------------------


def perm_test_group_diff(x, y_is_mdd, n_perm=20000, seed=0, alternative="two-sided"):
    """
    Permutation test for the difference in group means (MDD − HC).

    Parameters
    ----------
    x           : numeric array (n,)
    y_is_mdd    : boolean array (n,)  True = MDD, False = HC
    n_perm      : number of permutations
    alternative : "two-sided" | "less" (MDD < HC) | "greater" (MDD > HC)

    Returns
    -------
    obs      : observed mean difference (MDD − HC)
    p_perm   : permutation p-value
    cohen_d  : Cohen's d
    hedges_g : Hedges' g (bias-corrected)
    """
    rng   = np.random.default_rng(seed)
    x     = np.asarray(x, float)
    y     = np.asarray(y_is_mdd, bool)
    x_mdd = x[y][~np.isnan(x[y])]
    x_hc  = x[~y][~np.isnan(x[~y])]

    if len(x_mdd) < 5 or len(x_hc) < 5:
        return np.nan, np.nan, np.nan, np.nan

    obs  = float(np.mean(x_mdd) - np.mean(x_hc))
    d    = cohen_d(x_mdd, x_hc)
    g    = hedges_g(x_mdd, x_hc)
    allx = np.concatenate([x_mdd, x_hc])
    n1   = len(x_mdd)

    perm = np.array([
        np.mean(allx[(idx := rng.permutation(len(allx)))][:n1]) -
        np.mean(allx[idx][n1:])
        for _ in range(n_perm)
    ])

    if alternative == "less":
        p = (np.sum(perm <= obs) + 1) / (n_perm + 1)
    elif alternative == "greater":
        p = (np.sum(perm >= obs) + 1) / (n_perm + 1)
    else:
        p = (np.sum(np.abs(perm) >= abs(obs)) + 1) / (n_perm + 1)

    return obs, float(p), d, g