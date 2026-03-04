import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests, fdrcorrection
from utils.analysis.basic import perm_test_group_diff

# ---------------------------------------------------------------------------
# Permutation tests and FDR correction
# ---------------------------------------------------------------------------
def apply_fdr_within_group(df, raw_col, fdr_col, group_col="metric"):
    """
    Apply Benjamini-Hochberg FDR correction to a p-value column.

    If group_col is None: correct the entire column at once.
    Otherwise: correct within each unique value of group_col independently.
    """
    df = df.copy()
    df[fdr_col] = np.nan

    if group_col is None:
        pv = df[raw_col].values
        ok = ~np.isnan(pv)
        q  = np.full_like(pv, np.nan, dtype=float)
        if ok.sum() > 0:
            _, q_ok, _, _ = multipletests(pv[ok], method="fdr_bh")
            q[ok] = q_ok
        df[fdr_col] = q
    else:
        for grp in df[group_col].unique():
            m  = df[group_col] == grp
            pv = df.loc[m, raw_col].values
            ok = ~np.isnan(pv)
            q  = np.full_like(pv, np.nan, dtype=float)
            if ok.sum() > 0:
                _, q_ok, _, _ = multipletests(pv[ok], method="fdr_bh")
                q[ok] = q_ok
            df.loc[m, fdr_col] = q
    return df




def run_network_level_tests(df_spec, metrics, bands, n_perm=10000, seed=0):
    """
    Permutation test of spectral group differences per (network, band, metric).

    Independence holds: one observation per subject per network (no pooling).

    Parameters
    ----------
    df_spec  : long-format spectral features DataFrame
    metrics  : list[str]  – column names to test
    bands    : list[str]  – bands to include
    n_perm   : number of permutations
    seed     : random seed

    Returns
    -------
    DataFrame with columns: network, metric, band, mean_MDD, mean_HC, diff,
                            p_perm, p_fdr, cohen_d, hedges_g
    """
    rows     = []
    networks = sorted(df_spec["network"].dropna().unique())

    for net in networks:
        for metric in metrics:
            for band in bands:
                sub = df_spec[
                    (df_spec["network"] == net) & (df_spec["band"] == band)
                ]
                mdd_vals = sub[sub["group"] == "MDD"][metric].dropna().values
                hc_vals  = sub[sub["group"] == "HC"][metric].dropna().values

                if len(mdd_vals) < 2 or len(hc_vals) < 2:
                    continue

                y_is_mdd = np.array([True]  * len(mdd_vals) +
                                    [False] * len(hc_vals))
                x_all    = np.concatenate([mdd_vals, hc_vals])
                obs, p, d, g = perm_test_group_diff(
                    x_all, y_is_mdd, n_perm=n_perm, seed=seed, alternative="less"
                )
                rows.append({
                    "network":  net,
                    "metric":   metric,
                    "band":     band,
                    "mean_MDD": float(mdd_vals.mean()),
                    "mean_HC":  float(hc_vals.mean()),
                    "diff":     obs,
                    "p_perm":   p,
                    "cohen_d":  d,
                    "hedges_g": g,
                })

    df = pd.DataFrame(rows)
    for (m, b), grp in df.groupby(["metric", "band"]):
        _, q = fdrcorrection(grp["p_perm"].fillna(1.0).values)
        df.loc[grp.index, "p_fdr"] = q
    return df



