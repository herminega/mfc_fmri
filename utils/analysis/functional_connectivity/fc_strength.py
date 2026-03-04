"""
utils/functional_connectivity/fc_strength.py
---------------------------------------------
Compute per-network FC strength and build merged DataFrames for modeling.
"""

import numpy as np
import pandas as pd


def fc_strength_per_subject(subj_fc, net_names):
    """
    Mean off-diagonal FC strength per network row for one subject.

    Parameters
    ----------
    subj_fc   : (N, N) array – network-level Fisher-z FC matrix
    net_names : list[str]    – ordered network names

    Returns
    -------
    DataFrame with columns: network, fc_strength
    """
    mat = np.asarray(subj_fc, float).copy()
    np.fill_diagonal(mat, np.nan)
    return pd.DataFrame({
        "network":     net_names,
        "fc_strength": np.nanmean(mat, axis=1),
    })


def build_fc_strength_df(z_binned_netfcs, groups, net_names, bands):
    """
    Build a long-format DataFrame of per-network FC strength.

    Parameters
    ----------
    z_binned_netfcs : dict  band -> subject -> (N, N) Fisher-z matrix
    groups          : dict  subject -> "MDD" | "HC"
    net_names       : list[str]
    bands           : list[str]  – ordered band names to include

    Returns
    -------
    DataFrame with columns: network, fc_strength, subject, group, band
    """
    rows = []
    for band in bands:
        if band not in z_binned_netfcs:
            continue
        for subj, mat in z_binned_netfcs[band].items():
            d            = fc_strength_per_subject(mat, net_names)
            d["subject"] = subj
            d["group"]   = groups[subj]
            d["band"]    = band
            rows.append(d)
    return pd.concat(rows, ignore_index=True)


def compute_network_strength_df(z_binned_netfcs, groups, net_names):
    """
    Compute mean off-diagonal FC strength per (subject, group, band, network).

    Equivalent to build_fc_strength_df but uses the column name 'strength'
    (used by the statistical_analysis pipeline).

    Returns
    -------
    DataFrame with columns: subject, group, band, network, strength
    """
    rows = []
    for band, subj_dict in z_binned_netfcs.items():
        for subj, mat in subj_dict.items():
            group = groups[subj]
            for i, net in enumerate(net_names):
                vals = np.delete(mat[i, :], i)
                vals = vals[np.isfinite(vals)]
                if len(vals) == 0:
                    continue
                rows.append({
                    "subject":  subj,
                    "group":    group,
                    "band":     band,
                    "network":  net,
                    "strength": np.mean(vals),
                })
    return pd.DataFrame(rows)


def build_merged_df(df_fc, df_spec, metric):
    """
    Merge FC strength with spectral features at network level.

    Band-invariant metrics (total_*) are merged without the band key
    to avoid spurious replication across bands.

    Parameters
    ----------
    df_fc   : long-format FC strength DataFrame
    df_spec : long-format spectral features DataFrame
    metric  : str – column name in df_spec to merge

    Returns
    -------
    Merged DataFrame containing both fc_strength and spec columns
    """
    band_invariant = {"total_amp", "total_entropy", "total_centroid"}

    if metric in band_invariant:
        spec_small = (
            df_spec[["subject", "group", "network", metric]]
            .drop_duplicates(["subject", "group", "network"])
            .rename(columns={metric: "spec"})
        )
        return df_fc.merge(spec_small, on=["subject", "group", "network"], how="inner")
    else:
        spec_small = (
            df_spec[["subject", "group", "band", "network", metric]]
            .rename(columns={metric: "spec"})
        )
        return df_fc.merge(
            spec_small, on=["subject", "group", "band", "network"], how="inner"
        )
