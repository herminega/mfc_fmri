# utils/functional_connectivity/fc_statistical_analysis.py
"""
Statistical analysis of functional connectivity matrices.
Includes permutation tests, effect sizes, FDR correction,
network-level group comparisons, and FC coupling tests.
"""

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import fdrcorrection
from utils.analysis.basic import hedges_g, cohen_d, perm_test_group_diff

# ---------------------------------------------------------------------------
# Helpers (internal)
# ---------------------------------------------------------------------------

def dir_label(x):
    if x > 0:
        return "MDD > HC"
    elif x < 0:
        return "MDD < HC"
    else:
        return "≈ 0"

# Determine the "other" network for each edge
def get_partner(row, seed_network):
    return row["Net2"] if row["Net1"] == seed_network else row["Net1"]


# ---------------------------------------------------------------------------
# Network-pair (edgewise) and band-level tests
# ---------------------------------------------------------------------------

def networkpair_permtest(mdd_mats, hc_mats, n_perm=5000, seed=0):
    rng = np.random.default_rng(seed)
    n_net = mdd_mats.shape[1]
    diff_mat = np.zeros((n_net, n_net))
    p_mat = np.ones((n_net, n_net))
    d_mat = np.zeros((n_net, n_net))
    g_mat = np.zeros((n_net, n_net))

    for i in range(n_net):
        for j in range(i+1, n_net):
            mdd_vals = mdd_mats[:, i, j]
            hc_vals  = hc_mats[:, i, j]

            mdd_vals = mdd_vals[np.isfinite(mdd_vals)]
            hc_vals  = hc_vals[np.isfinite(hc_vals)]
            if len(mdd_vals) < 2 or len(hc_vals) < 2:
                continue

            true_diff = np.mean(mdd_vals) - np.mean(hc_vals)

            combined = np.concatenate([mdd_vals, hc_vals])
            n_mdd = len(mdd_vals)
            perm_diffs = np.empty(n_perm)

            for p in range(n_perm):
                perm = rng.permutation(combined)
                perm_diffs[p] = np.mean(perm[:n_mdd]) - np.mean(perm[n_mdd:])

            p_val = (np.sum(np.abs(perm_diffs) >= np.abs(true_diff)) + 1) / (n_perm + 1)

            d_val = cohen_d(mdd_vals, hc_vals)
            g_val = hedges_g(mdd_vals, hc_vals)

            diff_mat[i, j] = diff_mat[j, i] = true_diff
            p_mat[i, j] = p_mat[j, i] = p_val
            d_mat[i, j] = d_mat[j, i] = d_val
            g_mat[i, j] = g_mat[j, i] = g_val

    return diff_mat, p_mat, d_mat, g_mat

def summarize_network_differences(results_band, level_col="Band", top_n=5):
    levels = results_band[level_col].unique()
    for level in levels:
        df_band = results_band[results_band[level_col] == level].copy()
        df_band = df_band.sort_values("p").reset_index(drop=True)
        print(f"\n=== {level_col} {level} ===")
        for _, row in df_band.head(top_n).iterrows():
            print(f"{row['Net1']:>15} – {row['Net2']:<15}  "
                  f"ΔFC = {row['ΔFC']:+.3f}, p = {row['p']:.4f}, "
                  f"p_FDR = {row['p_FDR']:.4f} "
                  f"Cohen's d = {row['Cohen']:.3f}, "
                  f"Hedges' g = {row['Hedges']:.3f}, "
                  f"({'↑' if row['ΔFC'] > 0 else '↓'} in MDD)")


def compare_band_to_whole_network(summary_df, band, n_perm=10000, random_state=42, alternative="two-sided"):
    rng = np.random.default_rng(random_state)

    wb = summary_df[summary_df["Band"] == "Whole"].copy()
    bd = summary_df[summary_df["Band"] == band].copy()
    merged = wb.merge(bd, on="Network", suffixes=("_Whole", f"_{band}"))

    # magnitude difference (abs): improvement in separation regardless of direction
    diffs = merged[f"ΔFC_mean_{band}"].abs().values - merged["ΔFC_mean_Whole"].abs().values
    T_real = diffs.mean()

    perm_T = np.empty(n_perm)
    for p in range(n_perm):
        signs = rng.choice([-1, 1], size=len(diffs))
        perm_T[p] = np.mean(diffs * signs)

    if alternative == "two-sided":
        p_val = (np.sum(np.abs(perm_T) >= np.abs(T_real)) + 1) / (n_perm + 1)
    elif alternative == "greater":
        p_val = (np.sum(perm_T >= T_real) + 1) / (n_perm + 1)
    elif alternative == "less":
        p_val = (np.sum(perm_T <= T_real) + 1) / (n_perm + 1)
    else:
        raise ValueError("alternative must be 'two-sided', 'greater', or 'less'")

    sd = np.std(diffs, ddof=1)
    cohen_d = T_real / sd if sd > 0 else np.nan
    n = len(diffs)
    J = 1 - 3/(4*n - 9)
    hedges_g = cohen_d * J if np.isfinite(cohen_d) else np.nan

    return {"band": band, "T_real": T_real, "p_perm": p_val, "cohen_d": cohen_d, "hedges_g": hedges_g, "merged": merged}

def compute_slow_network_drivers(summary_df,
                                  whole_label="Whole",
                                  band_label="Slow-5",
                                  value_col="ΔFC_mean"):
    """
    Compute, for each network, how much the magnitude of the group difference
    (|ΔFC_mean|) is larger in Slow-X than in Wholeband.

    Returns a DataFrame sorted from strongest to weakest "driver".
    """

    # 1. Extract Whole and Slow-5 rows
    df_whole = summary_df[summary_df["Band"] == whole_label].copy()
    df_band  = summary_df[summary_df["Band"] == band_label].copy()

    # 2. Merge on Network so each row has Whole + Slow-X values
    merged = df_whole.merge(
        df_band,
        on="Network",
        suffixes=(f"_{whole_label}", f"_{band_label}")
    )

    # Rename for convenience
    col_whole = f"{value_col}_{whole_label}"
    col_band  = f"{value_col}_{band_label}"

    # 3. Compute absolute differences in magnitude
    merged["abs_whole"] = merged[col_whole].abs()
    merged["abs_band"]  = merged[col_band].abs()

    # d_abs > 0 ⇒ stronger effect in Slow-X than Whole
    merged["d_abs"] = merged["abs_band"] - merged["abs_whole"]

    # 4. Add direction info (hypo/hyper in Slow-X and Whole)

    merged["dir_Slow"] = merged[col_band].apply(dir_label)
    merged["dir_Whole"] = merged[col_whole].apply(dir_label)

    # 5. (Optional) z-score of d_abs across networks, for relative importance
    if merged["d_abs"].std(ddof=1) > 0:
        merged["d_abs_z"] = (merged["d_abs"] - merged["d_abs"].mean()) / merged["d_abs"].std(ddof=1)
    else:
        merged["d_abs_z"] = np.nan

    # 6. Sort: strongest drivers first
    merged_sorted = merged.sort_values("d_abs", ascending=False).reset_index(drop=True)

    return merged_sorted

def get_network_edges(results_band, seed_network, band_label="Slow-X"):
    """
    Extract all edges involving a given seed network (e.g. LimbicA)
    in a given band (e.g. Slow-X), from the 'results_band' DataFrame.

    Assumes 'results_band' has columns:
        ['Band', 'Net1', 'Net2', 'ΔFC', 'Cohen', 'Hedges', 'p', 'p_FDR', ...]
    """
    # Filter to band of interest
    df_band = results_band[results_band["Band"] == band_label].copy()

    # Keep only edges where the seed network is involved
    mask = (df_band["Net1"] == seed_network) | (df_band["Net2"] == seed_network)
    df_seed = df_band[mask].copy()

    if df_seed.empty:
        print(f"No edges found for {seed_network} in band {band_label}.")
        return df_seed

    df_seed["Connection"] = df_seed.apply(get_partner, axis=1, seed_network=seed_network)

    # Sort by ΔFC (from most negative to most positive)
    df_seed = df_seed.sort_values("ΔFC").reset_index(drop=True)

    # Optional: nicer column ordering
    cols = ["Connection", "ΔFC", "Band"]
    cols = [c for c in cols if c in df_seed.columns]  # keep only existing
    df_seed = df_seed[cols]

    return df_seed



# ---------------------------------------------------------------------------
# Network-level tests
# ---------------------------------------------------------------------------
def test_network_level(df_strength, n_perm=20000, seed=0):
    """
    Permutation test of MDD vs HC FC strength per (band, network).

    Parameters
    ----------
    df_strength : DataFrame with columns subject, group, band, network, strength

    Returns
    -------
    DataFrame sorted by band and FDR-corrected p-value.
    """
    rows = []
    for band in df_strength["band"].unique():
        df_band = df_strength[df_strength["band"] == band]
        for net in df_band["network"].unique():
            df_net   = df_band[df_band["network"] == net]
            x        = df_net["strength"].values
            y_is_mdd = (df_net["group"].values == "MDD")
            obs, p, d, g = perm_test_group_diff(
                x, y_is_mdd, n_perm=n_perm, seed=seed, alternative="less"
            )
            rows.append({
                "band":    band,
                "network": net,
                "ΔFC_mean": obs,
                "p_perm":  p,
                "cohen_d": d,
                "hedges_g": g,
                "n_MDD":   int(np.sum(y_is_mdd)),
                "n_HC":    int(np.sum(~y_is_mdd)),
            })

    res = pd.DataFrame(rows)
    res["q_fdr_bandwise"] = np.nan
    for band in res["band"].unique():
        mask = res["band"] == band
        _, qvals = fdrcorrection(res.loc[mask, "p_perm"].fillna(1.0).values)
        res.loc[mask, "q_fdr_bandwise"] = qvals

    return res.sort_values(["band", "q_fdr_bandwise"]).reset_index(drop=True)

def print_network_overview(res, alpha=0.05):
    """Print a readable overview of network-level FC group differences."""
    for band in sorted(res["band"].unique()):
        print("\n" + "=" * 60)
        print(f"BAND: {band}")
        print("=" * 60)
        dfb = res[res["band"] == band].copy()
        dfb = dfb.reindex(dfb["cohen_d"].abs().sort_values(ascending=False).index)
        for _, row in dfb.iterrows():
            direction = "↓ MDD" if row["ΔFC_mean"] < 0 else "↑ MDD"
            sig_mark  = ""
            if row["q_fdr_bandwise"] < alpha:
                sig_mark = " **FDR**"
            elif row["p_perm"] < alpha:
                sig_mark = " *p<.05 (unc)*"
            print(
                f"{row['network']:<15} "
                f"Δ={row['ΔFC_mean']:+.3f} | "
                f"d={row['cohen_d']:+.3f} | "
                f"p={row['p_perm']:.4f} | "
                f"q={row['q_fdr_bandwise']:.4f} | "
                f"{direction}{sig_mark}"
            )