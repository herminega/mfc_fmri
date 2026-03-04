"""
utils/modeling/mixed_models.py
--------------------------------
Mixed linear model pipeline for FC–spectral amplitude coupling analysis.

All functions that pool across networks assume a subject random intercept
to account for repeated network observations within subjects. For exact
independence, use run_all_networks_parametric (one model per network).
"""

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from utils.config import BANDS
from utils.analysis.basic import perm_test_group_diff
from utils.analysis.spectral.ms_statstical_analysis import (apply_fdr_within_group)
from utils.analysis.functional_connectivity.fc_strength import build_merged_df


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def subject_level_global_table(df_spec, metric, bands, agg="median"):
    """Reduce spectral features to one value per subject per band (median across networks)."""
    d = df_spec[df_spec["band"].isin(bands)].copy()
    return (
        d.groupby(["subject", "group", "band"])[metric]
         .agg(agg)
         .reset_index()
         .rename(columns={metric: "value"})
    )


def prepare_model_data(df_merged, band, metric, control_total_amp=False):
    """
    Filter to one band, log-transform amplitude metrics, grand-mean centre spec.

    Log-transform rationale
    -----------------------
    band_amp and total_amp are right-skewed. log1p compression improves
    linearity and stabilises variance. Verify with plot_linearity_check first.

    Grand-mean centring rationale
    -----------------------------
    Makes the intercept the grand-mean FC and reduces collinearity between
    the main effect of spec_c and its group interaction.

    Returns
    -------
    sub       : prepared DataFrame
    cov_terms : formula fragment string for extra covariates
    """
    sub = df_merged[df_merged["band"] == band].copy()

    if metric in ("band_amp", "total_amp"):
        sub["spec"] = np.log1p(sub["spec"].clip(lower=0))

    sub["spec_c"] = sub["spec"] - sub["spec"].mean()

    cov_terms = ""
    if control_total_amp and metric not in {"total_amp", "rel_band_amp"}:
        if "total_amp" in sub.columns:
            sub["total_amp_log"] = np.log1p(sub["total_amp"].clip(lower=0))
            sub["total_amp_c"]   = sub["total_amp_log"] - sub["total_amp_log"].mean()
            cov_terms = " + total_amp_c"

    return sub, cov_terms


# ---------------------------------------------------------------------------
# Model fitting helpers
# ---------------------------------------------------------------------------

def safe_get(model, key, attr="params"):
    """Safely extract a coefficient or p-value from a fitted statsmodels model."""
    if model is None:
        return np.nan
    return getattr(model, attr).get(key, np.nan)


def fit_mixedlm(formula, data, random_slopes_formula=None):
    """
    Fit a mixed linear model with subject random intercept.

    Parameters
    ----------
    formula               : Wilkinson formula string
    data                  : DataFrame with a 'subject' column
    random_slopes_formula : e.g. "~spec_c" adds a random slope for spec_c.
                            None (default) = random intercept only.
    """
    try:
        return smf.mixedlm(
            formula,
            data=data,
            groups=data["subject"],
            re_formula=random_slopes_formula,
        ).fit(reml=False, method="lbfgs", disp=False)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Permutation tests on subject-aggregated spectral features
# ---------------------------------------------------------------------------

def run_subject_level_tests(df_spec, metrics, bands=None, agg="median",
                             n_perm=20000, seed=0):
    """
    Permutation test for group differences in spectral features.

    One value per subject per band (median across networks) — independence holds.
    FDR correction applied within each metric across its bands.

    Returns
    -------
    DataFrame sorted by metric, FDR-corrected p-value, raw p-value.
    """
    if bands is None:
        bands = BANDS

    amp_metrics = {"band_amp", "rel_band_amp", "total_amp",
                   "band_entropy", "band_centroid"}
    rows = []

    for metric in metrics:
        g   = subject_level_global_table(df_spec, metric=metric, bands=bands, agg=agg)
        alt = "less" if metric in amp_metrics else "two-sided"

        for band in bands:
            gb       = g[g["band"] == band]
            x        = gb["value"].values
            y_is_mdd = (gb["group"].values == "MDD")

            obs, p, d, hg = perm_test_group_diff(
                x, y_is_mdd, n_perm=n_perm, seed=seed, alternative=alt
            )
            rows.append({
                "metric":            metric,
                "band":              band,
                "alternative":       alt,
                "mean_MDD":          float(np.nanmean(gb.loc[gb.group == "MDD", "value"])),
                "mean_HC":           float(np.nanmean(gb.loc[gb.group == "HC",  "value"])),
                "diff_MDD_minus_HC": float(obs) if not np.isnan(obs) else np.nan,
                "p_perm":            float(p)   if not np.isnan(p)   else np.nan,
                "cohen_d":           float(d)   if not np.isnan(d)   else np.nan,
                "hedges_g":          float(hg)  if not np.isnan(hg)  else np.nan,
                "n_MDD":             int(np.sum(gb.group == "MDD")),
                "n_HC":              int(np.sum(gb.group == "HC")),
            })

    res = pd.DataFrame(rows)
    res = apply_fdr_within_group(res, "p_perm", "q_fdr_metricwise", group_col="metric")
    return res.sort_values(["metric", "q_fdr_metricwise", "p_perm"],
                           na_position="last").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Global mixed model suite (networks pooled)
# ---------------------------------------------------------------------------

def run_mixed_model_parametric(df_merged, band, metric,
                                control_total_amp=False,
                                random_slopes=False):
    """
    Fit four mixed models pooling all networks.

    M1: fc ~ spec_c                   (coupling only)
    M2: fc ~ C(group)                 (group only)
    M3: fc ~ C(group) + spec_c        (additive)
    M4: fc ~ C(group) * spec_c        (interaction)

    random_slopes=True adds re_formula="~spec_c", allowing subjects to differ
    in their amplitude–FC coupling slope.
    """
    sub, cov_terms = prepare_model_data(df_merged, band, metric, control_total_amp)
    re_slope       = "~spec_c" if random_slopes else None

    m1 = fit_mixedlm(f"fc_strength ~ spec_c{cov_terms}",            sub, re_slope)
    m2 = fit_mixedlm(f"fc_strength ~ C(group){cov_terms}",          sub)
    m3 = fit_mixedlm(f"fc_strength ~ C(group) + spec_c{cov_terms}", sub, re_slope)
    m4 = fit_mixedlm(f"fc_strength ~ C(group) * spec_c{cov_terms}", sub, re_slope)

    coef = "C(group)[T.MDD]:spec_c"

    return {
        "metric":           metric,
        "band":             band,
        "n_obs":            len(sub),
        "n_subjects":       sub["subject"].nunique(),
        "random_slopes":    random_slopes,
        "beta_spec":        safe_get(m1, "spec_c"),
        "p_spec":           safe_get(m1, "spec_c",          "pvalues"),
        "beta_group":       safe_get(m2, "C(group)[T.MDD]"),
        "p_group":          safe_get(m2, "C(group)[T.MDD]", "pvalues"),
        "beta_group_adj":   safe_get(m3, "C(group)[T.MDD]"),
        "p_group_adj":      safe_get(m3, "C(group)[T.MDD]", "pvalues"),
        "beta_interaction": safe_get(m4, coef),
        "p_interaction":    safe_get(m4, coef,              "pvalues"),
        "AIC_coupling":     m1.aic if m1 else np.nan,
        "AIC_group":        m2.aic if m2 else np.nan,
        "AIC_additive":     m3.aic if m3 else np.nan,
        "AIC_interact":     m4.aic if m4 else np.nan,
        "m_coupling":       m1,
        "m_group":          m2,
        "m_additive":       m3,
        "m_interact":       m4,
    }


def run_mixed_suite_all(df_fc, df_spec, metrics, bands,
                        control_total_amp=False, random_slopes=False):
    """
    Run the mixed model suite for all metric × band combinations.

    Returns a DataFrame with one row per (metric, band) including model
    coefficients, p-values, and FDR-corrected significance flags.
    """
    rows = []
    for metric in metrics:
        df_merged = build_merged_df(df_fc, df_spec, metric)
        for band in bands:
            print(f"  Fitting {metric} | {band} ...", flush=True)
            rows.append(
                run_mixed_model_parametric(
                    df_merged, band, metric,
                    control_total_amp=control_total_amp,
                    random_slopes=random_slopes,
                )
            )

    df = pd.DataFrame(rows)
    df = apply_fdr_within_group(df, "p_spec",        "q_fdr_coupling",    "metric")
    df = apply_fdr_within_group(df, "p_interaction", "q_fdr_interaction", "metric")
    df["sig_coupling"]    = df["q_fdr_coupling"]    < 0.05
    df["sig_interaction"] = df["q_fdr_interaction"] < 0.05
    return df.sort_values(["metric", "band"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Per-network model suite (independence holds exactly)
# ---------------------------------------------------------------------------

def run_single_network_models(dn, net):
    """
    Fit M1/M2/M3 for a single network subset.

    dn must already have spec_c computed, one row per subject.
    Independence holds exactly here.
    """
    m1 = fit_mixedlm("fc_strength ~ spec_c",            dn)
    m2 = fit_mixedlm("fc_strength ~ C(group)",           dn)
    m3 = fit_mixedlm("fc_strength ~ C(group) + spec_c", dn)

    beta_group     = safe_get(m2, "C(group)[T.MDD]")
    beta_group_adj = safe_get(m3, "C(group)[T.MDD]")
    attenuation    = (
        (1 - beta_group_adj / beta_group) * 100
        if (not np.isnan(beta_group) and beta_group != 0)
        else np.nan
    )

    return {
        "network":        net,
        "beta_spec":      safe_get(m1, "spec_c"),
        "p_spec":         safe_get(m1, "spec_c",          "pvalues"),
        "beta_group":     beta_group,
        "p_group":        safe_get(m2, "C(group)[T.MDD]", "pvalues"),
        "beta_group_adj": beta_group_adj,
        "p_group_adj":    safe_get(m3, "C(group)[T.MDD]", "pvalues"),
        "attenuation_%":  attenuation,
        "AIC_coupling":   m1.aic if m1 else np.nan,
        "AIC_group":      m2.aic if m2 else np.nan,
        "AIC_additive":   m3.aic if m3 else np.nan,
    }


def run_all_networks_parametric(df_merged, band, metric):
    """
    Run per-network mixed model suite across all networks.

    FDR correction applied across networks for each p-value column.
    """
    sub = df_merged[df_merged["band"] == band].copy()

    if metric in ("band_amp", "total_amp"):
        sub["spec"] = np.log1p(sub["spec"].clip(lower=0))

    rows = []
    for net in sorted(sub["network"].unique()):
        dn           = sub[sub["network"] == net].copy()
        dn["spec_c"] = dn["spec"] - dn["spec"].mean()
        rows.append(run_single_network_models(dn, net))

    df = pd.DataFrame(rows)
    df = apply_fdr_within_group(df, "p_spec",      "q_spec",      group_col=None)
    df = apply_fdr_within_group(df, "p_group",     "q_group",     group_col=None)
    df = apply_fdr_within_group(df, "p_group_adj", "q_group_adj", group_col=None)
    df["spec_improves_model"] = df["AIC_additive"] < df["AIC_group"]

    return df.sort_values("attenuation_%", ascending=False).reset_index(drop=True)
