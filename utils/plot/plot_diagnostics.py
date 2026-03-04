"""
utils/plot/plot_diagnostics.py
------------------------------
Diagnostic plots for the FC–spectral amplitude coupling mixed model pipeline.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.nonparametric.smoothers_lowess import lowess


def plot_linearity_check(df_merged, band, metric, save_path=None):
    """
    Scatter of spec vs fc_strength per group with LOWESS trend lines.

    Run before fitting any models to verify the linearity assumption.
    Non-linearity here indicates log-transform may be needed.

    Parameters
    ----------
    df_merged : merged DataFrame with columns spec, fc_strength, band, group
    band      : frequency band to plot
    metric    : spectral metric name (used for axis label)
    save_path : optional path to save figure
    """
    sub = df_merged[df_merged["band"] == band].copy()
    sub["spec_c"] = sub["spec"] - sub["spec"].mean()
    colors = {"MDD": "tomato", "HC": "steelblue"}

    fig, ax = plt.subplots(figsize=(5, 4))
    for grp, gdf in sub.groupby("group"):
        ax.scatter(gdf["spec_c"], gdf["fc_strength"],
                   alpha=0.3, s=10, color=colors.get(grp, "grey"), label=grp)
        sorted_idx = np.argsort(gdf["spec_c"].values)
        xv = gdf["spec_c"].values[sorted_idx]
        yv = gdf["fc_strength"].values[sorted_idx]
        sm = lowess(yv, xv, frac=0.4)
        ax.plot(sm[:, 0], sm[:, 1], color=colors.get(grp, "grey"), linewidth=2)

    ax.set_xlabel(f"{metric} (centred)")
    ax.set_ylabel("FC strength (Fisher z)")
    ax.set_title(f"Linearity check — {band} | {metric}")
    ax.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_residual_diagnostics(model, title=""):
    """
    QQ plot and fitted-vs-residuals plot for a fitted mixed model.

    Checks
    ------
    QQ plot             : are residuals approximately normally distributed?
    Fitted vs residuals : is variance approximately constant (homoscedastic)?
                          A LOWESS trend far from zero indicates violation.

    Run on the interaction model (M4) after the suite to justify
    the parametric p-values.

    Parameters
    ----------
    model : fitted statsmodels MixedLM result, or None
    title : string appended to each subplot title
    """
    if model is None:
        print("Model is None — skipping diagnostics.")
        return

    resid  = model.resid
    fitted = model.fittedvalues

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    stats.probplot(resid, dist="norm", plot=axes[0])
    axes[0].set_title(f"QQ plot — {title}")

    axes[1].scatter(fitted, resid, alpha=0.3, s=10, color="steelblue")
    axes[1].axhline(0, color="black", linewidth=0.8)
    sm_fit = lowess(resid.values, fitted.values, frac=0.4)
    axes[1].plot(sm_fit[:, 0], sm_fit[:, 1], color="tomato", linewidth=2)
    axes[1].set_xlabel("Fitted values")
    axes[1].set_ylabel("Residuals")
    axes[1].set_title(f"Fitted vs residuals — {title}")

    plt.tight_layout()
    plt.show()