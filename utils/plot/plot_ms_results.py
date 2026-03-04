"""
utils/plot/plot_ms_results.py
------------------------------
Display and plot helpers for mixed model and per-network results
from the FC–spectral amplitude coupling pipeline.
"""

import matplotlib.pyplot as plt


def print_suite_results(df_suite):
    """Print the global mixed model suite results as a formatted table."""
    cols = [
        "metric", "band", "n_obs", "n_subjects",
        "beta_spec", "p_spec", "q_fdr_coupling", "sig_coupling",
        "beta_group", "p_group",
        "beta_group_adj", "p_group_adj",
        "beta_interaction", "p_interaction", "q_fdr_interaction", "sig_interaction",
        "AIC_coupling", "AIC_additive", "AIC_interact",
    ]
    present = [c for c in cols if c in df_suite.columns]
    print(df_suite[present].to_string(index=False, float_format=lambda x: f"{x:.4g}"))


def print_network_summary_table(df_net, band, metric, alpha=0.05):
    """
    Print a formatted per-network summary table.

    Parameters
    ----------
    df_net  : output of run_all_networks_parametric
    band    : frequency band label (for header)
    metric  : spectral metric name (for header)
    alpha   : significance threshold for FDR annotation
    """
    print(f"\n{'='*92}")
    print(f"All-network summary | {band} | {metric}")
    print(f"{'='*92}")
    print(f"{'Network':<15} {'beta_spec':>9} {'q_spec':>8} "
          f"{'beta_grp':>9} {'q_grp':>8} "
          f"{'beta_adj':>9} {'q_adj':>8} "
          f"{'atten%':>8} {'spec+AIC':>9}")
    print("-" * 92)
    for _, r in df_net.iterrows():
        s_sig = "**" if r["q_spec"]      < alpha else ("*" if r["p_spec"]      < alpha else "  ")
        g_sig = "**" if r["q_group"]     < alpha else ("*" if r["p_group"]     < alpha else "  ")
        a_sig = "**" if r["q_group_adj"] < alpha else ("*" if r["p_group_adj"] < alpha else "  ")
        imp   = "YES" if r["spec_improves_model"] else "no"
        print(f"  {r['network']:<13} "
              f"{r['beta_spec']:>9.3f} {r['q_spec']:>7.3f}{s_sig} "
              f"{r['beta_group']:>9.3f} {r['q_group']:>7.3f}{g_sig} "
              f"{r['beta_group_adj']:>9.3f} {r['q_group_adj']:>7.3f}{a_sig} "
              f"{r['attenuation_%']:>8.1f} {imp:>9}")
    print("\n  ** FDR<0.05   * uncorrected p<0.05")
    print("  atten%   = % change in group FC beta after spectral amplitude control")
    print("  spec+AIC = YES if additive model fits better than group-only (AIC)")


def plot_all_networks_summary(df_net, band, metric):
    """
    Two-panel summary plot for per-network mixed model results.

    Left  : horizontal bar chart of attenuation % per network.
    Right : scatter of raw vs amplitude-adjusted group FC betas,
            coloured by attenuation %.

    Parameters
    ----------
    df_net  : output of run_all_networks_parametric
    band    : frequency band label (for titles)
    metric  : spectral metric name (for titles)
    """
    df     = df_net.copy().sort_values("attenuation_%", ascending=True)
    colors = ["tomato" if v > 20 else "steelblue" for v in df["attenuation_%"]]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    ax.barh(df["network"], df["attenuation_%"], color=colors)
    ax.axvline(0,  color="black", linewidth=0.8)
    ax.axvline(20, color="grey",  linewidth=0.8, linestyle="--", label="20% threshold")
    ax.set_xlabel("Attenuation of group FC effect\nafter spectral amplitude control (%)")
    ax.set_title(
        f"How much does spectral amplitude\nexplain FC group difference?\n{band} | {metric}"
    )
    ax.legend(fontsize=8)

    ax = axes[1]
    sc = ax.scatter(
        df["beta_group"], df["beta_group_adj"],
        c=df["attenuation_%"], cmap="RdYlBu_r",
        s=80, zorder=3, vmin=-20, vmax=60,
    )
    plt.colorbar(sc, ax=ax, label="Attenuation %")
    lims = [
        min(df["beta_group"].min(), df["beta_group_adj"].min()) - 0.02,
        max(df["beta_group"].max(), df["beta_group_adj"].max()) + 0.02,
    ]
    ax.plot(lims, lims, "k--", alpha=0.4, label="No attenuation")
    ax.axhline(0, color="grey", linewidth=0.5)
    ax.axvline(0, color="grey", linewidth=0.5)
    for _, row in df.iterrows():
        ax.annotate(
            row["network"],
            (row["beta_group"], row["beta_group_adj"]),
            fontsize=7, xytext=(4, 2), textcoords="offset points",
        )
    ax.set_xlabel("Group FC difference (unadjusted beta)")
    ax.set_ylabel("Group FC difference (adjusted for spec amplitude beta)")
    ax.set_title(f"FC group effect: raw vs amplitude-adjusted\n{band} | {metric}")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.show()