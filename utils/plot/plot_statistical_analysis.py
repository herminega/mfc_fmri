import matplotlib.pyplot as plt

def plot_network_edges(df_seed, seed_network, band_label="Slow-5"):
    """
    Simple barplot of ΔFC for all edges of a seed network in a given band.
    df_seed is the output of get_network_edges().
    """
    if df_seed.empty:
        return

    plt.figure(figsize=(8, 4))
    df_plot = df_seed.sort_values("ΔFC")
    
    plt.barh(df_plot["Connection"], df_plot["ΔFC"])
    plt.axvline(0, color="k", linewidth=1)
    plt.xlabel("ΔFC (MDD – HC)")
    plt.title(f"{seed_network} connectivity differences ({band_label})")
    plt.tight_layout()
    plt.show()