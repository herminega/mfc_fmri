
import matplotlib.pyplot as plt
import seaborn as sns
import os

from utils.plot.plot_base import set_mpl_style
set_mpl_style()

def save_pdf_png(fig, outpath_no_ext, dpi=300):
    os.makedirs(os.path.dirname(outpath_no_ext), exist_ok=True)
    fig.savefig(outpath_no_ext + ".pdf", bbox_inches="tight")
    fig.savefig(outpath_no_ext + ".png", dpi=dpi, bbox_inches="tight")

def plot_similarity_by_imf(summary_imf, outpath_no_ext=None, title=None):
    # Ensure IMF is treated as ordered categorical on x-axis
    x = summary_imf["IMF"].astype(str).values

    fig, ax = plt.subplots(figsize=(6.5, 3.6), layout="constrained")

    ax.plot(x, summary_imf["MDD–MDD"], "o-", label="MDD–MDD")
    ax.plot(x, summary_imf["HC–HC"],   "o-", label="HC–HC")
    ax.plot(x, summary_imf["MDD–HC"],  "o-", label="MDD–HC")

    ax.set_xlabel("IMF index")
    ax.set_ylabel("Mean similarity")

    if title is None:
        title = "FC similarity across subjects by IMF"
    ax.set_title(title, pad=14)

    # Legend above, avoids covering data
    ax.legend(frameon=False, ncols=3, loc="upper center", bbox_to_anchor=(0.5, 0.98))

    ax.grid(alpha=0.15)

    if outpath_no_ext is not None:
        save_pdf_png(fig, outpath_no_ext)

    plt.show()
    return fig, ax


def plot_imf_frequency_distribution(df, outpath_no_ext=None):
    sns.set_theme(context="paper", style="whitegrid")

    fig, ax = plt.subplots(figsize=(6.5, 3.2), layout="constrained")

    sns.boxplot(
        data=df,
        x="IMF",
        y="Frequency",
        hue="Group",
        ax=ax,
        width=0.6,
        linewidth=1,
        fliersize=2
    )

    ax.set_yscale("log")
    ax.set_xlabel("IMF index")
    ax.set_ylabel("Center frequency (Hz)")
    ax.set_title("IMF center-frequency distribution", pad=10)

    # ---- reserve vertical space INSIDE axes for legend ----
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax * 1.6)   # creates empty space at the top

    # ---- legend INSIDE, clean, non-overlapping ----
    ax.legend(
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(0.02, 0.98),  # inside axes
        borderaxespad=0.0
    )

    ax.grid(alpha=0.15, which="both")

    if outpath_no_ext is not None:
        save_pdf_png(fig, outpath_no_ext)

    plt.show()
    return fig, ax




def plot_similarity_by_band(summary_band, outpath_no_ext=None, title=None):
    # keep band order as provided
    x = summary_band["Band"].astype(str).values

    fig, ax = plt.subplots(figsize=(6.5, 3.6), layout="constrained")

    ax.plot(x, summary_band["MDD–MDD"], "o-", label="MDD–MDD")
    ax.plot(x, summary_band["HC–HC"],   "o-", label="HC–HC")
    ax.plot(x, summary_band["MDD–HC"],  "o-", label="MDD–HC")

    ax.set_xlabel("Frequency band")
    ax.set_ylabel("Mean similarity")

    if title is None:
        title = "FC similarity across subjects by frequency band"
    ax.set_title(title, pad=14)

    ax.legend(frameon=False, ncols=3, loc="upper center", bbox_to_anchor=(0.5, 0.98))
    ax.grid(alpha=0.15)

    if outpath_no_ext is not None:
        save_pdf_png(fig, outpath_no_ext)

    plt.show()
    return fig, ax