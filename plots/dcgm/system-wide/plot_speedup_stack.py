import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_parquet_folder(input_dir, max_node_hours=100):
    """Read and concatenate all parquet files (job_id-indexed) in a folder."""
    input_dir = Path(input_dir)
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {input_dir}")

    files = sorted(input_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No .parquet files found in {input_dir}")

    frames = []
    for f in files:
        df = pd.read_parquet(f)
        for col in ("speedup", "node_hours"):
            if col not in df.columns:
                raise ValueError(f"{f} is missing required column: {col!r}")
        frames.append(df)

    combined = pd.concat(frames)  # preserve the job_id index

    dupes = combined.index.duplicated()
    if dupes.any():
        print(f"Warning: {dupes.sum()} duplicate job_id(s) across files; keeping first occurrence.")
        combined = combined[~combined.index.duplicated(keep="first")]

    # Filter out rows with node_hours over the threshold.
    before = len(combined)
    combined = combined[combined["node_hours"] <= max_node_hours]
    dropped = before - len(combined)
    if dropped:
        print(f"Filtered out {dropped} row(s) with node_hours > {max_node_hours}.")

    print(f"Loaded {len(files)} file(s), {len(combined)} job_id(s) after filtering.")
    return combined


def _draw_panel(
    ax, result_df, gpu_name, color, edgecolor, bins, density, show_xlabel, legend_label=None
):
    """Draw a single speedup histogram + cumulative % panel onto `ax`.

    Returns the twin axis (ax2) so the caller can align/clean it up.
    """
    df = result_df[["speedup", "node_hours"]].dropna()
    if df.empty:
        raise ValueError("No valid (non-NaN) rows to plot.")
    s = df["speedup"].to_numpy()

    weights = df["node_hours"].to_numpy()
    total = weights.sum()
    if total <= 0:
        raise ValueError("Sum of node_hours is non-positive; cannot compute cumulative %.")

    ax2 = ax.twinx()

    hist_label = legend_label if legend_label is not None else gpu_name

    counts, bin_edges, patches = ax.hist(
        s,
        bins=bins,
        weights=weights,
        edgecolor=edgecolor,
        color=color,
        density=density,
        label=hist_label,
        histtype="stepfilled",
        alpha=0.5,
        linewidth=2,
    )

    print(gpu_name)
    print(counts)
    print(bin_edges)

    order = np.argsort(s)
    cum = 100.0 * np.cumsum(weights[order]) / total
    ax2.plot(
        s[order],
        cum,
        linestyle=(0, (5, 1)),
        linewidth=2,
        color="mediumorchid",
        label=f"{gpu_name} cumulative %",
    )

    # X label only on the bottom panel.
    if show_xlabel:
        ax.set_xlabel("Speedup Relative to A100", fontsize=19)
    ax.set_ylabel("PDF (weighted by node-hours)" if density else "Node-hours", fontsize=19)
    ax2.set_ylabel("Cumulative percentage (%)", fontsize=17)
    ax2.set_ylim(0, 105)

    # Fix x-axis range and ticks: 0 to 3, every 0.2.
    # ax.set_xlim(0.9, 3.1)
    ax.set_xlim(-0.1, 4.1)
    # ax.set_xticks(np.arange(1, 3 + 0.01, 0.2))
    ax.set_xticks(np.arange(0, 4 + 0.01, 0.2))
    # ax.set_ylim(0, 700000)
    ax.set_ylim(0, 150000)
    ax.tick_params(axis="x", length=6)

    # Vertical split line at speedup == 1.0 (drawn on twin so it's above bars).
    ax2.axvline(x=1.0, color="dimgray", linestyle="dashed", linewidth=2, zorder=5)

    ax.tick_params(which="both", direction="in", labelsize=18)
    ax2.tick_params(which="both", direction="in", labelsize=18)

    # Legend to identify which GPU each panel corresponds to (since the x-label
    # is shared). Draw it on ax2 (the top-most twin axis) so it renders above
    # the x=1.0 vertical line, which is also on ax2.
    handles, labels = ax.get_legend_handles_labels()
    leg = ax2.legend(
        handles,
        labels,
        loc="upper left",
        fontsize=12,
        frameon=True,  # show the box
        framealpha=1.0,  # opaque so the line doesn't show through
        edgecolor="black",  # box border color
        facecolor="white",  # box fill
    )
    leg.set_zorder(20)  # above the axvline (zorder=5 on ax2)

    # Keep ax transparent so the twin axis (ax2) content shows through.
    ax2.set_zorder(ax.get_zorder() + 1)
    ax2.patch.set_visible(False)

    # Frame (spines) linewidth.
    frame_linewidth = 2
    for spine in ["top", "right", "bottom", "left"]:
        ax.spines[spine].set_linewidth(frame_linewidth)

    return ax2


def plot_speedup_distribution_stacked(
    top_df, top_name, bottom_df, bottom_name, outpath, bins, density=False
):
    """Two stacked panels sharing the x-axis, no vertical gap between them."""
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(13, 10), sharex=True, gridspec_kw={"hspace": 0.0}
    )

    # --- Top panel (e.g. Blackwell-Ultra) ---
    _draw_panel(
        ax_top,
        top_df,
        top_name,
        # color="gold",
        # edgecolor="darkgoldenrod",
        color="palegreen",
        edgecolor="forestgreen",
        bins=bins,
        density=density,
        show_xlabel=False,
    )

    # --- Bottom panel (e.g. H100) ---
    _draw_panel(
        ax_bot,
        bottom_df,
        bottom_name,
        # color="sandybrown",
        # edgecolor="darkorange",
        color="lightskyblue",
        edgecolor="dodgerblue",
        bins=bins,
        density=density,
        show_xlabel=True,
        legend_label="Hypothetical-Blackwell-Ultra\n(Non-GPU Portion Scale Up 4x)",
    )
    # ticks = np.arange(0, 700000, 100000)
    ticks = np.arange(0, 160000, 30000)
    for ax in (ax_top, ax_bot):
        ax.set_yticks(ticks)
        # ax.set_ylim(0, 700000)
        ax.set_ylim(0, 160000)
    # ax_top.get_yticklabels()[0].set_visible(False)

    fig.savefig(outpath, dpi=200, format="png", bbox_inches="tight")
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot two stacked node-hours-weighted speedup distributions "
        "(shared x-axis) from two folders of parquet files."
    )
    parser.add_argument(
        "--top-input-dir", required=True, help="Folder of parquet files for the TOP panel."
    )
    parser.add_argument(
        "--bottom-input-dir", required=True, help="Folder of parquet files for the BOTTOM panel."
    )
    parser.add_argument(
        "--top-gpu-name",
        default="Blackwell-Ultra",
        help="GPU name / label for the top panel (default: Blackwell-Ultra).",
    )
    parser.add_argument(
        "--bottom-gpu-name",
        default="H100",
        help="GPU name / label for the bottom panel (default: H100).",
    )
    parser.add_argument(
        "--outpath",
        default="speedup_distribution_stacked.png",
        help="Path to save the output plot.",
    )
    parser.add_argument(
        "--density",
        action="store_true",
        help="Plot a normalized density instead of raw node-hours.",
    )
    parser.add_argument(
        "--max-node-hours",
        type=float,
        default=720,
        help="Filter out rows with node_hours greater than this value (default: 720).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    top_df = load_parquet_folder(args.top_input_dir, max_node_hours=args.max_node_hours)
    bottom_df = load_parquet_folder(args.bottom_input_dir, max_node_hours=args.max_node_hours)

    plot_speedup_distribution_stacked(
        top_df=top_df,
        top_name=args.top_gpu_name,
        bottom_df=bottom_df,
        bottom_name=args.bottom_gpu_name,
        outpath=args.outpath,
        # bins=np.arange(1, 3.1, 0.05),
        bins=np.arange(0, 4.1, 0.05),
        density=args.density,
    )
    print(f"Plot saved to {args.outpath}")


if __name__ == "__main__":
    main()
