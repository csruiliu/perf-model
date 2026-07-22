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


def plot_speedup_distribution(
    result_df, gpu_name, outpath, bins=np.arange(0, 3.1, 0.1), density=False
):
    """Node-hours-weighted speedup histogram + cumulative % for one target GPU."""
    df = result_df[["speedup", "node_hours"]].dropna()
    if df.empty:
        raise ValueError("No valid (non-NaN) rows to plot.")
    s = df["speedup"].to_numpy()

    weights = df["node_hours"].to_numpy()
    total = weights.sum()
    if total <= 0:
        raise ValueError("Sum of node_hours is non-positive; cannot compute cumulative %.")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax2 = ax.twinx()
    counts, bin_edges, patches = ax.hist(
        s,
        bins=bins,
        weights=weights,
        # edgecolor="darkgoldenrod",
        # color="gold",
        edgecolor="darkorange",
        color="sandybrown",
        density=density,
        label=gpu_name,
        histtype="stepfilled",
        alpha=0.5,
        linewidth=3,
    )

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

    ax.set_xlabel(f"Workload on {gpu_name} Speedup Relative to A100", fontsize=19)
    ax.set_ylabel("PDF (weighted by node-hours)" if density else "Node-hours", fontsize=19)
    ax2.set_ylabel("Cumulative percentage (%)", fontsize=17)
    ax2.set_ylim(0, 105)

    # Fix x-axis range and ticks: 1 to 3, every 0.2.
    ax.set_xlim(-0.1, 3.1)
    ax.set_xticks(np.arange(0, 3.0 + 0.01, 0.2))
    ax.set_ylim(0, 250000)
    # Hide the x-axis tick marks (short lines) but keep the labels.
    ax.tick_params(axis="x", length=6)

    # --- Vertical split line at speedup == 1.0 ---
    # Draw on the twin axis (ax2) so it sits above the histogram bars.
    ax2.axvline(x=1.0, color="dimgray", linestyle="dashed", linewidth=2, zorder=5)

    ax.tick_params(which="both", direction="in", labelsize=18)
    ax2.tick_params(which="both", direction="in", labelsize=18)

    # Set frame (spines) linewidth
    frame_linewidth = 2
    for spine in ["top", "right", "bottom", "left"]:
        ax.spines[spine].set_linewidth(frame_linewidth)

    # lines1, labels1 = ax.get_legend_handles_labels()
    # lines2, labels2 = ax2.get_legend_handles_labels()
    # ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    fig.savefig(outpath, dpi=200, format="png", bbox_inches="tight")
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot a node-hours-weighted speedup distribution from a folder of parquet files."
    )
    parser.add_argument(
        "--input_dir",
        required=True,
        help="Path to a folder containing parquet files, each with 'speedup' and 'node_hours' columns.",
    )
    parser.add_argument(
        "--outpath",
        default="speedup_distribution.png",
        help="Path to save the output plot (default: speedup_distribution.png).",
    )
    parser.add_argument(
        "--gpu-name", default="H100", help="Name of the target GPU used for labels (default: H100)."
    )
    parser.add_argument(
        "--bins", type=int, default=40, help="Number of histogram bins (default: 40)."
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
        help="Filter out rows with node_hours greater than this value (default: 100).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    result_df = load_parquet_folder(args.input_dir, max_node_hours=args.max_node_hours)

    plot_speedup_distribution(
        result_df,
        gpu_name=args.gpu_name,
        outpath=args.outpath,
        bins=np.arange(0, 3.1, 0.1),
        density=args.density,
    )
    print(f"Plot saved to {args.outpath}")


if __name__ == "__main__":
    main()
