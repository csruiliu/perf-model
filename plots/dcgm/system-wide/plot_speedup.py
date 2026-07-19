import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_speedup_distribution(result_df, gpu_name, outpath, bins=40):
    """Node-hours-weighted speedup PDF + cumulative % for one target GPU."""
    fig, ax = plt.subplots()
    ax2 = ax.twinx()

    s = result_df["speedup"].to_numpy()
    weights = result_df["node_hours"].to_numpy()

    ax.hist(s, bins=bins, weights=weights, alpha=0.6, label=gpu_name)

    # Cumulative percentage curve.
    order = np.argsort(s)
    cum = 100.0 * np.cumsum(weights[order]) / np.sum(weights)
    ax2.plot(s[order], cum, linestyle="--", marker=".", label=f"{gpu_name} Cumulative %")

    ax.set_xlabel(f"Speedup relative to A100 ({gpu_name})")
    ax.set_ylabel("PDF (weighted by node-hours)")
    ax2.set_ylabel("Cumulative Percentage (%)")
    ax.legend(loc="center left")
    fig.savefig(outpath, dpi=200, format="png", bbox_inches="tight")
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot a node-hours-weighted speedup distribution from a parquet file."
    )
    parser.add_argument(
        "--input_path",
        required=True,
        help="Path to the input parquet file containing 'speedup' and 'node_hours' columns.",
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
    return parser.parse_args()


def main():
    args = parse_args()

    result_df = pd.read_parquet(args.input_path)

    plot_speedup_distribution(
        result_df, gpu_name=args.gpu_name, outpath=args.outpath, bins=args.bins
    )
    print(f"Plot saved to {args.outpath}")


if __name__ == "__main__":
    main()
