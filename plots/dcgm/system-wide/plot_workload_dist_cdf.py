import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description="Plot a CDF of jobs by node count on a log x-axis."
    )
    parser.add_argument("input", help="CSV file with columns: num_nodes,num_jobs,pct_of_jobs")
    parser.add_argument(
        "-o", "--output", default="job_cdf.pdf", help="Output figure file (default: job_cdf.pdf)"
    )
    args = parser.parse_args()

    # --- Load data ---
    df = pd.read_csv(args.input)
    df = df.sort_values("num_nodes")  # ensure sorted by node count

    nodes = df["num_nodes"].to_numpy()
    jobs = df["num_jobs"].to_numpy()

    # Cumulative fraction of jobs
    cdf = np.cumsum(jobs) / jobs.sum()

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(6, 2.3))

    ax.step(nodes, cdf, where="post", linewidth=1.8, color="#1f77b4")

    ax.set_xscale("log")
    ax.set_xlabel("Number of Nodes", fontsize=13)
    ax.set_ylabel("CDF of Jobs", fontsize=13)
    ax.set_ylim(0, 1.02)
    ax.set_xlim(1, nodes.max() * 1.1)

    # Log-scale ticks at powers of two (natural for HPC),
    # kept within the data range
    powers_of_two = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    ticks = [t for t in powers_of_two if t <= nodes.max() * 1.1]
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(t) for t in ticks])

    ax.grid(True, which="major", linestyle="--", alpha=0.4)

    # Annotate a few key percentiles (only those within the data range)
    for threshold in [1, 4, 32]:
        if threshold <= nodes.max():
            frac = jobs[nodes <= threshold].sum() / jobs.sum()
            op = "" if threshold == nodes.min() else "\u2264"
            ax.annotate(
                f"{op} {threshold} nodes ({frac * 100:.1f}%)",
                xy=(threshold, frac),
                xytext=(6, -10),
                textcoords="offset points",
                fontsize=10,
                color="dimgray",
            )

    # Remove ticks on x axis (keep the labels)

    ax.tick_params(which="both", direction="in", labelsize=10)

    # Make y axis ticks point inward
    # ax.tick_params(axis="y", direction="in", labelsize=11)

    # Set frame (spines) linewidth
    frame_linewidth = 2
    for spine in ["top", "right", "bottom", "left"]:
        ax.spines[spine].set_linewidth(frame_linewidth)

    fig.tight_layout()
    fig.savefig(args.output, bbox_inches="tight", dpi=150, format="png")
    print(f"Saved figure to {args.output}")


if __name__ == "__main__":
    main()
