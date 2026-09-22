#!/usr/bin/env python3
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLUMNS = ["Measurement", "SMOCC Lower", "SMOCC Mid", "SMOCC Upper", "SMOCC Mock"]

COLORS = [
    "silver",  # measurement (distinct)
    "lightsalmon",  # smocc_lower
    "salmon",  # smocc_mid
    "tomato",  # smocc_upper
    "red",  # mock_smocc
]
HATCHES = [
    "",  # measurement (no hatch)
    "//",  # smocc_lower
    "\\\\",  # smocc_mid
    "xx",  # smocc_upper
    "O",  # mock_smocc
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot measured vs. SMOCC runtimes from a CSV file."
    )
    parser.add_argument(
        "csv_path", type=Path, help="Path to the input CSV file (must contain a 'category' column)."
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output image path. Defaults to '<csv basename>_reduce.png'.",
    )
    parser.add_argument(
        "-e",
        "--exclude",
        action="append",
        default=None,
        metavar="CATEGORY",
        help="Category to exclude; may be given multiple times. Defaults to excluding 'A100-40G'.",
    )
    parser.add_argument(
        "--ylabel", default="Overall Runtime (second)", help="Label for the y axis."
    )
    parser.add_argument("--dpi", type=int, default=300, help="Output DPI.")
    return parser.parse_args()


def load_data(csv_path, exclude):
    df = pd.read_csv(csv_path)
    if "category" not in df.columns:
        raise ValueError(f"{csv_path} has no 'category' column.")

    categories_csv = list(dict.fromkeys(df["category"]))
    categories = [cat for cat in categories_csv if cat not in exclude]
    if not categories:
        raise ValueError("All categories were excluded; nothing to plot.")

    value_cols = [c for c in df.columns if c != "category"]
    data = np.array(
        [
            df[df["category"] == cat][value_cols].mean(axis=0).round().astype(int).values
            for cat in categories
        ]
    )
    return categories, data


def plot(categories, data, args):
    plt.rcParams["hatch.linewidth"] = 3  # default is 1.0

    n_cols = min(len(COLUMNS), data.shape[1])
    x = np.arange(len(categories))
    width = 0.17
    bar_width = width * 0.9  # leave a small gap between bars

    fig, ax = plt.subplots(figsize=(12, 4.5))

    # Reference measurement values (column 0) for each category
    measurement = data[:, 0]

    for i in range(n_cols):
        col = COLUMNS[i]
        offset = (i - n_cols / 2) * width + width / 2
        if i == 0:
            # First bar: keep the solid fill color, grey edge.
            ax.bar(
                x + offset,
                data[:, i],
                bar_width,
                label=col,
                color=COLORS[i],
                hatch=HATCHES[i],
                edgecolor="darkgrey",
                linewidth=3,
            )
        else:
            # Other bars: white fill, colored edge + hatch
            bars = ax.bar(
                x + offset,
                data[:, i],
                bar_width,
                label=col,
                color="white",  # or "none" for transparent
                hatch=HATCHES[i],
                edgecolor=COLORS[i],
                linewidth=3,
            )
            # Signed error percentage relative to the measurement bar
            err_pct = (data[:, i] - measurement) / measurement * 100
            labels = [f"{p:+.1f}%" for p in err_pct]
            ax.bar_label(bars, labels=labels, padding=2, fontsize=17)

    ax.set_ylabel(args.ylabel, fontsize=22)
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend(ncol=3, loc="upper left", columnspacing=0.6, fontsize=19)

    ax.set_ylim(0, np.max(data) * 1.5)

    # Remove ticks on x axis (keep the labels)
    ax.tick_params(axis="x", length=0, labelsize=22)

    # Make y axis ticks point inward
    ax.tick_params(axis="y", direction="in", labelsize=22)

    # Set frame (spines) linewidth
    frame_linewidth = 3
    for spine in ["top", "right", "bottom", "left"]:
        ax.spines[spine].set_linewidth(frame_linewidth)

    fig.tight_layout()
    return fig


def main():
    args = parse_args()
    exclude = set(args.exclude) if args.exclude else {"A100-40G"}

    output = args.output
    if output is None:
        output = args.csv_path.with_name(f"{args.csv_path.stem}_reduce.png")

    categories, data = load_data(args.csv_path, exclude)
    fig = plot(categories, data, args)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
