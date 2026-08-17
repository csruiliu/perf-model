"""Grouped bar charts: measured vs. SMOCC-predicted runtimes (BGW, fp64).

Usage:
    python plot_bgw.py results/run-2024-05-01
    python plot_bgw.py results/run-2024-05-01 --out-dir figures --dpi 300
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Style
# --------------------------------------------------------------------------- #
plt.rcParams.update({"hatch.linewidth": 3, "axes.linewidth": 2, "font.size": 20})

SERIES = ("Measurement", "SMOCC Lower", "SMOCC Mid", "SMOCC Upper", "SMOCC Mock")
HATCHES = ("", "//", "\\\\", "xx", "O")

PALETTES = {
    "eps": ("silver", "peachpuff", "burlywood", "orange", "peru"),
    "sig": ("silver", "yellowgreen", "mediumseagreen", "seagreen", "darkgreen"),
}

BAR_WIDTH = 0.14  # slot width per series
BAR_FILL = 0.9  # fraction of the slot actually drawn (leaves a gap)
EDGE_WIDTH = 3


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load_group_means(csv_path):
    """Return (categories, means) where means has shape (n_categories, n_value_cols).

    Rows sharing a ``category`` value are averaged; category order follows first
    appearance in the file.
    """
    df = pd.read_csv(csv_path)
    if "category" not in df.columns:
        raise ValueError(f"{csv_path}: missing required 'category' column")

    means = df.drop(columns="category").groupby(df["category"], sort=False).mean()
    return list(means.index), np.rint(means.to_numpy()).astype(int)


def stack_sources(data_dir, sources):
    """Concatenate several CSVs from ``data_dir`` into one matrix.

    ``sources`` maps a display label (e.g. "H100-SXM") to a file name. The label
    is used verbatim when the file holds a single category, otherwise it is
    combined with the in-file category name, so the x tick labels can never
    drift out of sync with the data.
    """
    labels, blocks = [], []
    for label, name in sources.items():
        path = data_dir / name
        if not path.is_file():
            raise FileNotFoundError(path)

        categories, values = load_group_means(path)
        if len(categories) == 1:
            labels.append(label)
        else:
            labels.extend(f"{label}\n{cat}" for cat in categories)
        blocks.append(values)

    data = np.vstack(blocks)
    if data.shape[1] != len(SERIES):
        raise ValueError(f"expected {len(SERIES)} value columns, found {data.shape[1]}")
    return labels, data


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #
def draw_grouped_bars(ax, data, colors, label_fontsize=20):
    n_groups, n_series = data.shape
    x = np.arange(n_groups)

    for i, (name, color, hatch) in enumerate(zip(SERIES, colors, HATCHES)):
        offset = (i - n_series / 2 + 0.5) * BAR_WIDTH
        # First series: solid fill, neutral edge. Rest: white fill, coloured edge.
        style = (
            {"color": color, "edgecolor": "darkgrey"}
            if i == 0
            else {"color": "white", "edgecolor": color}
        )
        bars = ax.bar(
            x + offset,
            data[:, i],
            BAR_WIDTH * BAR_FILL,
            label=name,
            hatch=hatch,
            linewidth=EDGE_WIDTH,
            **style,
        )
        ax.bar_label(bars, fmt="%d", padding=2, fontsize=label_fontsize)

    return x


def make_figure(
    data_dir,
    out_dir,
    sources,
    palette,
    out_name,
    ylabel="Overall Runtime (second)",
    headroom=1.35,
    figsize=(12, 4),
    dpi=150,
    label_fontsize=20,
):
    labels, data = stack_sources(data_dir, sources)

    fig, ax = plt.subplots(figsize=figsize)
    x = draw_grouped_bars(ax, data, PALETTES[palette], label_fontsize)

    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, data.max() * headroom)
    ax.legend(ncol=3, loc="upper left", columnspacing=0.6, fontsize=15, frameon=False)

    ax.tick_params(axis="x", length=0)  # keep labels, drop the ticks
    ax.tick_params(axis="y", direction="in")

    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{out_name}.png"
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)  # avoid leaking figures
    return out_path


# --------------------------------------------------------------------------- #
# Figures to build
# --------------------------------------------------------------------------- #
FIGURES = (
    {
        "out_name": "bgw_eps_fp64_bars",
        "palette": "eps",
        "headroom": 1.5,
        "sources": {
            "H100-SXM": "bgw-eps-fp64-h100-ref.csv",
            "A100-40GB": "bgw-eps-fp64-a100-ref.csv",
        },
    },
    {
        "out_name": "bgw_sig_fp64_bars",
        "palette": "sig",
        "headroom": 1.5,
        "sources": {
            "H100-SXM": "bgw-sig-fp64-h100-ref.csv",
            "A100-40GB": "bgw-sig-fp64-a100-ref.csv",
        },
    },
)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Plot BGW fp64 runtime comparisons from a results folder.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("data_dir", type=Path, help="folder containing the bgw-*-fp64-*-ref.csv files")
    p.add_argument(
        "-o",
        "--out-dir",
        type=Path,
        default=None,
        help="where to write the PNGs (default: same as data_dir)",
    )
    p.add_argument("--dpi", type=int, default=300, help="output resolution")
    p.add_argument(
        "--label-fontsize",
        type=int,
        default=20,
        help="font size of the numeric labels above the bars",
    )
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    data_dir = args.data_dir
    if not data_dir.is_dir():
        sys.exit(f"error: {data_dir} is not a directory")

    out_dir = args.out_dir if args.out_dir is not None else data_dir

    for spec in FIGURES:
        try:
            path = make_figure(
                data_dir, out_dir, dpi=args.dpi, label_fontsize=args.label_fontsize, **spec
            )
        except FileNotFoundError as exc:
            sys.exit(f"error: missing input file {exc}")
        print("wrote", path)


if __name__ == "__main__":
    main()
