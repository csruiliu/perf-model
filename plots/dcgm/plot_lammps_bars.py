#!/usr/bin/env python3
"""Plot LAMMPS runtime comparisons from the CSVs inside a given folder.

Usage:  python plot_lammps.py <folder> [-o <output-folder>]
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLUMNS = ["Measurement", "SMOCC Lower", "SMOCC Mid", "SMOCC Upper", "SMOCC Mock"]
HATCHES = ["", "//", "\\\\", "xx", "O"]

# One entry per output figure.
PLOTS = [
    {
        "csvs": ["lammps-fp64-a100-ref.csv", "lammps-fp64-h100-ref.csv"],
        "xlabels": ["H100-SXM", "A100-40GB"],
        "colors": ["silver", "lightsalmon", "salmon", "tomato", "red"],
        "out": "lammps_fp64.png",
        "headroom": 1.5,
    },
    {
        "csvs": ["lammps-fp32-a100-ref.csv"],
        "xlabels": ["H100-SXM", "A40"],
        "colors": ["silver", "lightskyblue", "skyblue", "deepskyblue", "steelblue"],
        "out": "lammps_fp32_ref_a100.png",
        "headroom": 1.5,
    },
    {
        "csvs": ["lammps-fp32-h100-ref.csv"],
        "xlabels": ["A100-40GB", "A40"],
        "colors": ["silver", "thistle", "plum", "violet", "mediumorchid"],
        "out": "lammps_fp32_ref_h100.png",
        "headroom": 1.2,
    },
]


def load(path):
    """Mean of every value column, one row per category (original CSV order)."""
    df = pd.read_csv(path)
    return df.groupby("category", sort=False).mean().astype(int).to_numpy()


def make_plot(
    data, xlabels, colors, out, headroom=1.5, width=0.14, label_fontsize=20, frame_lw=2.0
):
    x = np.arange(len(xlabels))
    bar_width = width * 0.9  # small gap between bars

    fig, ax = plt.subplots(figsize=(12, 4))
    for i, col in enumerate(COLUMNS):
        offset = (i - len(COLUMNS) / 2 + 0.5) * width
        # First series: solid fill; the rest: white fill with a coloured hatch.
        face, edge = (colors[i], "darkgrey") if i == 0 else ("white", colors[i])
        bars = ax.bar(
            x + offset,
            data[:, i],
            bar_width,
            label=col,
            color=face,
            edgecolor=edge,
            hatch=HATCHES[i],
            linewidth=3,
        )
        ax.bar_label(bars, padding=2, fontsize=label_fontsize)

    ax.set_ylabel("Overall Runtime (second)", fontsize=20)
    ax.set_xticks(x, xlabels)
    ax.set_ylim(0, data.max() * headroom)
    ax.legend(ncol=3, loc="upper left", columnspacing=0.6, fontsize=15)
    ax.tick_params(axis="x", length=0, labelsize=20)
    ax.tick_params(axis="y", direction="in", labelsize=20)
    for spine in ax.spines.values():
        spine.set_linewidth(frame_lw)

    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("folder", type=Path, help="folder containing the CSV files")
    p.add_argument(
        "-o", "--outdir", type=Path, help="where to write the PNGs (default: same as folder)"
    )
    args = p.parse_args()

    outdir = args.outdir or args.folder
    outdir.mkdir(parents=True, exist_ok=True)

    plt.rcParams["hatch.linewidth"] = 3  # default is 1.0

    for cfg in PLOTS:
        cfg = dict(cfg)
        csvs, out = cfg.pop("csvs"), cfg.pop("out")
        data = np.concatenate([load(args.folder / f) for f in csvs], axis=0)
        if len(data) != len(cfg["xlabels"]):
            raise ValueError(f"{out}: {len(data)} category rows but {len(cfg['xlabels'])} x-labels")
        make_plot(data, out=outdir / out, **cfg)
        print(f"wrote {outdir / out}")


if __name__ == "__main__":
    main()
