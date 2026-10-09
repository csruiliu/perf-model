#!/usr/bin/env python3
"""Plot measured vs. predicted runtime for several applications as side-by-side
panels that together fit one IEEE column (3.5 in). Fonts are set at their final
printed size, so include the output at width=\\columnwidth without rescaling.

Data are hard-coded in PANELS below (runtime in seconds)."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Legend labels match the estimator names used in the paper text.
LABELS = ["Measured", r"$\Theta_{\min}$", r"$\Theta_{\mathrm{mean}}$",
          r"$\Theta_{\max}$", r"$\Theta_{\mathrm{mock}}$"]
COLORS = ["silver", "royalblue", "salmon", "seagreen", "darkorange"]
HATCHES = ["", "//////", "\\\\\\\\\\\\", "xxxx", "oo"]

COLUMN_WIDTH_IN = 3.5  # IEEE two-column \columnwidth

# Each panel: (title, categories, rows). Each row is
# [Measured, Theta_min, Theta_mean, Theta_max, Theta_mock] for one category.
PANELS = [
    ("(a) MILC (FP32)", ["A40", "RTX8000"], [
        [1951, 1588, 1564, 1564, 1564],
        [2040, 2636, 1617, 1617, 1617],
    ]),
    ("(b) LAMMPS (FP32)", ["A40", "RTX8000"], [
        [970,1401,855,626,626],
        [1429, 2365, 1071, 709, 708],
    ]),
]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("-o", "--output", type=Path, default=Path("runtime_pair.png"))
    p.add_argument("--height", type=float, default=1.7, help="Figure height in inches.")
    p.add_argument("--no-sharey", action="store_true", help="Give each panel its own y axis.")
    return p.parse_args()


def draw_panel(ax, categories, data, title):
    x = np.arange(len(categories))
    n = data.shape[1]
    width = 0.17
    bar_w = width * 0.88
    meas = data[:, 0]

    for i in range(n):
        offset = (i - n / 2) * width + width / 2
        label = LABELS[i] if i < len(LABELS) else f"Series {i}"
        if i == 0:
            ax.bar(x + offset, data[:, i], bar_w, label=label, color=COLORS[0],
                   edgecolor="dimgrey", linewidth=0.6)
        else:
            bars = ax.bar(x + offset, data[:, i], bar_w, label=label, color="white",
                          hatch=HATCHES[i % len(HATCHES)],
                          edgecolor=COLORS[i % len(COLORS)], linewidth=0.6)
            err = (data[:, i] - meas) / meas * 100
            ax.bar_label(bars, labels=[f"{p:+.1f}%" for p in err],
                         rotation=90, padding=1.5, fontsize=5.5)

    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_title(title, fontsize=7.5, pad=2)
    ax.tick_params(axis="x", length=0, labelsize=7)
    ax.tick_params(axis="y", direction="in", labelsize=6.5, length=2)
    for s in ax.spines.values():
        s.set_linewidth(0.8)


def main():
    args = parse_args()

    plt.rcParams.update({"hatch.linewidth": 0.6, "font.size": 7,
                         "pdf.fonttype": 42, "ps.fonttype": 42})

    panels = [(title, cats, np.array(rows, dtype=float)) for title, cats, rows in PANELS]
    fig, axes = plt.subplots(1, len(panels), figsize=(COLUMN_WIDTH_IN, args.height),
                             sharey=not args.no_sharey, squeeze=False)
    axes = axes[0]

    for ax, (title, cats, data) in zip(axes, panels):
        draw_panel(ax, cats, data, title)

    ymax = max(d.max() for _, _, d in panels)
    for ax in axes:
        ax.set_ylim(0, ymax * 1.4)
    axes[0].set_ylabel("Runtime (s)", fontsize=7)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), fontsize=6.5,
               frameon=False, handlelength=1.6, columnspacing=0.9,
               handletextpad=0.4, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(pad=0.2, w_pad=0.4, rect=(0, 0, 1, 0.88))

    fig.savefig(args.output, bbox_inches="tight", pad_inches=0.01, dpi=300, format="png")
    plt.close(fig)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()