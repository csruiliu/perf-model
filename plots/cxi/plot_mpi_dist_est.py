import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["hatch.linewidth"] = 3  # default is 1.0

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Plot MPI message distribution from a CSV file.")
parser.add_argument("csv_file", help="Path to the input CSV file")
parser.add_argument(
    "-o",
    "--output",
    default="mpi_msg_dist_est.png",
    help="Path to the output image file (default: mpi_msg_dist_est.png)",
)
args = parser.parse_args()

# Load data from CSV
df = pd.read_csv(args.csv_file)
categories = df["category"].tolist()
new_mpi_model = df["predictions"].tolist()
measurement_ipm = df["measurements"].tolist()

# Colors matching the original chart
color_new_mpi = "royalblue"  # dark teal/navy
color_ipm = "darkorange"  # orange

hatch_new_mpi = "///"  # dark teal/navy
hatch_ipm = "\\\\\\"  # orange

x = np.arange(len(categories))
width = 0.3  # bar width

fig, ax = plt.subplots(figsize=(12, 5))

# On a log scale, zero values can't be drawn. We clip them to the axis
# bottom (1) so they don't render as a visible bar, then annotate.
bottom = 1


def clip(values):
    return [v if v > 0 else bottom for v in values]


edge_lw = 3  # bar edge linewidth
width = 0.35  # width of each individual bar
gap = 0.03  # gap between the two bars in a group (data units)

bars_new = ax.bar(
    x - (width + gap) / 2,
    clip(new_mpi_model),
    width,
    label="Predictions",
    color="white",
    hatch=hatch_new_mpi,
    linewidth=edge_lw,
    edgecolor=color_new_mpi,
)
bars_ipm = ax.bar(
    x + (width + gap) / 2,
    clip(measurement_ipm),
    width,
    label="Measurements",
    color="white",
    hatch=hatch_ipm,
    linewidth=edge_lw,
    edgecolor=color_ipm,
)

# Log scale on y-axis
ax.set_yscale("log")
ax.set_ylim(1, 10000)


# Annotate all bars. Small values (0 and 1) get labeled near the baseline
# since their bars are clipped; larger values get labeled on top of the bar.
def annotate_all(bars, values, color):
    for bar, val in zip(bars, values):
        if val <= 1:  # includes 0 and 1, bars are clipped to baseline
            y = 1.15
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                y,
                str(val),
                ha="center",
                va="bottom",
                color=color,
                fontsize=20,
                fontweight="bold",
            )
        else:
            y = val * 1.05  # slightly above the top of the bar
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                y,
                str(val),
                ha="center",
                va="bottom",
                color=color,
                fontsize=14,
                fontweight="bold",
            )


annotate_all(bars_new, new_mpi_model, color_new_mpi)
annotate_all(bars_ipm, measurement_ipm, color_ipm)

# Labels and title
ax.set_xticks(x)
ax.set_xticklabels(categories, rotation=0, ha="center")

# Legend at top
ax.legend(loc="upper center", ncol=2, frameon=True, fontsize=18)

# Set frame (spines) linewidth
frame_linewidth = 3
for spine in ["top", "right", "bottom", "left"]:
    ax.spines[spine].set_linewidth(frame_linewidth)

# X ticks: no tick marks, just labels
ax.tick_params(axis="x", length=0, labelsize=17)
# Y ticks: obvious, pointing inside, both major and minor
ax.tick_params(axis="y", which="major", direction="in", length=8, width=2, labelsize=18)
ax.tick_params(axis="y", which="minor", direction="in", length=4, width=1)

plt.tight_layout()
plt.savefig(args.output, dpi=300, bbox_inches="tight")
