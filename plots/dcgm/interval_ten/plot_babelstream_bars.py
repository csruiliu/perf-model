import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["hatch.linewidth"] = 3  # default is 1.0
columns = ["Measurement", "SMOCC Lower", "SMOCC Mid", "SMOCC Upper", "SMOCC Mock"]

categories = ["H100-SXM", "A40", "RTX8000"]

babelstream_a100_ref_df = pd.read_csv("babelstream-a100-ref.csv")
categories_csv = list(dict.fromkeys(babelstream_a100_ref_df["category"]))
value_cols = [c for c in babelstream_a100_ref_df.columns if c != "category"]
milc_fp64_a100_ref_data = np.array(
    [
        babelstream_a100_ref_df[babelstream_a100_ref_df["category"] == cat][value_cols]
        .mean(axis=0)
        .astype(int)
        .values
        for cat in categories_csv
    ]
)

data = milc_fp64_a100_ref_data

# Reference (measured) values: first column, one per category
measured = data[:, 0].astype(float)

# Error percentage of each column relative to the measured value
# shape: (n_categories, n_columns)
error_pct = (data - measured[:, None]) / measured[:, None] * 100.0

# 'measured' gets its own distinct color and no hatch.
# All smocc bars share a similar color, distinguished by hatch.
colors = [
    "silver",  # measured (distinct)
    "aquamarine",  # smocc_lower
    "turquoise",  # smocc_mid
    "mediumaquamarine",  # smocc_upper
    "lightseagreen",
]  # mock_smocc
hatches = [
    "",  # measured (no hatch)
    "//",  # smocc_lower
    "\\\\",  # smocc_mid
    "xx",  # smocc_upper
    "O",
]  # mock_smocc

x = np.arange(len(categories))
n_cols = len(columns)
width = 0.18
bar_width = width * 0.9

fig, ax = plt.subplots(figsize=(12, 4))

for i, col in enumerate(columns):
    offset = (i - n_cols / 2) * width + width / 2
    if i == 0:
        # First bar: keep the solid fill color, black edge
        bars = ax.bar(
            x + offset,
            data[:, i],
            bar_width,
            label=col,
            color=colors[i],
            hatch=hatches[i],
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
            hatch=hatches[i],
            edgecolor=colors[i],
            linewidth=3,
        )

    # Build error-percentage labels (no annotation on Measurement)
    if i == 0:
        labels = ["" for _ in error_pct[:, i]]
    else:
        labels = [f"{p:+.1f}%" for p in error_pct[:, i]]
    ax.bar_label(bars, labels=labels, padding=2, fontsize=14)

# ax.set_xlabel('Category')
ax.set_ylabel("Memory Bandwidth (GB/s)", fontsize=19)
# ax.set_title('Comparison by Category')
ax.set_xticks(x)
ax.set_xticklabels(categories)
ax.legend(ncol=3, loc="upper left", columnspacing=0.6, fontsize=15)

ax.set_ylim(0, np.max(data) * 1.5)

# Remove ticks on x axis (keep the labels)
ax.tick_params(axis="x", length=0, labelsize=20)

# Make y axis ticks point inward
ax.tick_params(axis="y", direction="in", labelsize=20)

# Set frame (spines) linewidth
frame_linewidth = 2
for spine in ["top", "right", "bottom", "left"]:
    ax.spines[spine].set_linewidth(frame_linewidth)

fig.tight_layout()
plt.savefig("babelstream_ref_a100.png", dpi=150)
