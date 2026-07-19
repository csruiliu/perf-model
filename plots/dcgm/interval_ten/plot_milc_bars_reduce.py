import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

columns = ["Measurement", "SMOCC Lower", "SMOCC Mid", "SMOCC Upper", "SMOCC Mock"]

plt.rcParams["hatch.linewidth"] = 3  # default is 1.0

# Category to exclude
exclude = "A100-40G"

milc_fp32_h100_ref_df = pd.read_csv("milc-fp32-h100-ref.csv")
categories_csv = list(dict.fromkeys(milc_fp32_h100_ref_df["category"]))

# Filter out the excluded category, preserving order
categories = [cat for cat in categories_csv if cat != exclude]

value_cols = [c for c in milc_fp32_h100_ref_df.columns if c != "category"]
milc_fp32_h100_ref_data = np.array(
    [
        milc_fp32_h100_ref_df[milc_fp32_h100_ref_df["category"] == cat][value_cols]
        .mean(axis=0)
        .astype(int)
        .values
        for cat in categories
    ]
)

data = milc_fp32_h100_ref_data

# 'measured' gets its own distinct color and no hatch.
# All smocc bars share a similar color, distinguished by hatch.
colors = [
    "silver",  # measured (distinct)
    "grey",  # smocc_lower
    "gray",  # smocc_mid
    "dimgrey",  # smocc_upper
    "dimgray",
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
width = 0.14
bar_width = width * 0.9  # leave a small gap between bars

fig, ax = plt.subplots(figsize=(12, 4))

# Reference measurement values (column 0) for each category
measurement = data[:, 0]

for i, col in enumerate(columns):
    offset = (i - n_cols / 2) * width + width / 2
    if i == 0:
        # First bar: keep the solid fill color, black edge. No label.
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
        # Signed error percentage relative to the measurement bar
        err_pct = (data[:, i] - measurement) / measurement * 100
        labels = [f"{p:+.1f}%" for p in err_pct]
        ax.bar_label(bars, labels=labels, padding=2, fontsize=15)

# ax.set_xlabel('Category')
ax.set_ylabel("Overall Runtime (second)", fontsize=20)
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
plt.savefig("milc_fp32_ref_h100_reduce.png", dpi=150)
