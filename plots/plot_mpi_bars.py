import matplotlib.pyplot as plt
import numpy as np

x_labels = [64, 128, 256, 512, 1024, 2048]
x = np.arange(len(x_labels))
bar_width = 0.45

measurement_count = [100000, 100000, 100000, 100000, 100000, 100000]
prediction_count = [62000, 48000, 43000, 42000, 100876, 101549]

measurement_size = [64, 128, 256, 512, 1024, 2048]
prediction_size = [None, None, None, None, 1024, 2048]  # None = handled separately

# --- Split definitions ---
split_configs = {
    0: {  # 64B ground-truth column
        "counts": [8308] * 11,
        "sizes": [12, 14, 16, 20, 24, 28, 32, 40, 48, 56, 64],
        "colors": [
            "#F3E5F5",
            "#E1BEE7",
            "#CE93D8",
            "#BA68C8",
            "#AB47BC",
            "#9C27B0",
            "#8E24AA",
            "#7B1FA2",
            "#6A1B9A",
            "#4A148C",
            "#2D0060",
        ],  # shades of purple light -> dark
    },
    1: {  # 128B ground-truth column
        "counts": [15000] * 6,
        "sizes": [80, 96, 112, 128, 160, 192],
        "colors": [
            "#B0E0B0",
            "#6DBF6D",
            "#3A9E3A",
            "#1F7A1F",
            "#0F5C0F",
            "#073D07",
        ],  # shades of green
    },
    2: {  # 256B ground-truth column
        "counts": [17654] * 5,
        "sizes": [224, 256, 320, 384, 448],
        "colors": ["#FFD580", "#FFA500", "#FF8C00", "#FF6600", "#CC4400"],  # shades of orange
    },
    3: {  # 512B ground-truth column
        "counts": [22200] * 4,
        "sizes": [512, 640, 768, 896],
        "colors": ["#FF7F7F", "#FF5733", "#C70039", "#900C3F"],  # shades of coral/red
    },
    4: {  # 1024B ground-truth column
        "counts": [24968] * 4,
        "sizes": [1024, 1280, 1536, 1792],
        "colors": ["#FF7F7F", "#FF5733", "#C70039", "#900C3F"],  # shades of coral/red
    },
}

fig, ax = plt.subplots(figsize=(18, 8))

# --- Ground-truth bars (unchanged) ---
bars1 = ax.bar(
    x - bar_width / 2,
    measurement_count,
    bar_width,
    label="Ground-truth",
    color="steelblue",
    alpha=1,
)

# --- Prediction bars: all indices except split ones ---
pred_x = x + bar_width / 2
bars2 = []
for i in range(len(x_labels)):
    if i in split_configs:
        bars2.append(None)  # placeholder
        continue
    bar = ax.bar(
        pred_x[i],
        prediction_count[i],
        bar_width,
        color="coral",
        alpha=0.85,
        label="Prediction" if i == 4 else "",
    )
    bars2.append(bar)

# --- Stacked prediction bars at each split index ---
all_split_bars = {}
for idx, cfg in split_configs.items():
    bottom = 0
    split_bars = []
    for j, (count, color) in enumerate(zip(cfg["counts"], cfg["colors"], strict=True)):
        bar = ax.bar(
            pred_x[idx],
            count,
            bar_width,
            bottom=bottom,
            color=color,
            alpha=1,
            label=f"Pred {cfg['sizes'][j]}B",
        )
        split_bars.append((bar, bottom, count))
        bottom += count
    all_split_bars[idx] = split_bars

# --- Annotate ground-truth bars ---
for _, (bar, size) in enumerate(zip(bars1, measurement_size, strict=True)):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 500,
        f"{size}B",
        ha="center",
        va="bottom",
        fontsize=14,
        color="steelblue",
        fontweight="bold",
    )

# --- Annotate normal prediction bars ---
for i, bar_container in enumerate(bars2):
    if bar_container is None:
        continue
    bar = bar_container[0]
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 500,
        f"{prediction_size[i]}B",
        ha="center",
        va="bottom",
        fontsize=14,
        color="coral",
        fontweight="bold",
    )

# --- Annotate stacked split bars ---
for idx, split_bars in all_split_bars.items():
    cfg = split_configs[idx]
    for j, (bar_container, bot, count) in enumerate(split_bars):
        bar = bar_container[0]
        mid_y = bot + count / 2
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            mid_y,
            f"{cfg['sizes'][j]} B\n{count}",
            ha="center",
            va="center",
            fontsize=12,
            color="black" if j < 3 else "white",
            fontweight="bold",
        )

ax.set_xticks(x)
ax.set_xticklabels(x_labels, fontsize=20)
ax.tick_params(axis="y", labelsize=20)
ax.set_xlabel("MPI Size (Bytes)", fontsize=20, fontweight="bold")
ax.set_ylabel("Message Count", fontsize=20, fontweight="bold")
ax.set_title("Ground-truth (left bars) vs Prediction (right bars)", fontsize=20, fontweight="bold")


ax.set_ylim(0, 120000)

plt.tight_layout()
plt.savefig("grouped_bar_annotated.png", dpi=150)
plt.show()
