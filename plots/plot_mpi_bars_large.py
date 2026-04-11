import matplotlib.pyplot as plt
import numpy as np

x_labels = [4096, 8192, 16384, 32768, 65536, 1048576]
x = np.arange(len(x_labels))
bar_width = 0.45

measurement_count = [200000, 400000, 800000, 1600000, 3200000, 50000000]
prediction_count = [150151, 250382, 446606, 786339, 1567983, 24962376]

measurement_size = [2048, 2048, 2048, 2048, 2048, 2048]
prediction_size = [2048, 2048, 2048, 2048, 2048, 2048]

split_configs = {}

fig, ax = plt.subplots(figsize=(18, 8))

# --- Ground-truth bars ---
bars1 = ax.bar(
    x - bar_width / 2,
    measurement_count,
    bar_width,
    label="Ground-truth",
    color="steelblue",
    alpha=0.85,
)

# --- Prediction bars ---
pred_x = x + bar_width / 2
bars2 = []
for i in range(len(x_labels)):
    if i in split_configs:
        bars2.append(None)
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

# --- Stacked prediction bars ---
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
            alpha=0.85,
            label=f"Pred {cfg['sizes'][j]}B",
        )
        split_bars.append((bar, bottom, count))
        bottom += count
    all_split_bars[idx] = split_bars

# --- Switch to log scale ---
ax.set_yscale("log")
ax.set_ylim(1e5, 2e8)  # adjust lower/upper bounds to your data range

# Multiplicative offset for annotations on log scale
LABEL_OFFSET = 1.15  # places text 15% above the bar top

# --- Annotate ground-truth bars ---
for _, (bar, size) in enumerate(zip(bars1, measurement_size, strict=True)):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() * LABEL_OFFSET,  # ← multiplicative, not additive
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
        bar.get_height() * LABEL_OFFSET,  # ← multiplicative, not additive
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
            f"{cfg['sizes'][j]}B\n{count}",
            ha="center",
            va="center",
            fontsize=9,
            color="black" if j < 3 else "white",
            fontweight="bold",
        )

ax.set_xticks(x)
ax.set_xticklabels(x_labels, fontsize=20)
ax.tick_params(axis="y", labelsize=20)
ax.set_xlabel("MPI Size (Bytes)", fontsize=20, fontweight="bold")
ax.set_ylabel("Message Count (log scale)", fontsize=20, fontweight="bold")
ax.set_title("Ground-truth (left bars) vs Prediction (right bars)", fontsize=20, fontweight="bold")

plt.tight_layout()
plt.savefig("grouped_bar_annotated.png", dpi=150)
plt.show()
