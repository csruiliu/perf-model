import matplotlib.pyplot as plt
import numpy as np

x_labels = [64, 256, 2048]
x = np.arange(len(x_labels))
bar_width = 0.2

# -------------------------------------------------------------------
# Data: replace with your actual osu_bibw measurements/predictions
# -------------------------------------------------------------------
# Node 0
node0_send_measurement = [100000, 100000, 100000]
node0_recv_measurement = [100000, 100000, 100000]
node0_send_prediction = [0, 0, 96174]
node0_recv_prediction = [0, 0, 98565.0]

# Node 1
node1_send_measurement = [100000, 100000, 100000]
node1_recv_measurement = [100000, 100000, 100000]
node1_send_prediction = [0, 0, 97475]
node1_recv_prediction = [0, 0, 97501]

measurement_size = [64, 256, 2048]

node0_send_prediction_size = [None, None, 2048]
node0_recv_prediction_size = [None, None, 2048]
node1_send_prediction_size = [None, None, 2048]
node1_recv_prediction_size = [None, None, 2048]

# -------------------------------------------------------------------
# Split configs: only indices 0 (64B) and 1 (256B) remain after trimming
# -------------------------------------------------------------------
split_configs = {
    0: {  # Node 0
        "send": {
            0: {
                "counts": [8013] * 11,
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
                ],
            },
            1: {
                "counts": [20012] * 5,
                "sizes": [224, 256, 320, 384, 448],
                "colors": ["#FFD580", "#FFA500", "#FF8C00", "#FF6600", "#CC4400"],
            },
        },
        "recv": {
            0: {
                "counts": [8059] * 11,
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
                ],
            },
            1: {
                "counts": [19948] * 5,
                "sizes": [224, 256, 320, 384, 448],
                "colors": ["#FFD580", "#FFA500", "#FF8C00", "#FF6600", "#CC4400"],
            },
        },
    },
    1: {  # Node 1
        "send": {
            0: {
                "counts": [8088] * 11,
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
                ],
            },
            1: {
                "counts": [19954] * 5,
                "sizes": [224, 256, 320, 384, 448],
                "colors": ["#FFD580", "#FFA500", "#FF8C00", "#FF6600", "#CC4400"],
            },
        },
        "recv": {
            0: {
                "counts": [7991] * 11,
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
                ],
            },
            1: {
                "counts": [20012] * 5,
                "sizes": [224, 256, 320, 384, 448],
                "colors": ["#FFD580", "#FFA500", "#FF8C00", "#FF6600", "#CC4400"],
            },
        },
    },
}


# -------------------------------------------------------------------
# Helper: draw one subplot
# -------------------------------------------------------------------
def draw_subplot(ax, node_id, measurement_count, prediction_count, pred_size_labels, split_cfg):

    offsets = {"send_meas": -1.5, "send_pred": -0.5, "recv_meas": 0.5, "recv_pred": 1.5}
    colors = {
        "send_meas": "steelblue",
        "send_pred": "coral",
        "recv_meas": "mediumseagreen",
        "recv_pred": "gold",
    }
    hatches = {"send_meas": "XX", "recv_meas": "XX"}  # ground-truth only
    bar_labels = {"send_meas": "Send Ground-truth", "recv_meas": "Recv Ground-truth"}

    pred_x_send = x + offsets["send_pred"] * bar_width
    pred_x_recv = x + offsets["recv_pred"] * bar_width

    # -- Ground-truth bars with hatching --
    for _, meas_data, offset_key in [
        ("send", measurement_count["send"], "send_meas"),
        ("recv", measurement_count["recv"], "recv_meas"),
    ]:
        bars = ax.bar(
            x + offsets[offset_key] * bar_width,
            meas_data,
            bar_width,
            label=bar_labels[offset_key],
            color=colors[offset_key],
            hatch=hatches[offset_key],
            edgecolor="white",
            alpha=1,
        )
        for bar, size in zip(bars, measurement_size, strict=True):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 500,
                f"{size}B",
                ha="center",
                va="bottom",
                fontsize=11,
                color=colors[offset_key],
                fontweight="bold",
            )

    # -- Prediction bars with optional splits --
    for direction_key, pred_data, pred_x, offset_key, pred_sizes in [
        ("send", prediction_count["send"], pred_x_send, "send_pred", pred_size_labels["send"]),
        ("recv", prediction_count["recv"], pred_x_recv, "recv_pred", pred_size_labels["recv"]),
    ]:
        cfg = split_cfg[direction_key]
        all_split_bars = {}

        for i in range(len(x_labels)):
            if i in cfg:
                # stacked bar
                bottom = 0
                split_bars = []
                for j, (count, color) in enumerate(
                    zip(cfg[i]["counts"], cfg[i]["colors"], strict=True)
                ):
                    bar = ax.bar(pred_x[i], count, bar_width, bottom=bottom, color=color, alpha=1)
                    split_bars.append((bar, bottom, count, cfg[i]["sizes"][j]))
                    bottom += count
                all_split_bars[i] = split_bars
            else:
                bar = ax.bar(pred_x[i], pred_data[i], bar_width, color=colors[offset_key], alpha=1)
                if pred_sizes[i] is not None:
                    bar_obj = bar[0]
                    ax.text(
                        bar_obj.get_x() + bar_obj.get_width() / 2,
                        bar_obj.get_height() + 500,
                        f"{pred_sizes[i]}B",
                        ha="center",
                        va="bottom",
                        fontsize=11,
                        color=colors[offset_key],
                        fontweight="bold",
                    )

        # annotate stacked segments
        for _, split_bars in all_split_bars.items():
            for j, (bar_container, bot, count, size) in enumerate(split_bars):
                bar = bar_container[0]
                mid_y = bot + count / 2
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    mid_y,
                    f"{size}B\n{count}",
                    ha="center",
                    va="center",
                    fontsize=10,
                    color="black" if j < 3 else "white",
                    fontweight="bold",
                )

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=16)
    ax.tick_params(axis="y", labelsize=16)
    ax.set_xlabel("MPI Message Size (B)", fontsize=16, fontweight="bold")
    ax.set_ylabel("Message Count", fontsize=16, fontweight="bold")
    # ax.set_title(f'Node {node_id}: Send & Recv — Ground-truth vs Prediction', fontsize=16, fontweight='bold')
    ax.set_ylim(0, 120000)
    ax.legend(fontsize=13, ncol=2)


# -------------------------------------------------------------------
# Build figure with two subplots
# -------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(24, 8), sharey=True)

for node_id, ax in enumerate(axes):
    draw_subplot(
        ax=ax,
        node_id=node_id,
        measurement_count={
            "send": node0_send_measurement if node_id == 0 else node1_send_measurement,
            "recv": node0_recv_measurement if node_id == 0 else node1_recv_measurement,
        },
        prediction_count={
            "send": node0_send_prediction if node_id == 0 else node1_send_prediction,
            "recv": node0_recv_prediction if node_id == 0 else node1_recv_prediction,
        },
        pred_size_labels={
            "send": node0_send_prediction_size if node_id == 0 else node1_send_prediction_size,
            "recv": node0_recv_prediction_size if node_id == 0 else node1_recv_prediction_size,
        },
        split_cfg=split_configs[node_id],
    )

fig.suptitle("osu_bibw: MPI Message Count — Node 0 vs Node 1", fontsize=20, fontweight="bold")
plt.tight_layout()
plt.savefig("osu_bibw_grouped_bar.png", dpi=150, bbox_inches="tight")
plt.show()
