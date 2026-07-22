import matplotlib.pyplot as plt
import numpy as np

# --- MPI Message Sizes (x-axis) ---
message_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
message_sizes_label = ["1B", "2B", "4B", "8B", "16B", "32B", "64B", "128B", "192B", "193B"]
x = np.arange(len(message_sizes))
bar_width = 0.5

# --- Placeholder data ---
# Top panel: Rank-0 'Sender' — stacked bars: hni_tx_ok_64, hni_tx_ok_65to127, hni_tx_ok_128to256
sender_tx_ok_64 = [2, 2, 2, 2, 0, 0, 0, 0, 0, 2]
sender_tx_ok_65to127 = [0, 0, 0, 0, 2, 2, 2, 0, 0, 0]
sender_tx_ok_128to255 = [0, 0, 0, 0, 0, 0, 0, 2, 2, 1]

# Bottom panel: Rank-1 'Receiver' — stacked bar: hni_rx_ok_64, etc.
receiver_rx_ok_64 = [2, 2, 2, 2, 0, 0, 0, 0, 0, 2]
receiver_rx_ok_65to127 = [0, 0, 0, 0, 2, 2, 2, 0, 0, 0]
receiver_rx_ok_128to255 = [0, 0, 0, 0, 0, 0, 0, 2, 2, 1]

# --- Colors for stacked segments ---
colors = ["#2c2c2c", "#7a7a7a", "#c0c0c0"]  # dark, mid, light grey

# --- Figure with 2 subplots side by side ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("OSU_BW", fontsize=15, fontweight="bold")


# ── Helper to draw a stacked bar subplot ─────────────────────────────────────
def draw_stacked_bar(ax, title, layers, layer_labels, colors, x, bar_width, message_sizes):
    bottoms = np.zeros(len(x))
    for layer, label, color in zip(layers, layer_labels, colors, strict=True):
        ax.bar(
            x,
            layer,
            bar_width,
            bottom=bottoms,
            label=label,
            color=color,
            edgecolor="black",
            linewidth=0.7,
        )
        bottoms += np.array(layer)

    ax.set_title(title, fontsize=12)
    ax.set_ylabel("Counts Per Message", fontsize=11)
    ax.set_xlabel("MPI Message Size", fontsize=11)
    ax.set_xticks(x)
    ax.set_ylim(1, 4)
    ax.set_yticks([2, 3, 4])
    ax.set_xticklabels([str(s) for s in message_sizes_label])
    ax.legend(loc="upper left", fontsize=9, frameon=True)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)


# ── Top subplot: Rank-0 'Sender' ─────────────────────────────────────────────
draw_stacked_bar(
    ax=ax1,
    title="Rank-0  'Sender'",
    layers=[sender_tx_ok_64, sender_tx_ok_65to127, sender_tx_ok_128to255],
    layer_labels=["hni_tx_ok_64", "hni_tx_ok_65_to_127", "hni_tx_ok_128_to_255"],
    colors=colors,
    x=x,
    bar_width=bar_width,
    message_sizes=message_sizes,
)

# ── Bottom subplot: Rank-0 'Receiver' ────────────────────────────────────────
draw_stacked_bar(
    ax=ax2,
    title="Rank-1  'Receiver'",
    layers=[receiver_rx_ok_64, receiver_rx_ok_65to127, receiver_rx_ok_128to255],
    layer_labels=["hni_rx_ok_64", "hni_rx_ok_65_to_127", "hni_rx_ok_128_to_255"],
    colors=colors,
    x=x,
    bar_width=bar_width,
    message_sizes=message_sizes,
)

plt.tight_layout()
plt.savefig("plot_header_inline.png", dpi=150)
