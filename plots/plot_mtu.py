import matplotlib.pyplot as plt
import numpy as np

# --- MPI Message Sizes (x-axis) ---
message_sizes = [1, 2, 4, 8, 16, 32, 64]
message_sizes_label = ["512B", "1KB", "2KB", "4KB", "8KB", "16KB", "32KB"]
x = np.arange(len(message_sizes))
bar_width = 0.5

# --- Placeholder data ---
# Top panel: Rank-0 'Sender' — stacked bars: hni_tx_ok_64, hni_tx_ok_65to127, hni_tx_ok_128to256
sender_tx_ok_512_to_1023    = [1, 0, 0, 0, 0, 0, 0]
sender_tx_ok_1024_to_2047   = [0, 1, 0, 0, 0, 0, 0]
sender_tx_ok_2048_to_4095   = [0, 0, 1, 2, 4, 8, 16]
sender_tx_ok_4096_to_8191   = [0, 0, 0, 0, 0, 0, 0]
sender_tx_ok_8192_to_max    = [0, 0, 0, 0, 0, 0, 0]

# Bottom panel: Rank-1 'Receiver' — stacked bar: hni_rx_ok_64, etc.
receiver_rx_ok_512_to_1023   = [1, 0, 0, 0, 0, 0, 0]
receiver_rx_ok_1024_to_2047  = [0, 1, 0, 0, 0, 0, 0]
receiver_rx_ok_2048_to_4095  = [0, 0, 1, 2, 4, 8, 16]
receiver_rx_ok_4096_to_8191  = [0, 0, 0, 0, 0, 0, 0]
receiver_rx_ok_8192_to_max   = [0, 0, 0, 0, 0, 0, 0]

# --- Colors for stacked segments ---
colors = ['#2c2c2c', '#7a7a7a', '#c0c0c0', '#f0f0f0', 'snow']   # dark, mid, light grey
hatch_patterns = ['//', '\\\\', 'xx', '++', 'ooo']  # diagonal, backslash, crosshatch, plus, circles
# --- Figure with 2 subplots side by side ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('OSU_BW', fontsize=15, fontweight='bold')

# ── Helper to draw a stacked bar subplot ─────────────────────────────────────
def draw_stacked_bar(ax, title, layers, layer_labels, colors, x, bar_width, message_sizes):
    bottoms = np.zeros(len(x))
    for layer, label, color, hatch in zip(layers, layer_labels, colors, hatch_patterns):
        ax.bar(x, layer, bar_width, bottom=bottoms,
               label=label, color=color, edgecolor='black', linewidth=0.7, hatch=hatch)
        bottoms += np.array(layer)

    ax.set_title(title, fontsize=12)
    ax.set_ylabel('Counts Per Message', fontsize=11)
    ax.set_xlabel('MPI Message Size', fontsize=11)
    ax.set_xticks(x)
    ax.set_ylim(0, 20)
    ax.set_yticks([0,2,4,6,8,10,12,14,16,18,20])
    ax.set_xticklabels([str(s) for s in message_sizes_label])
    ax.legend(loc='upper left', fontsize=9, frameon=True)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.set_axisbelow(True)

# ── Top subplot: Rank-0 'Sender' ─────────────────────────────────────────────
draw_stacked_bar(
    ax=ax1,
    title="Rank-0  'Sender'",
    layers=[sender_tx_ok_512_to_1023, sender_tx_ok_1024_to_2047, sender_tx_ok_2048_to_4095, sender_tx_ok_4096_to_8191, sender_tx_ok_8192_to_max],
    layer_labels=['hni_tx_ok_512_to_1023', 'hni_tx_ok_1024_to_2047', 'hni_tx_ok_2048_to_4095', 'hni_tx_ok_4096_to_8191', 'hni_tx_ok_8192_to_max'],
    colors=colors,
    x=x,
    bar_width=bar_width,
    message_sizes=message_sizes
)

# ── Bottom subplot: Rank-0 'Receiver' ────────────────────────────────────────
draw_stacked_bar(
    ax=ax2,
    title="Rank-1  'Receiver'",
    layers=[receiver_rx_ok_512_to_1023, receiver_rx_ok_1024_to_2047, receiver_rx_ok_2048_to_4095, receiver_rx_ok_4096_to_8191, receiver_rx_ok_8192_to_max],
    layer_labels=['hni_rx_ok_512_to_1023', 'hni_rx_ok_1024_to_2047', 'hni_rx_ok_2048_to_4095', 'hni_rx_ok_4096_to_8191', 'hni_rx_ok_8192_to_max'],
    colors=colors,
    x=x,
    bar_width=bar_width,
    message_sizes=message_sizes
)

plt.tight_layout()
plt.savefig('plot_mtu.png', dpi=150)
