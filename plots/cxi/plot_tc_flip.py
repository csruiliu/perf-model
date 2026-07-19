import matplotlib.pyplot as plt

# --- Placeholder data ---
# MPI message sizes (x-axis), typically powers of 2 in bytes
message_sizes_label = ["1B", "64B", "1KB", "16KB", "32KB"]

message_sizes = [1, 2, 4, 8, 16]

# hni_pkts sent by rank 0 (empty squares) - high at small sizes, drops quickly
hni_pkts_t0 = [2, 2, 3, 10, 3]

# hni_pkts sent by rank 1 (filled squares) - similar but slightly different
hni_pkts_t1 = [0, 0, 0, 0, 15]

# --- Plot ---
fig, ax = plt.subplots(figsize=(8, 3.5))

ax.plot(
    message_sizes,
    hni_pkts_t0,
    marker="s",
    markerfacecolor="salmon",
    markeredgecolor="salmon",
    color="salmon",
    linewidth=2,
    markersize=11,
    label="hni_pkts_sent_by_tc_0",
)

ax.plot(
    message_sizes,
    hni_pkts_t1,
    marker="h",
    markerfacecolor="teal",
    markeredgecolor="teal",
    color="teal",
    linewidth=2,
    markersize=11,
    linestyle="--",
    label="hni_pkts_sent_by_tc_1",
)

# --- Axis labels and title ---
ax.set_xlabel("MPI Message Size", fontsize=15)
ax.set_ylabel("Packet Counts Per Message", fontsize=15)

# --- Log scale on x-axis (typical for MPI benchmarks) ---
ax.set_xscale("log", base=2)
ax.set_xticks(message_sizes)
ax.set_xticklabels([str(s) for s in message_sizes_label], ha="center")
# plt.xticks(message_sizes, message_sizes_label)

# ax.tick_params(axis="x", length=0, labelsize=17)
# Y ticks: obvious, pointing inside, both major and minor
ax.tick_params(axis="y", labelsize=14)
ax.tick_params(axis="x", labelsize=14)

# Explicit tick positions
yticks = [0, 3, 6, 9, 12, 15]
ax.set_yticks(yticks)

# Explicit labels for those positions
ax.set_yticklabels(["0", "3", "6", "9", "12", "15"])

# Set frame (spines) linewidth
frame_linewidth = 3
for spine in ["top", "right", "bottom", "left"]:
    ax.spines[spine].set_linewidth(frame_linewidth)

# --- Legend ---
ax.legend(loc="upper left", frameon=True, fontsize=16)

ax.grid(True, which="both", linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig("mpi_tc_flip.png", dpi=150)
# plt.show()
