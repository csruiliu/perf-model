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
fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(
    message_sizes,
    hni_pkts_t0,
    marker="s",
    markerfacecolor="white",
    markeredgecolor="black",
    color="black",
    linewidth=1.2,
    markersize=8,
    label="hni_pkts_sent_by_tc_0",
)

ax.plot(
    message_sizes,
    hni_pkts_t1,
    marker="s",
    markerfacecolor="black",
    markeredgecolor="black",
    color="black",
    linewidth=1.2,
    markersize=8,
    linestyle="--",
    label="hni_pkts_sent_by_tc_1",
)

# --- Axis labels and title ---
ax.set_xlabel("MPI Message Size", fontsize=12)
ax.set_ylabel("Counts Per Message", fontsize=12)
ax.set_title("OSU_BW  —  Rank 0", fontsize=13)

# --- Log scale on x-axis (typical for MPI benchmarks) ---
ax.set_xscale("log", base=2)
ax.set_xticks(message_sizes)
ax.set_xticklabels([str(s) for s in message_sizes_label], rotation=45, ha="right")
# plt.xticks(message_sizes, message_sizes_label)

# --- Legend ---
ax.legend(loc="upper left", frameon=True, fontsize=10)

ax.grid(True, which="both", linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig("plot_tc_flip.png", dpi=150)
# plt.show()
