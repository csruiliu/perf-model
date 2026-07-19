import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

plt.rcParams["hatch.linewidth"] = 2  # default is 1.0

# (label, bottom_hatch, top_hatch)
# bottom_hatch: hatch pattern for bottom segment ('' = white/empty)
# top_hatch: None means no top segment (height=1), otherwise height=2
bar_groups = [
    [("1", "", None)],
    [("11", "", None), ("12", "///", None)],
    [("74", "///", None), ("75", "xx", None)],
    [("192", "xx", None), ("193", "", "xx")],
    [("202", "", "xx"), ("203", "", "\\\\\\")],
    [("458", "", "\\\\\\"), ("459", "", "-\\\\")],
    [("970", "", "-\\\\"), ("971", "", "oo")],
    [("1994", "", "oo"), ("1995", "", "||")],
    [],
]

fig, ax = plt.subplots(figsize=(15, 5))

pos = 0
x_tick_pos = []
x_tick_labels = []
ellipsis_xs = []
width = 0.6

for g_idx, group in enumerate(bar_groups):
    for label, bottom_hatch, top_hatch in group:
        # Bottom segment — height = 1, controllable hatch
        ax.bar(
            pos,
            1,
            width=width,
            color="white",
            edgecolor="black",
            linewidth=2,
            hatch=bottom_hatch,
            zorder=3,
        )
        # Top segment — height = 1 (stacked), only if top_hatch is specified
        if top_hatch is not None:
            ax.bar(
                pos,
                1,
                width=width,
                bottom=1,
                color="white",
                edgecolor="black",
                linewidth=2,
                hatch=top_hatch,
                zorder=3,
            )
        x_tick_pos.append(pos)
        x_tick_labels.append(label)
        pos += 1

    if g_idx < len(bar_groups) - 1:
        ellipsis_xs.append(pos)
        pos += 1  # gap for ellipsis

# Draw '...' between bar groups at baseline
for ex in ellipsis_xs:
    ax.text(ex, 0.2, "...", ha="center", va="bottom", fontsize=30, color="black")

# --- Legend ---
# Define all hatch patterns and their labels used in the chart
hatch_legend = [
    ("", "64B"),
    ("///", "65-127B"),
    ("xx", "128-255B"),
    ("\\\\\\", "256-511B"),
    ("-\\\\", "512-1023B"),
    ("oo", "1024-2047B"),
    ("||", "2048-4095B"),
]
legend_handles = [
    mpatches.Patch(facecolor="white", edgecolor="black", linewidth=1.2, hatch=hatch, label=label)
    for hatch, label in hatch_legend
]
ax.legend(
    handles=legend_handles,
    loc="upper left",
    fontsize=11,
    frameon=True,
    title="Data Packet Buckets",
    title_fontsize=11,
)

# --- Axis labels ---
ax.set_xlabel("MPI Message Sizes (Bytes)", fontsize=22)
ax.set_ylabel("Data Packets Per Message", fontsize=22)

# --- Axis ticks ---
ax.set_xticks(x_tick_pos)
ax.set_xticklabels(x_tick_labels, fontsize=17)
ax.set_yticks([0, 1, 2, 3])
ax.set_yticklabels([0, 1, 2, 3], fontsize=17)
ax.set_ylim(0, 3.1)
ax.set_xlim(-0.8, pos - 0.5)

# --- Styling ---
ax.tick_params(bottom=False)

# Set frame (spines) linewidth
frame_linewidth = 3
for spine in ["top", "right", "bottom", "left"]:
    ax.spines[spine].set_linewidth(frame_linewidth)

plt.tight_layout()
plt.savefig("mpi_msg_dist.png", dpi=300, format="png", bbox_inches="tight")
