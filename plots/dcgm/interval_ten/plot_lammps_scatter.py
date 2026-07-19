import csv

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.legend_handler import HandlerPatch
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Patch
from matplotlib.ticker import FixedLocator, FuncFormatter

mpl.rcParams["hatch.linewidth"] = 1.4  # default is ~1.0; increase for thicker hatch lines

# ---------------------------------------------------------------------------
# Configuration: which CSV corresponds to which reference GPU, and how the
# "category" values in each file map onto canonical target-GPU names.
# ---------------------------------------------------------------------------
csv_files = {
    "A100-40GB": "lammps-fp32-a100-ref.csv",  # reference = A100
    "H100-SXM": "lammps-fp32-h100-ref.csv",  # reference = H100
}

# Normalize the raw "category" strings in the CSVs to canonical names.
category_alias = {
    "H100": "H100-SXM",
    "H100-SXM": "H100-SXM",
    "A100-40G": "A100-40GB",
    "A100-40GB": "A100-40GB",
    "A40": "A40",
    "RTX8000": "RTX8000",
}


def load_ref_csv(path):
    """Read one reference CSV, aggregating (mean) rows that share a target."""
    buckets = {}  # target -> list of dict(meas, mid, lo, hi, mock)
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            target = category_alias.get(row["category"].strip(), row["category"].strip())
            buckets.setdefault(target, []).append(
                dict(
                    meas=float(row["measured"]),
                    smocc_mid=float(row["smocc_mid"]),
                    smocc_lo=float(row["smocc_lower"]),
                    smocc_hi=float(row["smocc_upper"]),
                    smocc_mock=float(row["mock_smocc"]),
                )
            )

    data = {}
    for target, rows in buckets.items():
        data[target] = {
            k: float(np.mean([r[k] for r in rows]))
            for k in ("meas", "smocc_mid", "smocc_lo", "smocc_hi", "smocc_mock")
        }
    return data


# Load both references from disk.
references = {ref: load_ref_csv(path) for ref, path in csv_files.items()}

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
ref_color = {"H100-SXM": "mediumseagreen", "A100-40GB": "royalblue"}
target_marker = {"A100-40GB": "o", "H100-SXM": "o", "A40": "^", "RTX8000": "D"}
# Both mid points are circles; distinguish targets by hatch.
target_hatch = {
    "A100-40GB": "//////",
    "H100-SXM": "\\\\\\\\\\\\",
    "A40": "xxx",
    "RTX8000": "\\\\\\",
}

# ---------------------------------------------------------------------------
# Keep ONLY reciprocal (swapped ref<->target) pairs.
# i.e. keep (refA, targetB) only if (refB, targetA) also exists.
# ---------------------------------------------------------------------------
all_points = {}
for ref_name, data in references.items():
    for target, v in data.items():
        all_points[(ref_name, target)] = v

reciprocal_points = {}
for (ref_name, target), v in all_points.items():
    partner = (target, ref_name)
    if partner in all_points:
        reciprocal_points[(ref_name, target)] = v

# Rebuild the per-reference dict containing only reciprocal entries, so all
# the downstream plotting (error bars, mock, legends) works unchanged.
references = {}
for (ref_name, target), v in reciprocal_points.items():
    references.setdefault(ref_name, {})[target] = v

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(5.8, 5.3))

# --- reciprocal (swapped ref/target) connecting lines ---
points = {k: (v["meas"], v["smocc_mid"]) for k, v in reciprocal_points.items()}
drawn = set()
for (ref_name, target), (x, y) in points.items():
    partner = (target, ref_name)
    key = frozenset([(ref_name, target), partner])
    if partner in points and key not in drawn:
        px, py = points[partner]
        ax.plot([x, px], [y, py], color="mediumslateblue", lw=1.8, ls="-", zorder=2)
        drawn.add(key)

# --- Mid error bars + Mock "x" ---
for ref_name, data in references.items():
    color = ref_color[ref_name]
    for target, v in data.items():
        meas, smocc_mid = v["meas"], v["smocc_mid"]
        yerr = np.array([[smocc_mid - v["smocc_hi"]], [v["smocc_lo"] - smocc_mid]])
        # Error bars only (no marker).
        ax.errorbar(
            meas,
            smocc_mid,
            yerr=yerr,
            fmt="none",
            ecolor=color,
            elinewidth=2.5,
            capsize=5,
            capthick=2,
            zorder=3,
        )
        # Circle marker with per-target hatch on top.
        ax.scatter(
            meas,
            smocc_mid,
            marker="o",
            s=160,
            facecolors="white",
            edgecolors=color,
            linewidths=2,
            hatch=target_hatch[target],
            zorder=3.5,
        )
        ax.scatter(meas, v["smocc_mock"], marker="x", color=color, s=40, linewidths=1.8, zorder=4)

# --- diagonal + percentage guide lines (NO in-plot text) ---
lims = [300, 1000]
ax.plot(lims, lims, ls="--", color="gray", lw=1.6, zorder=1)
guide_specs = [(0.10, "-."), (0.20, ":")]
for frac, style in guide_specs:
    xs = np.array(lims)
    ax.plot(xs, xs * (1 + frac), style, color="lightgray", lw=1.3, zorder=1)
    ax.plot(xs, xs * (1 - frac), style, color="lightgray", lw=1.3, zorder=1)

# --- log axes with plain tick labels ---
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(lims)
ax.set_ylim(lims)
ax.set_aspect("equal")

ticks = [300, 400, 550, 750, 1000]
plain = FuncFormatter(lambda v, pos: f"{int(v)}")
plain_no300 = FuncFormatter(lambda v, pos: "" if int(v) == 300 else f"{int(v)}")

ax.xaxis.set_major_locator(FixedLocator(ticks))
ax.xaxis.set_major_formatter(plain)  # x shows 300
ax.xaxis.set_minor_formatter(FuncFormatter(lambda v, pos: ""))

ax.yaxis.set_major_locator(FixedLocator(ticks))
ax.yaxis.set_major_formatter(plain_no300)  # y hides 300
ax.yaxis.set_minor_formatter(FuncFormatter(lambda v, pos: ""))

fig.canvas.draw()  # labels must be rendered before we tweak alignment
for lbl in ax.get_xticklabels():
    if lbl.get_text() == "300":
        lbl.set_horizontalalignment("right")

ax.set_xlabel("Measured runtime (s)", fontsize=22)
ax.set_ylabel("Predicted runtime (s)", fontsize=22)

ax.set_xlabel("Measured runtime (s)", fontsize=22)
ax.set_ylabel("Predicted runtime (s)", fontsize=22)

# --- legends ---
# Only show references / targets that actually appear in the plot.
present_refs = set(references.keys())
present_targets = {t for data in references.values() for t in data}

color_handles = [
    Line2D(
        [0],
        [0],
        marker="s",
        linestyle="",
        markerfacecolor=c,
        markeredgecolor=c,
        markersize=9,
        label=r,
    )
    for r, c in ref_color.items()
    if r in present_refs
]


class HandlerCircle(HandlerPatch):
    def create_artists(
        self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans
    ):
        r = min(width, height) / 2
        c = Circle(
            (width / 2 - xdescent, height / 2 - ydescent),
            r,
            facecolor="white",
            edgecolor=orig_handle.get_edgecolor(),
            hatch=orig_handle.get_hatch(),
            linewidth=orig_handle.get_linewidth(),
        )
        c.set_transform(trans)
        return [c]


shape_handles = [
    Patch(facecolor="white", edgecolor="black", linewidth=1.5, hatch=target_hatch[t], label=t)
    for t in target_hatch
    if t in present_targets
]

estimate_handles = [
    Line2D(
        [0],
        [0],
        marker="o",
        linestyle="",
        markerfacecolor="white",
        markeredgecolor="black",
        markersize=9,
        label="Mid",
    ),
    Line2D(
        [0],
        [0],
        marker="_",
        linestyle="",
        markeredgecolor="black",
        markersize=9,
        markeredgewidth=1.8,
        label="Lower / Upper",
    ),
    Line2D(
        [0],
        [0],
        marker="x",
        linestyle="",
        markeredgecolor="black",
        markersize=9,
        markeredgewidth=2,
        label="Mock",
    ),
    # Line2D([0], [0], color="0.4", lw=1.3, label="ref↔target"),
]
guide_handles = [
    Line2D([0], [0], ls="--", color="gray", lw=1.3, label="Perfect (y = x)"),
    Line2D([0], [0], ls="-.", color="lightgray", lw=1.0, label="±10%"),
    Line2D([0], [0], ls=":", color="lightgray", lw=1.0, label="±20%"),
]

leg1 = ax.legend(
    handles=color_handles,
    title="Reference GPU",
    loc="upper left",
    fontsize=10,
    title_fontsize=11,
    framealpha=0.9,
)
ax.add_artist(leg1)

leg2 = ax.legend(
    handles=shape_handles,
    title="Target GPU",
    loc="upper left",
    bbox_to_anchor=(0.0, 0.83),
    fontsize=10,
    title_fontsize=11,
    framealpha=0.9,
    handler_map={Patch: HandlerCircle()},
)
ax.add_artist(leg2)

leg3 = ax.legend(
    handles=estimate_handles,
    title="Estimate",
    loc="lower right",
    bbox_to_anchor=(1.0, 0.0),
    fontsize=10,
    title_fontsize=11,
    framealpha=0.9,
)
ax.add_artist(leg3)

leg4 = ax.legend(
    handles=guide_handles,
    title="Guide lines",
    loc="lower right",
    bbox_to_anchor=(1.0, 0.215),
    fontsize=10,
    title_fontsize=11,
    framealpha=0.9,
)

ax.tick_params(axis="x", length=0, labelsize=18)
ax.tick_params(axis="y", direction="in", labelsize=18)

frame_linewidth = 2.5
for spine in ["top", "right", "bottom", "left"]:
    ax.spines[spine].set_linewidth(frame_linewidth)

ax.grid(True, which="major", ls=":", color="0.9", zorder=0)
plt.tight_layout()
plt.savefig("lammps_fp32_scatter.png", dpi=300, bbox_inches="tight")
