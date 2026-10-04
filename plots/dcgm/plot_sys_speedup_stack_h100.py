import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

plt.rcParams['axes.formatter.useoffset'] = False
plt.rcParams['axes.formatter.limits'] = (-100, 100)

FRAME_LW = 2
CUM_COLOR = "mediumorchid"

# ---------------------------------------------------------------------------
# Naive spec-ratio baselines (no DCGM counters)
# ---------------------------------------------------------------------------
# Peak rates from Tables II and V (dense; TFLOPS, GB/s).
SPECS = {
    "A100-40GB": dict(tf64=19.5, tf32=156,  tf16=312,  fp64=9.7, fp32=19.5, fp16=78,
                      dram=1555, pcie=32),
    "H100":      dict(tf64=67,   tf32=495,  tf16=990,  fp64=34,  fp32=67,   fp16=133.6,
                      dram=3350, pcie=64),
    # NOTE: B300 vector fp16 (2200) equals TF16 in Table V; please verify.
    "B300":      dict(tf64=1.2,  tf32=1100, tf16=2200, fp64=1.2, fp32=75,   fp16=2200,
                      dram=7700, pcie=128),
}

BASELINE_REF = "A100-40GB"
BASELINE_TGT = "H100"

# Distinct colors AND line styles, so the lines stay readable in grayscale.
BASELINE_STYLES = {
    "Mem. BW":   dict(color="navy",      linestyle=(0, (6, 3))),
    "FP64":      dict(color="firebrick", linestyle=(0, (1.5, 1.5))),
    "Geo. mean": dict(color="darkgreen", linestyle=(0, (6, 2, 1.5, 2))),
}
BASELINE_LW = 2.5


def spec_ratio_baselines(ref, tgt):
    """Return [(name, speedup)] for the three naive spec-ratio baselines."""
    r = {k: SPECS[tgt][k] / SPECS[ref][k] for k in SPECS[ref]}
    geo = float(np.exp(np.mean(np.log(list(r.values())))))
    return [("Mem. BW", r["dram"]), ("FP64", r["fp64"]), ("Geo. mean", geo)]


def _draw_baselines(ax_hi, ax_lo, baselines, xlim, show_label):
    """Vertical spec-ratio lines through both y-segments of one panel.

    Lines beyond the x-range still get a legend entry (the clipped artist
    exists) and are marked with an arrow at the panel edge.
    """
    offaxis = 0
    for name, value in baselines:
        style = BASELINE_STYLES[name]
        label = f"Spec Scale: {name} ({value:.2f}\u00d7)" if show_label else "_nolegend_"
        ax_lo.axvline(value, linewidth=BASELINE_LW, zorder=3, label=label, **style)
        ax_hi.axvline(value, linewidth=BASELINE_LW, zorder=3, **style)
        if not (xlim[0] <= value <= xlim[1]):
            right = value > xlim[1]
            ax_lo.text(
                0.995 if right else 0.005, 0.92 - 0.14 * offaxis,
                (f"{name} {value:.2f}\u00d7 \u2192" if right
                 else f"\u2190 {name} {value:.2f}\u00d7"),
                transform=ax_lo.transAxes, ha="right" if right else "left",
                va="top", fontsize=15, color=style["color"], zorder=21,
                bbox=dict(facecolor="white", edgecolor="none", pad=1.5),
            )
            offaxis += 1


def load_parquet_folder(input_dir, max_node_hours=100):
    """Read and concatenate all parquet files (job_id-indexed) in a folder."""
    input_dir = Path(input_dir)
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {input_dir}")

    files = sorted(input_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No .parquet files found in {input_dir}")

    frames = []
    for f in files:
        df = pd.read_parquet(f)
        for col in ("speedup", "node_hours"):
            if col not in df.columns:
                raise ValueError(f"{f} is missing required column: {col!r}")
        frames.append(df)

    combined = pd.concat(frames)  # preserve the job_id index

    dupes = combined.index.duplicated()
    if dupes.any():
        print(f"Warning: {dupes.sum()} duplicate job_id(s) across files; keeping first occurrence.")
        combined = combined[~combined.index.duplicated(keep="first")]

    # Filter out rows with node_hours over the threshold.
    before = len(combined)
    combined = combined[combined["node_hours"].between(0, max_node_hours, inclusive="right")]
    dropped = before - len(combined)
    if dropped:
        print(f"Filtered out {dropped} row(s) with node_hours > {max_node_hours}.")

    print(f"Loaded {len(files)} file(s), {len(combined)} job_id(s) after filtering.")
    return combined


def total_node_hours(result_df):
    """Total node-hours over all valid rows (the normalization denominator)."""
    df = result_df[["speedup", "node_hours"]].dropna()
    total = df["node_hours"].to_numpy().sum()
    if total <= 0:
        raise ValueError("Sum of node_hours is non-positive; cannot normalize.")
    return total


YMODE_LABELS = {
    "absolute": "Node-hours",
    "fraction": "Fraction of Node-Hours",
    "percent": "Node-Hours (% of total)",
}


def _full_scale(y_mode, denom):
    """Left-axis value that corresponds to 100% of node-hours."""
    return {"absolute": denom, "fraction": 1.0, "percent": 100.0}[y_mode]


# ---------------------------------------------------------------------------
# Layout helpers
# ---------------------------------------------------------------------------
def _make_broken_panel(fig, subplot_spec, lo_span, hi_span, gap, sharex=None):
    """Split one panel into an upper and a lower y-segment.

    Height ratios equal the y-spans, so one percentage point has the same
    physical height in both segments (and in both panels).
    """
    inner = subplot_spec.subgridspec(2, 1, height_ratios=[hi_span, lo_span], hspace=gap)
    ax_hi = fig.add_subplot(inner[0], sharex=sharex)
    ax_lo = fig.add_subplot(inner[1], sharex=sharex or ax_hi)
    return ax_hi, ax_lo


def _add_cumulative_axis(fig, ax_hi, ax_lo):
    """One continuous axis spanning both segments, for the cumulative curve."""
    p_hi, p_lo = ax_hi.get_position(), ax_lo.get_position()
    ax_cum = fig.add_axes(
        [p_lo.x0, p_lo.y0, p_lo.width, p_hi.y1 - p_lo.y0], sharex=ax_lo
    )
    ax_cum.patch.set_visible(False)
    ax_cum.xaxis.set_visible(False)
    ax_cum.yaxis.tick_right()
    ax_cum.yaxis.set_label_position("right")
    for side in ("left", "top", "bottom"):
        ax_cum.spines[side].set_visible(False)
    ax_cum.spines["right"].set_linewidth(FRAME_LW)
    return ax_cum


def _style_segments(ax_hi, ax_lo):
    """Hide the inner spines and draw break marks on the left (broken) axis.

    The right spine belongs to the cumulative axis, which is continuous, so
    it carries no break marks.
    """
    for ax in (ax_hi, ax_lo):
        for side in ("left", "top", "bottom"):
            ax.spines[side].set_linewidth(FRAME_LW)
        ax.spines["right"].set_visible(False)
        ax.tick_params(which="both", direction="in", labelsize=19)
    ax_hi.spines["bottom"].set_visible(False)
    ax_lo.spines["top"].set_visible(False)
    ax_hi.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    ax_lo.tick_params(axis="x", length=4.5, width=2)

    mark = dict(
        marker=[(-1, -0.5), (1, 0.5)], markersize=16, linestyle="none",
        color="k", mec="k", mew=FRAME_LW, clip_on=False,
    )
    ax_hi.plot([0], [0], transform=ax_hi.transAxes, **mark)
    ax_lo.plot([0], [1], transform=ax_lo.transAxes, **mark)


def _set_break_limits(ax_hi, ax_lo, y_mode, full, lo_pct, hi_pct, top_pct, step_pct):
    """Apply the broken y-range. Limits are given in percent of total."""
    u = full / 100.0
    ax_lo.set_ylim(0, lo_pct * u)
    ax_hi.set_ylim(hi_pct * u, top_pct * u)

    # Ticks at multiples of step_pct; skip the ones that sit on the break
    # edges or the top edge, where labels would collide.
    ticks = np.arange(0, top_pct + 1e-9, step_pct)
    ax_lo.set_yticks([t * u for t in ticks if t < lo_pct - 1e-9])
    ax_hi.set_yticks([t * u for t in ticks if hi_pct + 1e-9 < t < top_pct - 1e-9])

    if y_mode == "percent":
        fmt = mticker.PercentFormatter(xmax=100, decimals=0)
    elif y_mode == "fraction":
        fmt = mticker.PercentFormatter(xmax=1, decimals=0)
    else:
        fmt = None
    if fmt is not None:
        for ax in (ax_hi, ax_lo):
            ax.yaxis.set_major_formatter(fmt)


def _align_cumulative_axis(ax_cum, ax_hi, ax_lo, full):
    """Put 0 and 100 on the right axis level with 0% and 100% on the left.

    If the left axis stops below 100%, there is nothing to align with, so the
    right axis falls back to a plain 0-110 range.
    """
    p_hi, p_lo = ax_hi.get_position(), ax_lo.get_position()
    y_lo, y_hi = ax_hi.get_ylim()
    if y_lo <= full <= y_hi:
        y100 = p_hi.y0 + (full - y_lo) / (y_hi - y_lo) * p_hi.height
        frac = (y100 - p_lo.y0) / (p_hi.y1 - p_lo.y0)
        ax_cum.set_ylim(0, 100.0 / frac)
    else:
        ax_cum.set_ylim(0, 110)
    ax_cum.yaxis.set_major_locator(mticker.MultipleLocator(20))


def _center_left_label(fig, ax_hi, ax_lo, text, fontsize):
    """Center the left y-label on the whole panel, just left of the tick labels."""
    ax_lo.set_ylabel(text, fontsize=fontsize)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    labels = [
        t for ax in (ax_hi, ax_lo) for t in ax.get_yticklabels()
        if t.get_visible() and t.get_text()
    ]
    x_left = min(t.get_window_extent(renderer).x0 for t in labels)
    pad_px = 6 * fig.dpi / 72.0
    bb = ax_lo.bbox
    x = (x_left - pad_px - bb.x0) / bb.width
    y_mid = (ax_lo.get_position().y0 + ax_hi.get_position().y1) / 2.0
    y = (y_mid - ax_lo.get_position().y0) / ax_lo.get_position().height
    ax_lo.yaxis.set_label_coords(x, y, transform=ax_lo.transAxes)


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------
def _draw_panel(
    ax_hi, ax_lo, ax_cum, result_df, gpu_name, color, edgecolor, bins,
    y_mode, denom, legend_label=None, baselines=None, xlim=None,
    baseline_legend=True,
):
    """Draw one speedup histogram (in both y-segments) + cumulative % curve.

    `denom` is the node-hour total used for normalization; pass the same value
    for both panels when they describe the same job set.
    `baselines` is an optional [(name, speedup)] list of spec-ratio lines.
    Returns the bar heights so the caller can check them against the break.
    """
    df = result_df[["speedup", "node_hours"]].dropna()
    if df.empty:
        raise ValueError("No valid (non-NaN) rows to plot.")
    s = df["speedup"].to_numpy()
    node_hours = df["node_hours"].to_numpy()
    total = node_hours.sum()
    if total <= 0:
        raise ValueError("Sum of node_hours is non-positive; cannot compute cumulative %.")

    weights = node_hours * (_full_scale(y_mode, denom) / denom)

    hist_kw = dict(
        bins=bins, weights=weights, color=color, edgecolor=edgecolor,
        histtype="stepfilled", alpha=0.5, linewidth=2,
    )
    hist_label = legend_label if legend_label is not None else gpu_name
    counts, bin_edges, _ = ax_lo.hist(s, label=hist_label, **hist_kw)
    ax_hi.hist(s, **hist_kw)

    # Baselines are drawn after the histogram so the legend lists it first.
    if baselines:
        _draw_baselines(ax_hi, ax_lo, baselines, xlim, show_label=baseline_legend)

    in_range = (s >= bin_edges[0]) & (s <= bin_edges[-1])
    outside = node_hours[~in_range].sum()
    if outside > 0:
        print(
            f"{gpu_name}: {100.0 * outside / total:.2f}% of node-hours fall "
            f"outside the bin range [{bin_edges[0]}, {bin_edges[-1]}] and are not shown."
        )

    print(gpu_name)
    print(f"  total node-hours = {total:,.1f} (denominator = {denom:,.1f})")
    print(f"  bar heights sum to {counts.sum():.6g}")
    print(counts)
    print(bin_edges)

    order = np.argsort(s)
    cum = 100.0 * np.cumsum(node_hours[order]) / total
    ax_cum.plot(
        s[order], cum, linestyle=(0, (5, 1)), linewidth=2, color=CUM_COLOR,
        label=f"{gpu_name} cumulative %",
    )
    ax_cum.set_ylabel("Cumulative Percentage (%)", fontsize=20)
    ax_cum.tick_params(which="both", direction="in", labelsize=23)

    handles, labels = ax_lo.get_legend_handles_labels()
    leg = ax_cum.legend(
        handles, labels, loc="upper left", fontsize=18, frameon=True,
        framealpha=1.0, edgecolor="black", facecolor="white",
    )
    leg.set_zorder(20)
    return counts


def _check_bars_against_break(name, counts, full, lo_pct, hi_pct, top_pct):
    """Warn when a bar top would be hidden in the gap or cut off at the top."""
    pct = 100.0 * counts / full
    in_gap = (pct > lo_pct) & (pct < hi_pct)
    if in_gap.any():
        print(
            f"Warning ({name}): {in_gap.sum()} bar(s) end inside the axis break "
            f"({lo_pct:g}%-{hi_pct:g}%): {np.round(pct[in_gap], 2)}%. Their tops are not visible."
        )
    if (pct > top_pct).any():
        print(f"Warning ({name}): bar(s) exceed the {top_pct:g}% axis top and are clipped.")


def plot_speedup_distribution_stacked(
    top_df,
    bottom_df,
    outpath,
    bins,
    y_mode="percent",
    shared_denominator=True,
    ybreak=(15.0, 90.0),
    ytop=105.0,
    ytick_step=5.0,
    xlim=(-0.1, 3.1),
    xtick_step=0.2,
    baselines=None,
    baseline_offaxis="extend",
):
    """Two stacked panels, shared x-axis, identical broken y-axes."""
    lo_pct, hi_pct = ybreak
    if not 0 < lo_pct < hi_pct < ytop:
        raise ValueError(f"Need 0 < break_low < break_high < ytop; got {ybreak}, ytop={ytop}.")

    # Widen the x-range so every baseline line is visible ("extend"), or keep
    # it and mark off-axis lines with an arrow at the edge ("arrow").
    if baselines and baseline_offaxis == "extend":
        vmax = max(v for _, v in baselines)
        if vmax > xlim[1]:
            new_hi = np.ceil(vmax / xtick_step - 1e-9) * xtick_step + 0.1
            print(f"Extending x-axis upper limit from {xlim[1]} to {new_hi:.2f} "
                  f"to show the {vmax:.2f}x baseline.")
            xlim = (xlim[0], float(new_hi))

    top_total = total_node_hours(top_df)
    bot_total = total_node_hours(bottom_df)
    if shared_denominator:
        denom_top = denom_bot = top_total
        rel_diff = abs(top_total - bot_total) / top_total
        if rel_diff > 1e-6:
            print(
                f"Note: panel totals differ by {100 * rel_diff:.3f}% "
                f"({top_total:,.1f} vs {bot_total:,.1f}); using the top panel's "
                f"total for both. Pass --per-panel-denominator to normalize each "
                f"panel by its own total."
            )
    else:
        denom_top, denom_bot = top_total, bot_total

    # The spec-ratio baseline has no non-GPU term, so the same lines appear in
    # both panels; their legend entries are shown in the top panel only.
    panel_cfg = [
        dict(df=top_df, denom=denom_top, gpu_name="H100",
             color="gold", edgecolor="darkgoldenrod", legend_label=None,
             baseline_legend=True),
        dict(df=bottom_df, denom=denom_bot, gpu_name="H100-NG2",
             color="sandybrown", edgecolor="darkorange",
             legend_label="Hypothetical H100\n(Non-GPU Portion Scale Up 2x)",
             baseline_legend=False),
    ]

    # --- Layout: every segment exists before any overlay axis is placed ---
    fig = plt.figure(figsize=(14, 10))
    outer = fig.add_gridspec(2, 1, hspace=0.0)
    segments, sharex = [], None
    for i in range(2):
        ax_hi, ax_lo = _make_broken_panel(
            fig, outer[i], lo_span=lo_pct, hi_span=ytop - hi_pct, gap=0.06, sharex=sharex
        )
        sharex = sharex or ax_hi
        segments.append((ax_hi, ax_lo))
    cum_axes = [_add_cumulative_axis(fig, hi, lo) for hi, lo in segments]

    xticks = np.arange(np.ceil(xlim[0] / xtick_step - 1e-9) * xtick_step,
                       xlim[1] + 1e-9, xtick_step)
    xticks = np.round(xticks, 10)

    for i, ((ax_hi, ax_lo), ax_cum, cfg) in enumerate(zip(segments, cum_axes, panel_cfg)):
        counts = _draw_panel(
            ax_hi, ax_lo, ax_cum, cfg["df"], cfg["gpu_name"], cfg["color"],
            cfg["edgecolor"], bins, y_mode, cfg["denom"], cfg["legend_label"],
            baselines=baselines, xlim=xlim, baseline_legend=cfg["baseline_legend"],
        )
        full = _full_scale(y_mode, cfg["denom"])
        _check_bars_against_break(cfg["gpu_name"], counts, full, lo_pct, hi_pct, ytop)

        _style_segments(ax_hi, ax_lo)
        _set_break_limits(ax_hi, ax_lo, y_mode, full, lo_pct, hi_pct, ytop, ytick_step)
        ax_lo.set_xlim(*xlim)
        ax_lo.set_xticks(xticks)
        _align_cumulative_axis(ax_cum, ax_hi, ax_lo, full)

        is_bottom = i == len(segments) - 1
        ax_lo.tick_params(axis="x", labelbottom=is_bottom)
        if is_bottom:
            ax_lo.set_xlabel("Speedup Relative to A100", fontsize=26)

    for ax_hi, ax_lo in segments:
        _center_left_label(fig, ax_hi, ax_lo, YMODE_LABELS[y_mode], fontsize=21)

    fig.savefig(outpath, dpi=300, format="png", bbox_inches="tight")
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot two stacked node-hours-weighted speedup distributions "
        "(shared x-axis, broken y-axis) from two folders of parquet files."
    )
    parser.add_argument("--top-input-dir", required=True,
                        help="Folder of parquet files for the TOP panel.")
    parser.add_argument("--bottom-input-dir", required=True,
                        help="Folder of parquet files for the BOTTOM panel.")
    parser.add_argument("--outpath", default="speedup_distribution_stacked.png",
                        help="Path to save the output plot.")
    parser.add_argument(
        "--y_mode", choices=("absolute", "fraction", "percent"), default="percent",
        help="Left y-axis unit: raw node-hours, fraction of total, or percent of "
        "total. Default: percent.",
    )
    parser.add_argument(
        "--per-panel-denominator", dest="shared_denominator", action="store_false",
        help="Normalize each panel by its own node-hour total instead of a shared one.",
    )
    parser.add_argument("--max-node-hours", type=float, default=720,
                        help="Filter out rows with node_hours greater than this value (default: 720).")
    parser.add_argument(
        "--ybreak", type=float, nargs=2, default=(15.0, 90.0), metavar=("LOW", "HIGH"),
        help="Omitted y-range, in percent of total node-hours (default: 15 90).",
    )
    parser.add_argument("--ytop", type=float, default=105.0,
                        help="Top of the upper y-segment, in percent of total (default: 105).")
    parser.add_argument("--ytick-step", type=float, default=5.0,
                        help="Left y-axis tick spacing, in percent of total (default: 5).")
    parser.add_argument(
        "--xlim", type=float, nargs=2, default=(-0.1, 3.1), metavar=("XMIN", "XMAX"),
        help="Speedup range shown on the x-axis (default: -0.1 3.1).",
    )
    parser.add_argument("--no-baselines", dest="baselines", action="store_false",
                        help="Do not draw the spec-ratio baseline lines.")
    parser.add_argument(
        "--baseline-offaxis", choices=("extend", "arrow"), default="extend",
        help="For baseline lines beyond --xlim: widen the x-axis to show them "
        "('extend', default) or keep the x-axis and mark them with an arrow "
        "at the panel edge ('arrow').",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    top_df = load_parquet_folder(args.top_input_dir, max_node_hours=args.max_node_hours)
    bottom_df = load_parquet_folder(args.bottom_input_dir, max_node_hours=args.max_node_hours)

    baselines = None
    if args.baselines:
        baselines = spec_ratio_baselines(BASELINE_REF, BASELINE_TGT)
        for name, v in baselines:
            print(f"Baseline {BASELINE_REF} -> {BASELINE_TGT}, {name}: {v:.3f}x")

    plot_speedup_distribution_stacked(
        top_df=top_df,
        bottom_df=bottom_df,
        outpath=args.outpath,
        bins=np.arange(0, 4.1, 0.05),
        y_mode=args.y_mode,
        shared_denominator=args.shared_denominator,
        ybreak=tuple(args.ybreak),
        ytop=args.ytop,
        ytick_step=args.ytick_step,
        xlim=tuple(args.xlim),
        baselines=baselines,
        baseline_offaxis=args.baseline_offaxis,
    )
    print(f"Plot saved to {args.outpath}")


if __name__ == "__main__":
    main()