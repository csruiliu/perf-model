import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.transforms import blended_transform_factory
import numpy as np
import pandas as pd


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
    # It enters the geometric-mean baseline (28x ratio).
    "B300":      dict(tf64=1.2,  tf32=1100, tf16=2200, fp64=1.2, fp32=75,   fp16=2200,
                      dram=7700, pcie=128),
}

BASELINE_REF = "A100-40GB"
BASELINE_TGT = "B300"

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


def _draw_baselines(ax, baselines, xlim, show_label):
    """Vertical spec-ratio lines on one panel.

    Lines beyond the x-range still get a legend entry (the clipped artist
    exists) and are marked with an arrow at the panel edge.
    """
    offaxis = 0
    for name, value in baselines:
        style = BASELINE_STYLES[name]
        label = f"Spec Scale: {name} ({value:.2f}\u00d7)" if show_label else "_nolegend_"
        ax.axvline(value, linewidth=BASELINE_LW, zorder=3, label=label, **style)
        if not (xlim[0] <= value <= xlim[1]):
            right = value > xlim[1]
            ax.text(
                0.995 if right else 0.005, 0.92 - 0.10 * offaxis,
                (f"{name} {value:.2f}\u00d7 \u2192" if right
                 else f"\u2190 {name} {value:.2f}\u00d7"),
                transform=ax.transAxes, ha="right" if right else "left",
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


# y-axis mode -> left-axis label. The unit is carried by the tick labels
# themselves for the normalized modes, so the label stays short.
YMODE_LABELS = {
    "absolute": "Node-hours",
    "fraction": "Fraction of Node-Hours",
    "percent": "Node-Hours (% of total)",
}


def _nice_upper_limit(value):
    """Round `value` up to 1, 2, 2.5, or 5 times a power of ten."""
    if value <= 0:
        return 1.0
    exp = np.floor(np.log10(value))
    frac = value / 10**exp
    for step in (1.0, 1.65, 2.0, 2.5, 5.0, 10.0):
        if frac <= step:
            return step * 10**exp
    return 10 ** (exp + 1)


def _draw_panel(
    ax,
    result_df,
    gpu_name,
    color,
    edgecolor,
    bins,
    y_mode,
    denom,
    show_xlabel,
    show_ylabels=True,
    legend_label=None,
    baselines=None,
    xlim=None,
    baseline_legend=True,
    legend_anchor=None,
    legend_fontsize=16,
):
    """Draw a single speedup histogram + cumulative % panel onto `ax`.

    `denom` is the node-hour total used for normalization; pass the *same*
    value for both panels when the two panels describe the same job set, so
    the bar heights are directly comparable.
    `show_ylabels` controls whether this panel gets the left/right axis titles
    (in the stacked figure only one panel carries them, centered on both).
    `baselines` is an optional [(name, speedup)] list of spec-ratio lines.
    `legend_anchor` = (x, y) puts the legend's upper-right corner at speedup x
    and axes-fraction height y; None keeps the original upper-left placement.

    Returns (ax2, counts, bin_edges, leg, cum_line) so the caller can autoscale
    and check the legend placement.
    """
    df = result_df[["speedup", "node_hours"]].dropna()
    if df.empty:
        raise ValueError("No valid (non-NaN) rows to plot.")
    s = df["speedup"].to_numpy()

    node_hours = df["node_hours"].to_numpy()
    total = node_hours.sum()

    # --- Normalization -------------------------------------------------
    if y_mode == "absolute":
        weights = node_hours
    elif y_mode == "fraction":
        weights = node_hours / denom
    elif y_mode == "percent":
        weights = 100.0 * node_hours / denom
    else:
        raise ValueError(f"Unknown y_mode: {y_mode!r}")
    # -------------------------------------------------------------------

    ax2 = ax.twinx()

    hist_label = legend_label if legend_label is not None else gpu_name

    counts, bin_edges, patches = ax.hist(
        s,
        bins=bins,
        weights=weights,
        edgecolor=edgecolor,
        color=color,
        label=hist_label,
        histtype="stepfilled",
        alpha=0.5,
        linewidth=2,
    )

    # Baselines are drawn after the histogram so the legend lists it first.
    if baselines:
        _draw_baselines(ax, baselines, xlim, show_label=baseline_legend)

    # Sanity check: warn if data falls outside the bin range, in which case
    # the bars will not sum to 1 (or 100%).
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
    (cum_line,) = ax2.plot(
        s[order],
        cum,
        linestyle=(0, (5, 1)),
        linewidth=2,
        color="mediumorchid",
        label=f"{gpu_name} cumulative %",
    )

    # X label only on the bottom panel.
    if show_xlabel:
        ax.set_xlabel("Speedup Relative to A100", fontsize=26)
    # Y titles only on the panel that carries the shared titles.
    if show_ylabels:
        ax.set_ylabel(YMODE_LABELS[y_mode], fontsize=21)
        ax2.set_ylabel("Cumulative Percentage (%)", fontsize=21)

    ax2.set_ylim(0, 110)

    ax2.tick_params(which="both", direction="in", labelsize=21)

    handles, labels = ax.get_legend_handles_labels()
    leg_kw = dict(frameon=True, framealpha=1.0, edgecolor="black", facecolor="white")
    if legend_anchor is None:
        leg = ax2.legend(handles, labels, loc="upper left", fontsize=18, **leg_kw)
    else:
        # x in speedup units, y in axes fraction (independent of y_mode).
        leg = ax2.legend(
            handles, labels, loc="upper right", fontsize=legend_fontsize,
            handlelength=1.6, borderpad=0.4, labelspacing=0.4,
            bbox_to_anchor=legend_anchor,
            bbox_transform=blended_transform_factory(ax.transData, ax.transAxes),
            **leg_kw,
        )
    leg.set_zorder(20)

    # Keep ax transparent so the twin axis (ax2) content shows through.
    ax2.set_zorder(ax.get_zorder() + 1)
    ax2.patch.set_visible(False)

    frame_linewidth = 2
    for spine in ["top", "right", "bottom", "left"]:
        ax.spines[spine].set_linewidth(frame_linewidth)

    return ax2, counts, bin_edges, leg, cum_line


def _default_legend_x(xlim, baselines, gap=0.07):
    """Right edge for the legend: just left of a baseline line that sits near
    the right edge of the axis (so the legend never covers it), else the edge."""
    span = xlim[1] - xlim[0]
    near = [v for _, v in (baselines or []) if xlim[1] - 0.25 * span < v <= xlim[1]]
    return (min(near) - gap) if near else (xlim[1] - 0.02 * span)


def _check_legend_overlap(name, fig, ax, leg, counts, bin_edges, cum_line, baselines):
    """Warn if the legend box covers bars, a baseline line or the cumulative curve."""
    fig.canvas.draw()
    bb = leg.get_window_extent(fig.canvas.get_renderer())
    (x0, y0), (x1, y1) = ax.transData.inverted().transform([[bb.x0, bb.y0], [bb.x1, bb.y1]])
    problems = []
    lefts, rights = bin_edges[:-1], bin_edges[1:]
    covered = (rights > x0) & (lefts < x1) & (counts > y0)
    if covered.any():
        problems.append(f"{covered.sum()} histogram bar(s)")
    hit = [n for n, v in (baselines or []) if x0 <= v <= x1]
    if hit:
        problems.append("baseline line(s): " + ", ".join(hit))
    pts = cum_line.get_transform().transform(cum_line.get_xydata())
    inside = ((pts[:, 0] > bb.x0) & (pts[:, 0] < bb.x1) & (pts[:, 1] > bb.y0) & (pts[:, 1] < bb.y1))
    if inside.any():
        problems.append("the cumulative curve")
    if problems:
        print(f"Warning ({name}): legend overlaps " + "; ".join(problems)
              + ". Adjust --top/--bottom-legend-anchor or --legend-fontsize.")
    else:
        print(f"{name}: legend placement clear "
              f"(x {x0:.2f}-{x1:.2f}, y {y0:.2f}-{y1:.2f} on the left axis).")


def plot_speedup_distribution_stacked(
    top_df,
    bottom_df,
    outpath,
    bins,
    y_mode="percent",
    shared_denominator=True,
    xlim=(-0.1, 4.1),
    xtick_step=0.2,
    baselines=None,
    baseline_offaxis="extend",
    legend_right=True,
    top_legend_anchor=None,
    bottom_legend_anchor=None,
    legend_fontsize=16,
    fig_height=7.0,
):
    """Two stacked panels sharing the x-axis, no vertical gap between them."""
    # Widen the x-range so every baseline line is visible ("extend"), or keep
    # it and mark off-axis lines with an arrow at the edge ("arrow").
    if baselines and baseline_offaxis == "extend":
        vmax = max(v for _, v in baselines)
        if vmax > xlim[1]:
            new_hi = np.ceil(vmax / xtick_step - 1e-9) * xtick_step + 0.1
            print(f"Extending x-axis upper limit from {xlim[1]} to {new_hi:.2f} "
                  f"to show the {vmax:.2f}x baseline.")
            xlim = (xlim[0], float(new_hi))

    # Legends go into the empty space on the right: upper-right corner just
    # left of the rightmost baseline line and just below the cumulative
    # curve's 100% plateau (100/110 of the right axis = 0.909).
    if legend_right:
        x_right = _default_legend_x(xlim, baselines)
        top_anchor = tuple(top_legend_anchor) if top_legend_anchor else (x_right, 0.87)
        bot_anchor = tuple(bottom_legend_anchor) if bottom_legend_anchor else (x_right, 0.87)
    else:
        top_anchor = bot_anchor = None

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(14, fig_height), sharex=True, gridspec_kw={"hspace": 0.0}
    )

    top_total = total_node_hours(top_df)
    bot_total = total_node_hours(bottom_df)

    if shared_denominator:
        # Same job set in both panels -> one denominator keeps the two panels
        # on exactly the same footing.
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
    ax2_top, counts_top, edges_top, leg_top, cum_top = _draw_panel(
        ax_top,
        top_df,
        gpu_name="Blackwell-Ultra",
        color="palegreen",
        edgecolor="forestgreen",
        bins=bins,
        y_mode=y_mode,
        denom=denom_top,
        show_xlabel=False,
        show_ylabels=True,   # the shared titles live on the top panel
        baselines=baselines,
        xlim=xlim,
        baseline_legend=True,
        legend_anchor=top_anchor,
        legend_fontsize=legend_fontsize,
    )

    ax2_bot, counts_bot, edges_bot, leg_bot, cum_bot = _draw_panel(
        ax_bot,
        bottom_df,
        gpu_name="Blackwell-Ultra-NG4",
        color="lightskyblue",
        edgecolor="dodgerblue",
        bins=bins,
        y_mode=y_mode,
        denom=denom_bot,
        show_xlabel=True,
        show_ylabels=False,
        # Wrapped onto two lines so the legend fits the narrow right-hand gap.
        legend_label=("Hypothetical-Blackwell-Ultra\n(Non-Kernel Portion Scale Up 4x)"),
        baselines=baselines,
        xlim=xlim,
        baseline_legend=False,
        legend_anchor=bot_anchor,
        legend_fontsize=legend_fontsize,
    )

    ymax_data = max(counts_top.max(), counts_bot.max())

    # Keep at most ~21 labelled ticks (as in the original 0-4 axis); on a wider
    # axis, label every other tick and keep the rest as unlabelled minor ticks.
    xticks_minor = np.round(
        np.arange(np.ceil(xlim[0] / xtick_step - 1e-9) * xtick_step,
                  xlim[1] + 1e-9, xtick_step), 10)
    label_step = xtick_step * int(np.ceil(len(xticks_minor) / 21))
    xticks = np.round(
        np.arange(np.ceil(xlim[0] / label_step - 1e-9) * label_step,
                  xlim[1] + 1e-9, label_step), 10)

    for ax, ax2 in ((ax_top, ax2_top), (ax_bot, ax2_bot)):
        is_top = ax is ax_top
        # Round the top of the axis up to a "nice" number with ~15% headroom
        # so the legend does not overlap the tallest bar.
        ax.set_ylim(0, _nice_upper_limit(1.15 * ymax_data))
        # The bottom panel drops a left tick that sits exactly on its top edge,
        # so it does not collide with the top panel's 0% at the shared boundary.
        ytop = ax.get_ylim()[1]
        yticks = mticker.MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10]).tick_values(0, ytop)
        eps = 1e-9 * ytop
        yticks = [t for t in yticks
                  if -eps <= t <= ytop + eps and (is_top or t < ytop - eps)]
        ax.yaxis.set_major_locator(mticker.FixedLocator(yticks))
        ax2.yaxis.set_major_locator(mticker.FixedLocator([0, 25, 50, 75, 100]))

        # Append "%" to the left-axis tick labels. `xmax` is the value that
        # corresponds to 100%, so it must match the chosen y_mode.
        if y_mode == "percent":
            ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100, decimals=0))
        elif y_mode == "fraction":
            ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))

        ax.set_xlim(*xlim)
        ax.set_xticks(xticks)
        if label_step > xtick_step:
            ax.set_xticks(xticks_minor, minor=True)
            ax.tick_params(axis="x", which="minor", length=3, width=1.5)

        ax.tick_params(axis="x", length=4.5, width=2)
        ax.tick_params(which="both", direction="in", labelsize=21)

    # One left and one right y-axis title for the whole figure: keep them on the
    # top panel but center them on the panel boundary (y = 0 in the top panel's
    # axes coordinates, since hspace = 0). Matplotlib still places them
    # horizontally clear of the tick labels.
    ax_top.yaxis.label.set_y(0.0)
    ax2_top.yaxis.label.set_y(0.0)

    if legend_right:
        _check_legend_overlap("top panel", fig, ax_top, leg_top, counts_top,
                              edges_top, cum_top, baselines)
        _check_legend_overlap("bottom panel", fig, ax_bot, leg_bot, counts_bot,
                              edges_bot, cum_bot, baselines)

    fig.savefig(outpath, dpi=300, format="png", bbox_inches="tight")
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot two stacked node-hours-weighted speedup distributions "
        "(shared x-axis) from two folders of parquet files."
    )
    parser.add_argument(
        "--top-input-dir", required=True, help="Folder of parquet files for the TOP panel."
    )
    parser.add_argument(
        "--bottom-input-dir", required=True, help="Folder of parquet files for the BOTTOM panel."
    )
    parser.add_argument(
        "--outpath",
        default="speedup_distribution_stacked.png",
        help="Path to save the output plot.",
    )
    parser.add_argument(
        "--y_mode",
        choices=("absolute", "fraction", "percent"),
        default="percent",
        help="Left y-axis unit: raw node-hours, fraction of total node-hours "
        "(bars sum to 1), or percent of total node-hours (bars sum to 100). "
        "Both normalized modes are labelled with a %% sign. Default: percent.",
    )
    parser.add_argument(
        "--per-panel-denominator",
        dest="shared_denominator",
        action="store_false",
        help="Normalize each panel by its own node-hour total instead of using "
        "a single shared denominator.",
    )
    parser.add_argument(
        "--max-node-hours",
        type=float,
        default=720,
        help="Filter out rows with node_hours greater than this value (default: 720).",
    )
    parser.add_argument(
        "--xlim", type=float, nargs=2, default=(-0.1, 4.1), metavar=("XMIN", "XMAX"),
        help="Speedup range shown on the x-axis (default: -0.1 4.1).",
    )
    parser.add_argument("--no-baselines", dest="baselines", action="store_false",
                        help="Do not draw the spec-ratio baseline lines.")
    parser.add_argument(
        "--baseline-offaxis", choices=("extend", "arrow"), default="extend",
        help="For baseline lines beyond --xlim: widen the x-axis to show them "
        "('extend', default) or keep the x-axis and mark them with an arrow "
        "at the panel edge ('arrow').",
    )
    parser.add_argument(
        "--legend-left", dest="legend_right", action="store_false",
        help="Keep the original upper-left legends instead of moving them right.",
    )
    parser.add_argument(
        "--top-legend-anchor", type=float, nargs=2, metavar=("X", "Y"), default=None,
        help="Upper-right corner of the top legend: X in speedup units, Y as a "
        "fraction of the panel height (default: auto, just left of the "
        "rightmost baseline line, Y=0.87).",
    )
    parser.add_argument(
        "--bottom-legend-anchor", type=float, nargs=2, metavar=("X", "Y"), default=None,
        help="Upper-right corner of the bottom legend (same units as above).",
    )
    parser.add_argument("--legend-fontsize", type=float, default=16,
                        help="Legend font size when legends are on the right (default: 16).")
    parser.add_argument("--fig-height", type=float, default=7.0,
                        help="Total figure height in inches for both panels "
                        "(default: 7.0; previously 10). Width stays 14.")
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
        xlim=tuple(args.xlim),
        baselines=baselines,
        baseline_offaxis=args.baseline_offaxis,
        legend_right=args.legend_right,
        top_legend_anchor=args.top_legend_anchor,
        bottom_legend_anchor=args.bottom_legend_anchor,
        legend_fontsize=args.legend_fontsize,
        fig_height=args.fig_height,
    )
    print(f"Plot saved to {args.outpath}")


if __name__ == "__main__":
    main()