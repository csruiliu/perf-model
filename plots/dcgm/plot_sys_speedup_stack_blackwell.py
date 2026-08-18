import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd


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
    combined = combined[combined["node_hours"] <= max_node_hours]
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
    legend_label=None,
):
    """Draw a single speedup histogram + cumulative % panel onto `ax`.

    `denom` is the node-hour total used for normalization; pass the *same*
    value for both panels when the two panels describe the same job set, so
    the bar heights are directly comparable.

    Returns (ax2, counts) so the caller can align axes and autoscale.
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
    ax2.plot(
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
    ax.set_ylabel(YMODE_LABELS[y_mode], fontsize=21)
    ax2.set_ylabel("Cumulative Percentage (%)", fontsize=20)

    ax2.set_ylim(0, 110)

    ax2.tick_params(which="both", direction="in", labelsize=23)

    handles, labels = ax.get_legend_handles_labels()
    leg = ax2.legend(
        handles,
        labels,
        loc="upper left",
        fontsize=18,
        frameon=True,
        framealpha=1.0,
        edgecolor="black",
        facecolor="white",
    )
    leg.set_zorder(20)

    # Keep ax transparent so the twin axis (ax2) content shows through.
    ax2.set_zorder(ax.get_zorder() + 1)
    ax2.patch.set_visible(False)

    frame_linewidth = 2
    for spine in ["top", "right", "bottom", "left"]:
        ax.spines[spine].set_linewidth(frame_linewidth)

    return ax2, counts


def plot_speedup_distribution_stacked(
    top_df,
    bottom_df,
    outpath,
    bins,
    y_mode="percent",
    shared_denominator=True,
):
    """Two stacked panels sharing the x-axis, no vertical gap between them."""
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(14, 10), sharex=True, gridspec_kw={"hspace": 0.0}
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

    _, counts_top = _draw_panel(
        ax_top,
        top_df,
        gpu_name="Blackwell-Ultra",
        color="palegreen",
        edgecolor="forestgreen",
        bins=bins,
        y_mode=y_mode,
        denom=denom_top,
        show_xlabel=False,
    )

    _, counts_bot = _draw_panel(
        ax_bot,
        bottom_df,
        gpu_name="Blackwell-Ultra-NG4",
        color="lightskyblue",
        edgecolor="dodgerblue",
        bins=bins,
        y_mode=y_mode,
        denom=denom_bot,
        show_xlabel=True,
        legend_label="Hypothetical-Blackwell-Ultra\n(Non-GPU Portion Scale Up 4x)",
    )

    ymax_data = max(counts_top.max(), counts_bot.max())

    for ax in (ax_top, ax_bot):
        # Round the top of the axis up to a "nice" number with ~15% headroom
        # so the legend does not overlap the tallest bar.
        ax.set_ylim(0, _nice_upper_limit(1.15 * ymax_data))
        ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=6, steps=[1, 2, 2.5, 5, 10]))

        # Append "%" to the left-axis tick labels. `xmax` is the value that
        # corresponds to 100%, so it must match the chosen y_mode.
        if y_mode == "percent":
            ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100, decimals=0))
        elif y_mode == "fraction":
            ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))

        ax.set_xlim(-0.1, 4.1)
        ax.set_xticks(np.arange(0, 4 + 0.01, 0.2))

        ax.tick_params(axis="x", length=4.5, width=2)
        ax.tick_params(which="both", direction="in", labelsize=19)

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
    return parser.parse_args()


def main():
    args = parse_args()

    top_df = load_parquet_folder(args.top_input_dir, max_node_hours=args.max_node_hours)
    bottom_df = load_parquet_folder(args.bottom_input_dir, max_node_hours=args.max_node_hours)

    plot_speedup_distribution_stacked(
        top_df=top_df,
        bottom_df=bottom_df,
        outpath=args.outpath,
        bins=np.arange(0, 4.1, 0.05),
        y_mode=args.y_mode,
        shared_denominator=args.shared_denominator,
    )
    print(f"Plot saved to {args.outpath}")


if __name__ == "__main__":
    main()