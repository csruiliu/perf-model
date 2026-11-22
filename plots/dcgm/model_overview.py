"""
Overview figure for "Towards Workload-Scale Performance Prediction
From Passive GPU Telemetry".

Panels
  (a) Actual on/off activity inside one reference sample (profiler-style
      trace) overlaid with what DCGM records: the dashed line is the
      interval average, which is all that survives; timing is lost.
  (b) Model projection: the reference interval is decomposed into
      kernel / expected overlap / PCIe / residual (Eqs. 2-4, 13, 14) and each
      part is rescaled to the target; tau_tgt is solved from Eq. 17.

All numbers in both panels are computed from the timeline in CONFIG using
the model equations, so the figure stays internally consistent if you edit
the CONFIG block.

Usage:  python fig_model_overview.py      -> fig_model_overview.pdf / .png
        FONT_STYLE=serif python fig_model_overview.py   (Times version)
        SANS_FONT="Roboto" python fig_model_overview.py

Font setup on Debian (README)
  Roboto (TrueType, works as is):
      sudo apt install fonts-roboto
  Helvetica look (free clone, ships as .otf; convert once to .ttf):
      sudo apt install fonts-texgyre fontforge
      mkdir -p ~/.local/share/fonts
      for s in regular italic bold bolditalic; do
        fontforge -lang=ff -c 'Open($1); Generate($2)' \
          /usr/share/texmf/fonts/opentype/public/tex-gyre/texgyreheros-$s.otf \
          ~/.local/share/fonts/texgyreheros-$s.ttf
      done
      fc-cache -f
  Then clear matplotlib's font cache:  rm -rf ~/.cache/matplotlib
  Check embedding:  pdffonts fig_model_overview.pdf   (type should be TrueType)
"""

import os

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle

# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------
TAU = 10.0  # sampling interval tau (s); Perlmutter collects at 0.1 Hz

# Illustrative reference timeline (start, end) in seconds.
KERNELS = [(0.6, 2.2), (2.8, 4.6), (5.2, 6.4), (7.4, 9.0)]
PCIE = [(0.0, 0.8), (4.3, 5.0), (8.8, 9.3)]

# Illustrative target scaling (pick to match the text, e.g. A100 -> H100).
KERNEL_SPEEDUP = 2.0   # smallest ceiling ratio in Eq. 12
PCIE_SPEEDUP = 2.0     # beta_pcie^tgt / beta_pcie^ref (64 / 32 GB/s)
RHO_RES = 1.0          # residual scale factor (Eq. 15)

SHOW_OTHER = True      # gray "Other" lane in (a); no dashed mean (DCGM does not report it)

FONT_STYLE = os.environ.get("FONT_STYLE", "sans")   # "sans" (modern) or "serif" (Times, matches body)
# Sans-serif face to use. Helvetica-style options on Debian:
#   "TeX Gyre Heros" (apt: fonts-texgyre)  or  "Nimbus Sans" (apt: fonts-urw-base35)
# Other:  "Roboto" (apt: fonts-roboto)
SANS_FONT = os.environ.get("SANS_FONT", "TeX Gyre Heros")

FIG_WIDTH = 3.5        # in; IEEE single column (use 7.16 for figure*)
FIG_HEIGHT = 2.35 if not SHOW_OTHER else 2.6
OUT = "fig_model_overview" if FONT_STYLE == "sans" else "fig_model_overview_serif"

C_KERNEL = "#1D9E75"   # teal
C_PCIE = "#D85A30"     # coral
C_RES = "#B4B2A9"      # gray
C_TEXT = "#2C2C2A"

# ----------------------------------------------------------------------------
# Model quantities (computed, do not edit)
# ----------------------------------------------------------------------------
def total(iv):
    return sum(e - s for s, e in iv)

def union(a, b):
    ev = sorted(a + b)
    merged = []
    for s, e in ev:
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [tuple(m) for m in merged]

def complement(iv, lo=0.0, hi=TAU):
    gaps, cur = [], lo
    for s, e in iv:
        if s > cur:
            gaps.append((cur, s))
        cur = max(cur, e)
    if cur < hi:
        gaps.append((cur, hi))
    return gaps

OTHER = complement(union(KERNELS, PCIE))

# Reference side
t_k_ref = total(KERNELS)                       # = tau * A_gract   (Eq. 4)
t_p_ref = total(PCIE)                          # PCIe time          (Eq. 13)
A_gract = t_k_ref / TAU
pcie_frac = t_p_ref / TAU   # t_pcie / tau (duty cycle used in Eq. 3)
ov_ref = t_k_ref * t_p_ref / TAU               # expected overlap   (Eq. 3)
t_res_ref = max(TAU - t_k_ref - t_p_ref + ov_ref, 0.0)   # residual (Eq. 14)

# Target side
t_k_tgt = t_k_ref / KERNEL_SPEEDUP             # Eq. 12
t_p_tgt = t_p_ref / PCIE_SPEEDUP               # Eq. 13
t_res_tgt = t_res_ref * RHO_RES                # Eq. 15
S = t_k_tgt + t_p_tgt + t_res_tgt
disc = S ** 2 - 4 * t_k_tgt * t_p_tgt
tau_tgt = (S + disc ** 0.5) / 2                # Eq. 17 (larger root)
ov_tgt = t_k_tgt * t_p_tgt / tau_tgt

print(f"A_gract={A_gract:.2f}  t_pcie/tau={pcie_frac:.2f}")
print(f"ref: kernel={t_k_ref:.2f} pcie={t_p_ref:.2f} overlap={ov_ref:.2f} "
      f"residual={t_res_ref:.2f}")
print(f"tgt: kernel={t_k_tgt:.2f} pcie={t_p_tgt:.2f} overlap={ov_tgt:.2f} "
      f"residual={t_res_tgt:.2f} tau_tgt={tau_tgt:.2f}")

# ----------------------------------------------------------------------------
# Style
# ----------------------------------------------------------------------------
def prefer_ttf(family):
    """Use TrueType files for `family` when both .otf and .ttf are installed.

    matplotlib embeds CFF-based .otf fonts in PDFs with a mismatched font
    type (pdffonts: "Mismatch between font type and embedded font file"),
    which PDF checkers such as IEEE PDF eXpress may reject.
    """
    from matplotlib import font_manager as fm
    entries = [f for f in fm.fontManager.ttflist if f.name == family]
    if any(f.fname.lower().endswith(".ttf") for f in entries):
        fm.fontManager.ttflist = [
            f for f in fm.fontManager.ttflist
            if not (f.name == family and f.fname.lower().endswith(".otf"))]
        fm.fontManager._findfont_cached.cache_clear()
    elif entries:
        print(f"note: only .otf files found for '{family}'; convert them to "
              ".ttf for clean PDF embedding (see README in the script header)")

if FONT_STYLE == "sans":
    prefer_ttf(SANS_FONT)
    # Modern look: clean sans-serif with math set in the same face.
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": [SANS_FONT, "Helvetica", "Arial",
                            "Liberation Sans", "DejaVu Sans"],
        "mathtext.fontset": "custom",
        "mathtext.cal": "sans",   # silences the 'cursive' findfont warning
        "mathtext.rm": "sans",
        "mathtext.it": "sans:italic",
        "mathtext.bf": "sans:bold",
    })
else:
    # Classic IEEE look: Times to match the body text.
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "STIXGeneral",
                       "DejaVu Serif"],
        "mathtext.fontset": "stix",
    })

mpl.rcParams.update({
    "font.size": 7.5,
    "axes.titlesize": 7.5,
    "axes.labelsize": 7.5,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.linewidth": 0.5,
    "xtick.major.width": 0.5,
    "xtick.major.size": 2,
    "hatch.linewidth": 1.5,   # orange overlap hatch in (b) and legend
    "pdf.fonttype": 42,   # embed TrueType (IEEE PDF eXpress friendly)
    "ps.fonttype": 42,
})

fig, (ax_a, ax_b) = plt.subplots(
    2, 1, figsize=(FIG_WIDTH, FIG_HEIGHT), sharex=True,
    gridspec_kw={"height_ratios": [1.25 if not SHOW_OTHER else 1.6, 1.0],
                 "hspace": 0.5},
)

def clean(ax):
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, TAU)

def title(ax, text):
    ax.set_title(text, loc="left", pad=3, color=C_TEXT)

# ----------------------------------------------------------------------------
# (a) Actual activity + what DCGM records
# ----------------------------------------------------------------------------
LANE_H = 0.8   # height of an "on" pulse

def activity_lane(ax, iv, base, col, mean=None, label=None):
    """On/off trace for one activity; dashed line = DCGM interval average."""
    xs, ys = [0.0], [base]
    for s, e in iv:
        xs += [s, s, e, e]
        ys += [base, base + LANE_H, base + LANE_H, base]
    xs.append(TAU); ys.append(base)
    ax.fill_between(xs, base, ys, color=col, alpha=0.30, lw=0)
    ax.plot(xs, ys, color=col, lw=0.9, solid_joinstyle="miter")
    if mean is not None:
        ax.hlines(base + LANE_H * mean, 0, TAU, colors=col,
                  linestyles=(0, (3, 2)), lw=0.9)
        ax.text(TAU * 1.01, base + LANE_H * mean, label, color=col,
                va="center", ha="left", fontsize=7, clip_on=False)

lanes = [("Kernels", KERNELS, C_KERNEL, A_gract,
          rf"$A_{{\mathrm{{gract}}}}={A_gract:.2f}$"),
         ("PCIe", PCIE, C_PCIE, pcie_frac,
          rf"$A_{{\mathrm{{pcie}}}}={pcie_frac:.2f}$")]
if SHOW_OTHER:
    lanes.append(("Residual", OTHER, C_RES, None, None))

STEP = 1.15
yticks = []
for i, (name, iv, col, mean, label) in enumerate(lanes):
    base = (len(lanes) - 1 - i) * STEP
    activity_lane(ax_a, iv, base, col, mean, label)
    yticks.append(base + LANE_H / 2)
ax_a.set_yticks(yticks)
ax_a.set_yticklabels([name for name, *_ in lanes])
ax_a.set_ylim(-0.1, (len(lanes) - 1) * STEP + LANE_H + 0.1)
clean(ax_a)
title(ax_a, r"(a) Actual activity; DCGM reports only the dashed mean")

# ----------------------------------------------------------------------------
# (b) Projection
# ----------------------------------------------------------------------------
def stacked(ax, y, tk, tp, ov, tr, h=0.62):
    segs = [(tk - ov, C_KERNEL, None),          # kernel only
            (ov, C_KERNEL, "//////"),             # expected overlap
            (tp - ov, C_PCIE, None),            # PCIe only
            (tr, C_RES, None)]                  # residual
    x = 0.0
    for w, col, hatch in segs:
        if hatch:
            ax.add_patch(Rectangle((x, y), w, h, facecolor=C_KERNEL,
                                   edgecolor=C_PCIE, hatch=hatch, lw=0))
        else:
            ax.add_patch(Rectangle((x, y), w, h, facecolor=col,
                                   edgecolor="white", lw=0.4))
        x += w
    return x

end_ref = stacked(ax_b, 1.15, t_k_ref, t_p_ref, ov_ref, t_res_ref)
end_tgt = stacked(ax_b, 0.15, t_k_tgt, t_p_tgt, ov_tgt, t_res_tgt)
ax_b.text(end_ref - 0.08, 1.15 + 0.31, rf"$\tau^{{\mathrm{{ref}}}}={TAU:.0f}$s", ha="right",
          va="center", fontsize=7, color=C_TEXT)
ax_b.text(end_tgt + 0.12, 0.15 + 0.31,
          rf"$\tau^{{\mathrm{{tgt}}}}={tau_tgt:.1f}$s",
          ha="left", va="center", fontsize=7, color=C_TEXT)
ax_b.set_yticks([1.46, 0.46])
ax_b.set_yticklabels(["Reference", "Target"])
ax_b.set_ylim(0.0, 1.95)
clean(ax_b)
ax_b.set_xlabel("Time within sample (s)", labelpad=1.5)
title(ax_b, "(b) Each component rescaled; target interval solved")

# ----------------------------------------------------------------------------
# Shared legend
# ----------------------------------------------------------------------------
handles = [Patch(facecolor=C_KERNEL, label="Kernel"),
           Patch(facecolor=C_PCIE, label="PCIe"),
           Patch(facecolor=C_KERNEL, edgecolor=C_PCIE, hatch="////", lw=0,
                 label="Potential overlap"),
           Patch(facecolor=C_RES, label="Residual")]
fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False,
           bbox_to_anchor=(0.5, 1.005), handlelength=1.2, columnspacing=0.9,
           handletextpad=0.4)

# Margins in inches, converted to figure fractions so they survive height changes.
fig.subplots_adjust(left=0.17, right=0.81,
                    top=1 - 0.42 / FIG_HEIGHT, bottom=0.36 / FIG_HEIGHT)
fig.savefig(f"{OUT}.pdf")
fig.savefig(f"{OUT}.png", dpi=300)
print(f"wrote {OUT}.pdf and {OUT}.png")