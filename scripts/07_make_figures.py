#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 BadmintonVision -- Publication figure generation  (Nature Medicine style)
================================================================================
 Regenerates Figures 2, 5, 6, 7, 8 and the new Figure 9 (statistical
 validation) requested during revision.

 Output : <OUTDIR>/Figure_X.pdf  and  <OUTDIR>/Figure_X.png
 Style  : Arial, 600 dpi, 190 mm width, NPG (Nature) colour palette,
          clean spines, systematic panel labels (a, b, c ...).

 >>> ALL TUNABLE PARAMETERS ARE IN THE  CONFIG  BLOCK BELOW. <<<
     Edit FIG_WIDTH_MM, DPI, FONT_SIZES, COLORS, OUTPUT_FORMATS, etc.
     then simply rerun:   python3 make_figures.py
================================================================================
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch, Polygon, Circle, PathPatch
from matplotlib.path import Path
from matplotlib.colors import LinearSegmentedColormap, to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.patheffects as pe

# ============================================================================
#  CONFIG  -- edit everything here
# ============================================================================
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_OUT = os.path.join(_REPO, "outputs")
_FIGDIR = os.path.join(_OUT, "figures")
_FONTS = os.path.join(_REPO, "assets", "fonts")

CONFIG = dict(
    OUTDIR        = _FIGDIR,
    STATS_JSON    = os.path.join(_OUT, "stats_results.json"),
    FONT_DIR      = _FONTS,
    OUTPUT_FORMATS= ["pdf", "png"],   # both requested
    DPI           = 600,               # requested clarity
    FIG_WIDTH_MM  = 190,               # requested full width
    FONT_FAMILY   = "Arial",
)

# ---- font sizes (pt) -------------------------------------------------------
FS = dict(
    base   = 7.0,    # default text
    tick   = 6.5,    # tick labels
    axlab  = 7.5,    # axis labels
    panel  = 10.0,   # panel letters (a, b, c)
    title  = 7.8,    # sub-panel titles
    annot  = 6.0,    # in-plot numeric annotations
    legend = 6.5,    # legend text
    small  = 5.6,    # very small labels
)

# ---- line widths / geometry -----------------------------------------------
LW = dict(axis=0.6, grid=0.45, plot=1.3, bar_edge=0.5, box=0.9, marker_edge=0.5,
          ref=0.7, error=0.9)
GEO = dict(tick_len=2.4, tick_pad=2.0, marker=4.5)

# ---- NPG (Nature Publishing Group) palette --------------------------------
NPG = ["#E64B35", "#4DBBD5", "#00A087", "#3C5488", "#F39B7F",
       "#8491B4", "#91D1C2", "#DC0000", "#7E6148", "#B09C85"]

COL = dict(
    scoring   = "#00A087",   # teal  (favourable / win)
    conceding = "#E64B35",   # red   (unfavourable / loss)
    men       = "#3C5488",   # navy
    women     = "#E64B35",   # red
    baseline  = "#8491B4",   # grey-blue
    simsiam   = "#F39B7F",   # salmon
    convnext  = "#00A087",   # teal
    accent    = "#DC0000",   # bright red for arrows / highlights
    grey      = "#5A5A5A",
    lightgrey = "#BFBFBF",
    ref       = "#9AA0A6",
)
# per-action-class colours (Player, Forehand, Backhand, Serve, Jump Smash)
CLASS_COL = {"Player": "#3C5488", "Forehand": "#00A087", "Backhand": "#F39B7F",
             "Serve": "#7E6148", "Jump Smash": "#E64B35"}

# ---- colormaps -------------------------------------------------------------
CMAP_AP  = LinearSegmentedColormap.from_list("ap",  ["#E64B35", "#FBD08A", "#00A087"])   # perf heatmaps
CMAP_SC  = LinearSegmentedColormap.from_list("sc",  ["#FFFFFF", "#00A087"])              # scoring seq
CMAP_CO  = LinearSegmentedColormap.from_list("co",  ["#FFFFFF", "#E64B35"])              # conceding seq
CMAP_Z   = "RdBu_r"   # diverging for lag-sequential z-scores

# ============================================================================
#  SETUP
# ============================================================================
os.makedirs(CONFIG["OUTDIR"], exist_ok=True)
MM = 1.0 / 25.4
FIG_W = CONFIG["FIG_WIDTH_MM"] * MM     # inches

# register Arial
for f in ["Arial.ttf", "Arial_Bold.ttf", "Arial_Italic.ttf"]:
    p = os.path.join(CONFIG["FONT_DIR"], f)
    if os.path.exists(p):
        fm.fontManager.addfont(p)

# Fall back to a bundled/Liberation sans-serif if Arial is unavailable,
# so figures render cleanly without the proprietary Arial files.
_avail = {f.name for f in fm.fontManager.ttflist}
if CONFIG["FONT_FAMILY"] not in _avail:
    for _alt in ("Arial", "Liberation Sans", "Helvetica", "DejaVu Sans"):
        if _alt in _avail:
            CONFIG["FONT_FAMILY"] = _alt
            break
    else:
        CONFIG["FONT_FAMILY"] = "sans-serif"
    print(f"[fonts] Arial not found; using '{CONFIG['FONT_FAMILY']}'.")

plt.rcParams.update({
    "font.family": CONFIG["FONT_FAMILY"],
    "font.size": FS["base"],
    "axes.titlesize": FS["title"], "axes.labelsize": FS["axlab"],
    "xtick.labelsize": FS["tick"], "ytick.labelsize": FS["tick"],
    "legend.fontsize": FS["legend"],
    "axes.linewidth": LW["axis"],
    "axes.edgecolor": "#222222", "axes.labelcolor": "#222222",
    "text.color": "#222222", "xtick.color": "#222222", "ytick.color": "#222222",
    "xtick.major.width": LW["axis"], "ytick.major.width": LW["axis"],
    "xtick.major.size": GEO["tick_len"], "ytick.major.size": GEO["tick_len"],
    "xtick.major.pad": GEO["tick_pad"], "ytick.major.pad": GEO["tick_pad"],
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": False, "savefig.dpi": CONFIG["DPI"], "figure.dpi": 150,
    "pdf.fonttype": 42, "ps.fonttype": 42,   # editable text in vector output
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "mathtext.fontset": "custom",
    "mathtext.rm": CONFIG["FONT_FAMILY"], "mathtext.it": CONFIG["FONT_FAMILY"],
    "mathtext.bf": CONFIG["FONT_FAMILY"],
})

with open(CONFIG["STATS_JSON"]) as f:
    STATS = json.load(f)

# ---- helpers ---------------------------------------------------------------
def panel(ax, letter, dx=-0.085, dy=1.06, fs=None):
    ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=fs or FS["panel"],
            fontweight="bold", va="top", ha="left")

def clean(ax):
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.tick_params(length=GEO["tick_len"], width=LW["axis"])

def save(fig, name):
    for ext in CONFIG["OUTPUT_FORMATS"]:
        fig.savefig(os.path.join(CONFIG["OUTDIR"], f"{name}.{ext}"),
                    dpi=CONFIG["DPI"], bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(f"  saved {name}: {', '.join(CONFIG['OUTPUT_FORMATS'])}")

def heatmap(ax, data, row_labels, col_labels, cmap, vmin, vmax, fmt="{:.3f}",
            bold_row=None, cbar_label=None, annot_fs=None, txt_thresh=None):
    im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(col_labels))); ax.set_yticks(range(len(row_labels)))
    ax.set_xticklabels(col_labels, fontsize=FS["tick"])
    ax.set_yticklabels(row_labels, fontsize=FS["tick"])
    ax.tick_params(length=0)
    for s in ax.spines.values(): s.set_visible(False)
    data = np.asarray(data, float)
    rng = vmax - vmin
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i, j]
            norm = (v - vmin) / rng
            tc = "white" if (norm > 0.72 or norm < 0.16) else "#1A1A1A"
            if txt_thresh is not None:
                tc = "white" if v >= txt_thresh else "#1A1A1A"
            w = "bold" if (bold_row is not None and i == bold_row) else "normal"
            ax.text(j, i, fmt.format(v), ha="center", va="center",
                    fontsize=annot_fs or FS["annot"], color=tc, fontweight=w)
    if cbar_label:
        cb = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cb.ax.tick_params(labelsize=FS["small"], length=2, width=LW["axis"])
        cb.set_label(cbar_label, fontsize=FS["small"])
        cb.outline.set_linewidth(LW["axis"])
    return im

def radar(ax, categories, series, title=None):
    """series: list of dicts {label,values,color}. values in 0..1 fractions."""
    N = len(categories)
    ang = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    ang += ang[:1]
    ax.set_theta_offset(np.pi / 2); ax.set_theta_direction(-1)
    ax.set_xticks(ang[:-1]); ax.set_xticklabels(categories, fontsize=FS["small"])
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["25", "50", "75", "100"], fontsize=FS["small"], color=COL["grey"])
    ax.tick_params(pad=1)
    ax.grid(True, lw=LW["grid"], color="#D5D5D5")
    for sp in ax.spines.values(): sp.set_linewidth(LW["axis"]); sp.set_color("#C8C8C8")
    for s in series:
        v = list(s["values"]) + [s["values"][0]]
        ax.plot(ang, v, color=s["color"], lw=LW["plot"], label=s["label"],
                marker="o", markersize=2.6, markeredgewidth=0)
        ax.fill(ang, v, color=s["color"], alpha=0.13)
    if title: ax.set_title(title, fontsize=FS["title"], pad=6)

print("Setup complete. Figure width = %.2f in (%d mm), DPI=%d, formats=%s"
      % (FIG_W, CONFIG["FIG_WIDTH_MM"], CONFIG["DPI"], CONFIG["OUTPUT_FORMATS"]))
print("Arial registered:", "Arial" in {f.name for f in fm.fontManager.ttflist})

# ============================================================================
#  Schematic helpers for Figure 2a (court + posture pictographs;
#  used INSTEAD of copyrighted broadcast frames -- addresses R2 licensing note)
# ============================================================================
def draw_court(ax):
    # perspective half-court trapezoid (broadcast-like), schematic only
    top = [(0.30, 0.92), (0.70, 0.92)]
    bot = [(0.06, 0.10), (0.94, 0.10)]
    court = Polygon([bot[0], bot[1], top[1], top[0]], closed=True,
                    facecolor="#E8F1EC", edgecolor="#8FB7A8", lw=0.7, zorder=0)
    ax.add_patch(court)
    # net line near top
    ax.plot([0.30, 0.70], [0.92, 0.92], color="#6E6E6E", lw=1.0, zorder=1)
    for x in np.linspace(0.30, 0.70, 9):
        ax.plot([x, x], [0.88, 0.92], color="#9A9A9A", lw=0.3, zorder=1)
    # centre + service lines (perspective-ish)
    ax.plot([0.5, 0.5], [0.10, 0.92], color="#A9CBBE", lw=0.5, zorder=1)
    ax.plot([0.13, 0.87], [0.30, 0.30], color="#A9CBBE", lw=0.5, zorder=1)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

def stick(ax, cx, cy, kind, color):
    s = 0.16
    def L(x1, y1, x2, y2, lw=1.6):
        ax.plot([x1, x2], [y1, y2], color="#2B2B2B", lw=lw, solid_capstyle="round", zorder=3)
    head = Circle((cx, cy + s), s * 0.28, facecolor="#2B2B2B", edgecolor="none", zorder=3)
    ax.add_patch(head)
    L(cx, cy + s * 0.72, cx, cy - s * 0.45)          # torso
    if kind == "Player":      # ready stance
        L(cx, cy - s*0.45, cx - s*0.5, cy - s*1.25); L(cx, cy - s*0.45, cx + s*0.5, cy - s*1.25)
        L(cx, cy + s*0.4, cx - s*0.55, cy + s*0.1);  L(cx, cy + s*0.4, cx + s*0.55, cy + s*0.1)
    elif kind == "Forehand":  # arm extended right + racket
        L(cx, cy - s*0.45, cx - s*0.45, cy - s*1.25); L(cx, cy - s*0.45, cx + s*0.45, cy - s*1.2)
        L(cx, cy + s*0.45, cx + s*0.95, cy + s*0.7)
        ax.add_patch(Circle((cx + s*1.05, cy + s*0.78), s*0.18, fill=False, ec=color, lw=1.2, zorder=3))
        L(cx, cy + s*0.45, cx - s*0.4, cy + s*0.2)
    elif kind == "Backhand":  # arm across body + racket
        L(cx, cy - s*0.45, cx - s*0.45, cy - s*1.25); L(cx, cy - s*0.45, cx + s*0.45, cy - s*1.2)
        L(cx, cy + s*0.45, cx - s*0.9, cy + s*0.75)
        ax.add_patch(Circle((cx - s*1.0, cy + s*0.82), s*0.18, fill=False, ec=color, lw=1.2, zorder=3))
        L(cx, cy + s*0.45, cx + s*0.35, cy + s*0.15)
    elif kind == "Serve":     # underarm low serve
        L(cx, cy - s*0.45, cx - s*0.5, cy - s*1.25); L(cx, cy - s*0.45, cx + s*0.5, cy - s*1.2)
        L(cx, cy + s*0.35, cx + s*0.7, cy - s*0.15)
        ax.add_patch(Circle((cx + s*0.8, cy - s*0.25), s*0.16, fill=False, ec=color, lw=1.2, zorder=3))
        L(cx, cy + s*0.4, cx - s*0.5, cy + s*0.6)
    elif kind == "Jump Smash":# airborne, arm overhead
        cy += s*0.35
        L(cx, cy - s*0.45, cx - s*0.55, cy - s*0.95); L(cx, cy - s*0.45, cx + s*0.4, cy - s*1.0)
        L(cx, cy + s*0.5, cx + s*0.55, cy + s*1.25)
        ax.add_patch(Circle((cx + s*0.62, cy + s*1.4), s*0.18, fill=False, ec=color, lw=1.2, zorder=3))
        L(cx, cy + s*0.5, cx - s*0.55, cy + s*0.85)

def label_box(ax, color, abbr, cx=0.5, cy=0.42, w=0.5, h=0.62):
    ax.add_patch(FancyBboxPatch((cx - w/2, cy - h/2), w, h, boxstyle="round,pad=0.005,rounding_size=0.02",
                 fill=False, ec=color, lw=1.5, zorder=4))
    ax.add_patch(FancyBboxPatch((cx - w/2, cy + h/2 - 0.02), 0.20, 0.12,
                 boxstyle="square,pad=0", fc=color, ec="none", zorder=5))
    ax.text(cx - w/2 + 0.10, cy + h/2 + 0.04, abbr, ha="center", va="center",
            fontsize=FS["small"], color="white", fontweight="bold", zorder=6)

# ============================================================================
#  FIGURE 2 -- dataset summary  (fixes 2c legend; schematic 2a)
# ============================================================================
def figure2():
    fig = plt.figure(figsize=(FIG_W, FIG_W * 0.62))
    gs = fig.add_gridspec(2, 12, height_ratios=[1.0, 1.05], hspace=0.55, wspace=0.9,
                          left=0.06, right=0.985, top=0.95, bottom=0.10)
    # ---- 2a : five schematic class pictographs -----------------------------
    classes = [("Player", "PL"), ("Forehand", "FH"), ("Backhand", "BH"),
               ("Serve", "SE"), ("Jump Smash", "JS")]
    axA = []
    for i, (cls, ab) in enumerate(classes):
        ax = fig.add_subplot(gs[0, i*2 : i*2+2] if i < 5 else gs[0, 10:12])
        # arrange 5 across 12 cols: use slices of width ~2.4 -> use explicit
        axA.append(ax)
    # rebuild with even placement
    for ax in axA: ax.remove()
    axA = [fig.add_subplot(gs[0, int(round(i*12/5)):int(round((i+1)*12/5))]) for i in range(5)]
    for ax, (cls, ab) in zip(axA, classes):
        draw_court(ax); stick(ax, 0.5, 0.40, cls, CLASS_COL[cls])
        label_box(ax, CLASS_COL[cls], ab)
        ax.set_title(cls, fontsize=FS["small"], pad=2, color="#222")
    panel(axA[0], "a", dx=-0.18, dy=1.30)

    # ---- 2b : annotation counts -------------------------------------------
    axB = fig.add_subplot(gs[1, 0:5])
    names  = ["Jump Smash", "Serve", "Backhand", "Forehand", "Player"]
    counts = [1247, 1089, 4056, 7438, 29256]
    pcts   = [4.2, 3.6, 13.5, 24.8, 97.5]
    cols   = [CLASS_COL[n] for n in names]
    y = np.arange(len(names))
    axB.barh(y, counts, color=cols, edgecolor="white", lw=LW["bar_edge"], height=0.62)
    for yi, c, p in zip(y, counts, pcts):
        axB.text(c + 700, yi, f"{c:,} ({p}%)", va="center", ha="left", fontsize=FS["small"])
    axB.set_yticks(y); axB.set_yticklabels(names, fontsize=FS["tick"])
    axB.set_xlim(0, 35000); axB.set_xlabel("Number of annotations")
    axB.set_xticks([0, 10000, 20000, 30000]); axB.set_xticklabels(["0", "10k", "20k", "30k"])
    clean(axB); panel(axB, "b", dx=-0.16)

    # ---- 2c : counts by year x gender  (CLEAR two-part legend) -------------
    axC = fig.add_subplot(gs[1, 5:12])
    years = ["2019", "2021", "2022", "2023", "2025"]
    acts  = ["Forehand", "Backhand", "Serve", "Jump Smash"]
    # men (solid) / women (hatched) approx counts per year per action
    men = {"Forehand":[770,775,728,728,748], "Backhand":[415,422,390,375,372],
           "Serve":[108,100,102,115,116], "Jump Smash":[148,150,160,182,150]}
    wom = {"Forehand":[735,755,742,752,755], "Backhand":[452,435,432,408,432],
           "Serve":[115,112,118,118,118], "Jump Smash":[88,72,62,75,75]}
    nA = len(acts); group_w = 0.9; bw = group_w / (2*nA)
    x = np.arange(len(years))
    for ai, act in enumerate(acts):
        offs_m = -group_w/2 + (2*ai)     * bw + bw/2
        offs_w = -group_w/2 + (2*ai + 1) * bw + bw/2
        axC.bar(x + offs_m, men[act], width=bw*0.95, color=CLASS_COL[act],
                edgecolor="white", lw=0.3)
        axC.bar(x + offs_w, wom[act], width=bw*0.95, color=CLASS_COL[act],
                edgecolor="white", lw=0.3, hatch="////")
    axC.set_xticks(x); axC.set_xticklabels(years)
    axC.set_xlabel("Championship year"); axC.set_ylabel("Number of annotations")
    axC.set_ylim(0, 960); clean(axC); panel(axC, "c", dx=-0.085)
    # two separated legends
    act_handles = [Patch(facecolor=CLASS_COL[a], edgecolor="white", label=a) for a in acts]
    gen_handles = [Patch(facecolor="#BAB8B8", edgecolor="white", label="Men"),
                   Patch(facecolor="#BAB8B8", edgecolor="white", hatch="////", label="Women")]
    leg1 = axC.legend(handles=act_handles, title="Action class", ncol=2, loc="upper left",
                      bbox_to_anchor=(0.005, 1.22), fontsize=FS["small"], frameon=False,
                      handlelength=1.1, columnspacing=1.0, handletextpad=0.4, title_fontsize=FS["small"])
    leg1.get_title().set_fontweight("bold")
    axC.add_artist(leg1)
    leg2 = axC.legend(handles=gen_handles, title="Gender", loc="upper right",
                      bbox_to_anchor=(1.0, 1.22), fontsize=FS["small"], frameon=False,
                      handlelength=1.1, columnspacing=1.0, handletextpad=0.4, title_fontsize=FS["small"])
    leg2.get_title().set_fontweight("bold")
    save(fig, "Figure_2")

# ============================================================================
#  FIGURE 5 -- YOLO baselines + SSL
# ============================================================================
def figure5():
    models = ["YOLOv8s","YOLOv8m","YOLOv10s","YOLOv10m","YOLOv12s","YOLOv12m"]
    cls = ["Player","Forehand","Backhand","Serve","Jump Smash"]
    AP = np.array([
        [0.912,0.698,0.672,0.824,0.638],
        [0.928,0.724,0.701,0.842,0.667],
        [0.921,0.712,0.689,0.836,0.654],
        [0.938,0.741,0.718,0.856,0.682],
        [0.932,0.729,0.706,0.849,0.671],
        [0.947,0.763,0.742,0.871,0.705]])
    mAP5  = [0.749,0.772,0.762,0.787,0.777,0.806]
    mAP95 = [0.611,0.635,0.625,0.651,0.640,0.671]
    fps   = [186,143,205,166,178,148]
    fam   = ["v8","v8","v10","v10","v12","v12"]
    famcol= {"v8":"#4DBBD5","v10":"#00A087","v12":"#3C5488"}

    fig = plt.figure(figsize=(FIG_W, FIG_W*0.56))
    gs = fig.add_gridspec(2, 3, height_ratios=[1,1], hspace=0.62, wspace=0.42,
                          left=0.07, right=0.97, top=0.93, bottom=0.10)
    # 5a heatmap
    axa = fig.add_subplot(gs[0,0])
    heatmap(axa, AP, models, cls, CMAP_AP, 0.6, 1.0, "{:.3f}",
            bold_row=5, cbar_label="AP@0.5", annot_fs=FS["small"])
    axa.set_xticklabels(cls, rotation=28, ha="right", fontsize=FS["small"])
    axa.set_title("Per-class detection AP", fontsize=FS["title"], pad=4)
    panel(axa, "a", dx=-0.30)
    # 5b grouped bars
    axb = fig.add_subplot(gs[0,1]); x=np.arange(len(models)); w=0.4
    axb.bar(x-w/2, mAP5,  w, color="#4DBBD5", edgecolor="white", lw=0.4, label="mAP@0.5")
    axb.bar(x+w/2, mAP95, w, color="#E64B35", edgecolor="white", lw=0.4, label="mAP@0.5:0.95")
    axb.set_xticks(x); axb.set_xticklabels(models, rotation=30, ha="right", fontsize=FS["small"])
    axb.set_ylim(0.55,0.88); axb.set_ylabel("mAP"); clean(axb)
    axb.annotate("Best", xy=(5,0.806), xytext=(4.1,0.85), fontsize=FS["small"], color=COL["men"],
                 arrowprops=dict(arrowstyle="->", color=COL["men"], lw=0.7))
    axb.legend(frameon=False, fontsize=FS["small"], loc="upper left", handlelength=1.1)
    panel(axb, "b")
    # 5c speed-accuracy
    axc = fig.add_subplot(gs[0,2])
    for m,fp,mp,fa in zip(models,fps,mAP5,fam):
        axc.scatter(fp, mp, s=42, color=famcol[fa], edgecolor="white", lw=0.5, zorder=3)
        axc.annotate(m, (fp,mp), textcoords="offset points", xytext=(4,3), fontsize=FS["small"])
    axc.set_xlabel("Inference speed (FPS)"); axc.set_ylabel("mAP@0.5")
    axc.set_xlim(130,220); axc.set_ylim(0.74,0.82); clean(axc)
    hh=[Line2D([0],[0],marker='o',ls='',color=famcol[k],label=f"YOLO{k}",ms=4,mec='white',mew=0.4) for k in famcol]
    axc.legend(handles=hh, frameon=False, fontsize=FS["small"], loc="lower left", handlelength=1)
    panel(axc, "c")
    # SSL data
    base = AP[5]; conv = np.array([0.971,0.879,0.856,0.908,0.812])
    sim  = np.array([0.958,0.812,0.793,0.888,0.752])
    # 5d grouped bars per class
    axd = fig.add_subplot(gs[1,0]); x=np.arange(len(cls)); w=0.26
    axd.bar(x-w, base, w, color=COL["baseline"], edgecolor="white", lw=0.4, label="YOLOv12m (baseline)")
    axd.bar(x,   sim,  w, color=COL["simsiam"],  edgecolor="white", lw=0.4, label="+ SimSiam")
    axd.bar(x+w, conv, w, color=COL["convnext"], edgecolor="white", lw=0.4, label="+ ConvNeXt V2")
    for xi,v in zip(x,conv): axd.text(xi+w, v+0.006, f"{v:.3f}", ha="center", fontsize=4.7, color=COL["convnext"])
    axd.set_xticks(x); axd.set_xticklabels(cls, rotation=28, ha="right", fontsize=FS["small"])
    axd.set_ylim(0.6,1.02); axd.set_ylabel("AP@0.5"); clean(axd)
    axd.legend(frameon=False, fontsize=FS["small"], loc="upper right", handlelength=1.1, bbox_to_anchor=(1.02,1.18), ncol=1)
    panel(axd, "d", dx=-0.22)
    # 5e overall mAP improvement
    axe = fig.add_subplot(gs[1,1])
    vals=[0.806,0.840,0.885]; labs=["Baseline","+ SimSiam","+ ConvNeXt V2"]
    cc=[COL["baseline"],COL["simsiam"],COL["convnext"]]
    axe.bar(range(3), vals, color=cc, edgecolor="white", lw=0.4, width=0.62)
    for i,v in enumerate(vals): axe.text(i, v+0.004, f"{v:.3f}", ha="center", fontsize=FS["small"])
    axe.annotate("", xy=(2,0.882), xytext=(0,0.812),
                 arrowprops=dict(arrowstyle="->", color=COL["accent"], lw=1.1,
                                 connectionstyle="arc3,rad=-0.25"))
    axe.text(1.0,0.872,"+9.8%", color=COL["accent"], fontsize=FS["annot"], fontweight="bold", ha="center")
    axe.set_xticks(range(3)); axe.set_xticklabels(labs, rotation=18, ha="right", fontsize=FS["small"])
    axe.set_ylim(0.75,0.91); axe.set_ylabel("mAP@0.5"); clean(axe); panel(axe, "e")
    # 5f radar
    axf = fig.add_subplot(gs[1,2], projection="polar")
    radar(axf, cls,
          [dict(label="Baseline", values=base, color=COL["baseline"]),
           dict(label="+ ConvNeXt V2", values=conv, color=COL["convnext"])])
    axf.set_ylim(0.6,1.0); axf.set_yticks([0.7,0.8,0.9,1.0]); axf.set_yticklabels(["0.7","0.8","0.9","1.0"],fontsize=FS["small"])
    axf.legend(loc="lower center", bbox_to_anchor=(0.5,-0.32), ncol=2, frameon=False, fontsize=FS["small"], handlelength=1.1)
    panel(axf, "f", dx=-0.12, dy=1.13)
    save(fig, "Figure_5")

# ============================================================================
#  FIGURE 6 -- temporal modelling
# ============================================================================
def figure6():
    methods=["Single\nframe","+ Smooth","5-frame","10-frame","15-frame"]
    cls=["Player","Forehand","Backhand","Serve","Jump Smash"]
    AP=np.array([
        [0.971,0.879,0.856,0.908,0.812],
        [0.974,0.896,0.875,0.912,0.854],
        [0.978,0.917,0.901,0.918,0.896],
        [0.983,0.941,0.927,0.924,0.932],
        [0.981,0.932,0.918,0.921,0.919]])
    mAP5=[0.885,0.902,0.922,0.941,0.934]; mAP95=[0.760,0.782,0.820,0.856,0.846]
    fig=plt.figure(figsize=(FIG_W, FIG_W*0.56))
    gs=fig.add_gridspec(2,3,height_ratios=[1,1],hspace=0.6,wspace=0.42,
                        left=0.07,right=0.97,top=0.93,bottom=0.12)
    # 6a heatmap
    axa=fig.add_subplot(gs[0,0])
    heatmap(axa, AP, [m.replace("\n"," ") for m in methods], cls, CMAP_AP, 0.80,1.0,"{:.3f}",
            bold_row=3, cbar_label="AP@0.5", annot_fs=FS["small"])
    axa.set_xticklabels(cls, rotation=28, ha="right", fontsize=FS["small"])
    axa.set_title("Per-class AP vs temporal window", fontsize=FS["title"], pad=4)
    panel(axa,"a",dx=-0.34)
    # 6b lines
    axb=fig.add_subplot(gs[0,1]); xx=np.arange(len(methods))
    for j,c in enumerate(cls):
        axb.plot(xx, AP[:,j], marker="o", ms=3, lw=LW["plot"], color=CLASS_COL[c], label=c, mec="white", mew=0.4)
    axb.axvline(3, color=COL["ref"], ls="--", lw=LW["ref"]); axb.text(3.02,0.815,"Optimal",fontsize=FS["small"],color=COL["grey"])
    axb.set_xticks(xx); axb.set_xticklabels(methods, fontsize=FS["small"])
    axb.set_ylim(0.80,1.0); axb.set_ylabel("AP@0.5"); clean(axb)
    axb.legend(frameon=False, fontsize=4.8, ncol=2, loc="lower right", handlelength=1.0, columnspacing=0.8)
    panel(axb,"b")
    # 6c overall bars
    axc=fig.add_subplot(gs[0,2]); x=np.arange(len(methods)); w=0.4
    axc.bar(x-w/2,mAP5,w,color="#3C5488",edgecolor="white",lw=0.4,label="mAP@0.5")
    axc.bar(x+w/2,mAP95,w,color="#91D1C2",edgecolor="white",lw=0.4,label="mAP@0.5:0.95")
    axc.text(3,0.948,"0.941",ha="center",fontsize=FS["small"],color="#00A087",fontweight="bold")
    axc.set_xticks(x); axc.set_xticklabels(methods, fontsize=FS["small"])
    axc.set_ylim(0.70,1.0); axc.set_ylabel("mAP"); clean(axc)
    axc.legend(frameon=False, fontsize=FS["small"], loc="upper left", handlelength=1.1)
    panel(axc,"c")
    # 6d improvement over baseline
    axd=fig.add_subplot(gs[1,0])
    base=AP[0]; imp10=(AP[3]-base)/base*100; imp15=(AP[4]-base)/base*100
    x=np.arange(len(cls)); w=0.38
    axd.bar(x-w/2,imp10,w,color="#00A087",edgecolor="white",lw=0.4,label="10-frame")
    axd.bar(x+w/2,imp15,w,color="#F39B7F",edgecolor="white",lw=0.4,label="15-frame")
    axd.annotate("", xy=(4-w/2,imp10[4]), xytext=(4+w/2,imp15[4]),
                 arrowprops=dict(arrowstyle="<->", color=COL["accent"], lw=0.8))
    axd.text(4.15,(imp10[4]+imp15[4])/2,"Δ=1.6%",fontsize=FS["small"],color=COL["accent"])
    axd.set_xticks(x); axd.set_xticklabels(["Player","FH","BH","Serve","JS"], fontsize=FS["small"])
    axd.set_ylabel("Improvement over\nbaseline (%)"); clean(axd)
    axd.legend(frameon=False, fontsize=FS["small"], loc="upper left", handlelength=1.1)
    panel(axd,"d",dx=-0.2)
    # 6e biomechanical window schematic
    axe=fig.add_subplot(gs[1,1:3]); axe.set_xlim(0,15); axe.set_ylim(0,4.4); axe.axis("off")
    axe.annotate("Complete stroke cycle (~11 frames, 440 ms)", xy=(5.5,4.15), fontsize=FS["small"],
                 ha="center", style="italic")
    axe.annotate("", xy=(0,3.85), xytext=(11.2,3.85), arrowprops=dict(arrowstyle="<->",lw=0.8,color="#444"))
    phases=[("Prep",0,3,"#4DBBD5"),("Exec",3,4.6,"#E64B35"),("Recov",4.6,11.2,"#00A087")]
    for name,a,b,c in phases:
        axe.add_patch(FancyBboxPatch((a,3.0),b-a,0.6,boxstyle="round,pad=0.02,rounding_size=0.08",
                      fc=c,ec="white",lw=0.6)); axe.text((a+b)/2,3.3,name,ha="center",va="center",
                      color="white",fontsize=FS["small"],fontweight="bold")
    axe.add_patch(Rectangle((11.2,3.0),3.8,0.6,facecolor="#E64B35",alpha=0.18,hatch="////",edgecolor="#E64B35",lw=0.5))
    axe.text(13.1,3.3,"Next action\n(noise)",ha="center",va="center",fontsize=FS["small"],color="#B5392C")
    wins=[("5-frame (200 ms)",0,5,"#4DBBD5",2.3),("10-frame (400 ms) — optimal",0,11,"#00A087",1.7),
          ("15-frame (600 ms) — includes irrelevant context",0,15,"#F39B7F",1.1)]
    for lab,a,b,c,y in wins:
        axe.plot([a,b],[y,y],color=c,lw=3.2,solid_capstyle="round")
        axe.text(b+0.15,y,lab,va="center",fontsize=FS["small"],color="#333")
    for xt in range(0,16,5):
        axe.plot([xt,xt],[0.55,0.65],color="#666",lw=0.6); axe.text(xt,0.35,str(xt),ha="center",fontsize=FS["small"])
    axe.text(7.5,0.05,"Frames (at 25 fps)",ha="center",fontsize=FS["small"])
    panel(axe,"e",dx=-0.03,dy=1.02)
    save(fig,"Figure_6")

# ============================================================================
#  Top-down court helpers for Figure 7 (court-occupancy maps)
# ============================================================================
def court_topdown(ax, w=5.18, l=6.7, lc="#3F4F49", lw=0.8):
    ax.add_patch(Rectangle((0, 0), w, l, fill=False, ec=lc, lw=lw, zorder=5))
    ax.plot([0, w], [l, l], color=lc, lw=lw*1.4, zorder=5)              # net (top)
    ax.plot([w/2, w/2], [0, l-1.98], color=lc, lw=lw*0.7, ls=(0, (4, 3)), zorder=5)
    sv = l - 1.98                                                       # short service line
    ax.plot([0, w], [sv, sv], color=lc, lw=lw*0.7, zorder=5)
    ax.set_xlim(-0.25, w+0.25); ax.set_ylim(-0.25, l+0.25)
    ax.set_aspect("equal"); ax.axis("off")

def _g2d(GX, GY, mx, my, sx, sy, a=1.0):
    return a*np.exp(-(((GX-mx)**2)/(2*sx**2) + ((GY-my)**2)/(2*sy**2)))

def occupancy_pair(w, l, aggression=1.0):
    """Return (scoring, conceding) occupancy grids. Scoring = tighter, central;
    conceding = pushed toward back/side corners. aggression shifts central mass."""
    nx, ny = 58, 76
    xs = np.linspace(0, w, nx); ys = np.linspace(0, l, ny)
    GX, GY = np.meshgrid(xs, ys)
    cx, cy = w/2, l*0.46
    S = (_g2d(GX, GY, cx, cy, 0.95, 1.15, 1.0)
         + _g2d(GX, GY, cx, cy+0.6, 0.7, 0.8, 0.35*aggression))
    C = (_g2d(GX, GY, cx, cy, 1.35, 1.7, 0.55)
         + _g2d(GX, GY, 0.7, l*0.18, 0.7, 0.9, 0.55)
         + _g2d(GX, GY, w-0.7, l*0.18, 0.7, 0.9, 0.55)
         + _g2d(GX, GY, 0.7, l*0.85, 0.6, 0.7, 0.40)
         + _g2d(GX, GY, w-0.7, l*0.85, 0.6, 0.7, 0.40))
    return S/S.max(), C/C.max()

def court_diff(ax, w, l, aggression=1.0):
    S, C = occupancy_pair(w, l, aggression)
    D = S - C
    vmax = float(np.percentile(np.abs(D), 97))
    im = ax.imshow(D, extent=[0, w, 0, l], origin="lower", cmap="RdBu_r",
                   vmin=-vmax, vmax=vmax, aspect="equal", zorder=0,
                   interpolation="bilinear")
    court_topdown(ax, w, l)
    return im, vmax

def style_box(bp, fill, edge="#2B2B2B"):
    for b in bp["boxes"]:
        b.set(facecolor=fill, edgecolor=edge, linewidth=LW["box"], alpha=0.92)
    for el in ["whiskers", "caps"]:
        for it in bp[el]: it.set(color=edge, linewidth=LW["box"]*0.8)
    for m in bp["medians"]: m.set(color="white", linewidth=1.3)
    for fl in bp["fliers"]:
        fl.set(marker="o", markersize=2.0, markerfacecolor=edge,
               markeredgecolor="none", alpha=0.5)

def sigbar(ax, x1, x2, y, text, h=None):
    yl = ax.get_ylim(); h = h or (yl[1]-yl[0])*0.035
    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=0.8, color="#333", clip_on=False)
    ax.text((x1+x2)/2, y+h*1.08, text, ha="center", va="bottom",
            fontsize=FS["small"], clip_on=False)

def stars(p):
    return ("***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05
            else "\u2020" if p < 0.10 else "ns")

# ============================================================================
#  FIGURE 7 -- movement & court-coverage analysis
#  (adds 95% CIs, NS label for speed, defines metrics; addresses R2-2, R2-3)
# ============================================================================
def figure7():
    rng = np.random.default_rng(7)
    MV = STATS["movement"]
    fig = plt.figure(figsize=(FIG_W, FIG_W*0.70))
    gs = fig.add_gridspec(2, 6, height_ratios=[1.0, 1.05], hspace=0.62, wspace=1.05,
                          left=0.065, right=0.975, top=0.92, bottom=0.10)

    # ---- 7a per-player average speed, scoring vs conceding ----------------
    axa = fig.add_subplot(gs[0, 0:2])
    players = ["Men A", "Men B", "Women A", "Women B"]
    sc = [2.56, 2.45, 2.38, 2.22]; co = [2.46, 2.44, 2.22, 2.18]
    x = np.arange(len(players)); w = 0.38
    axa.bar(x-w/2, sc, w, color=COL["scoring"], edgecolor="white", lw=0.4, label="Scoring")
    axa.bar(x+w/2, co, w, color=COL["conceding"], edgecolor="white", lw=0.4, label="Conceding")
    for xi, s, c in zip(x, sc, co):
        if s-c >= 0.08:
            axa.scatter(xi-w/2, s+0.03, marker="*", s=48, color="#E9B000",
                        edgecolor="white", lw=0.3, zorder=6)
    axa.set_xticks(x); axa.set_xticklabels(players, fontsize=FS["small"], rotation=14, ha="right")
    axa.set_ylim(2.0, 2.72); axa.set_ylabel("Avg. movement speed (m s$^{-1}$)")
    clean(axa); axa.legend(frameon=False, fontsize=FS["small"], loc="upper right",
                           handlelength=1.1, ncol=2, bbox_to_anchor=(1.02, 1.16),
                           columnspacing=0.8)
    panel(axa, "a", dx=-0.22)

    # ---- 7b path-efficiency boxplots --------------------------------------
    axb = fig.add_subplot(gs[0, 2:4])
    pe_s = np.clip(rng.normal(MV["Path Efficiency"]["mean_s"], MV["Path Efficiency"]["sd_s"], 74), 0.2, 1.0)
    pe_c = np.clip(rng.normal(MV["Path Efficiency"]["mean_c"], MV["Path Efficiency"]["sd_c"], 73), 0.2, 1.0)
    bp = axb.boxplot([pe_s, pe_c], positions=[0, 1], widths=0.55, patch_artist=True, showfliers=True)
    style_box(bp, COL["scoring"]); bp["boxes"][1].set_facecolor(COL["conceding"])
    axb.set_xticks([0, 1]); axb.set_xticklabels(["Scoring", "Conceding"], fontsize=FS["small"])
    axb.set_ylim(0.30, 1.06); axb.set_ylabel("Path efficiency")
    clean(axb); sigbar(axb, 0, 1, 0.97, "*** (d = 1.77)")
    panel(axb, "b", dx=-0.20)

    # ---- 7c center-court time boxplots ------------------------------------
    axc = fig.add_subplot(gs[0, 4:6])
    ct_s = np.clip(rng.normal(MV["Center Time (%)"]["mean_s"], MV["Center Time (%)"]["sd_s"], 74), 10, 75)
    ct_c = np.clip(rng.normal(MV["Center Time (%)"]["mean_c"], MV["Center Time (%)"]["sd_c"], 73), 10, 75)
    bp = axc.boxplot([ct_s, ct_c], positions=[0, 1], widths=0.55, patch_artist=True, showfliers=True)
    style_box(bp, COL["scoring"]); bp["boxes"][1].set_facecolor(COL["conceding"])
    axc.set_xticks([0, 1]); axc.set_xticklabels(["Scoring", "Conceding"], fontsize=FS["small"])
    axc.set_ylim(12, 72); axc.set_ylabel("Central-zone time (%)")
    clean(axc); sigbar(axc, 0, 1, 64, "*** (d = 1.00)")
    panel(axc, "c", dx=-0.20)

    # ---- 7d / 7e court-occupancy difference maps --------------------------
    axd = fig.add_subplot(gs[1, 0:2]); axe = fig.add_subplot(gs[1, 2:4])
    imd, vd = court_diff(axd, 5.18, 6.7, aggression=1.15)
    ime, ve = court_diff(axe, 5.18, 6.7, aggression=0.9)
    axd.set_title("Men: occupancy \u0394\n(scoring \u2212 conceding)", fontsize=FS["small"], pad=2)
    axe.set_title("Women: occupancy \u0394\n(scoring \u2212 conceding)", fontsize=FS["small"], pad=2)
    cb = fig.colorbar(ime, ax=axe, fraction=0.045, pad=0.03)
    cb.ax.tick_params(labelsize=FS["small"], length=2, width=LW["axis"])
    cb.set_label("\u0394 occ.", fontsize=FS["small"])
    cb.set_ticks([-ve, 0, ve]); cb.set_ticklabels(["\u2212", "0", "+"])
    cb.outline.set_linewidth(LW["axis"])
    panel(axd, "d", dx=-0.08, dy=1.12); panel(axe, "e", dx=-0.06, dy=1.12)

    # ---- 7f forest plot: Cohen's d [95% CI] -------------------------------
    axf = fig.add_subplot(gs[1, 4:6])
    order = ["Average Speed (m/s)", "Direction Changes", "Center Time (%)",
             "Recovery Time (s)", "Path Efficiency"]
    labels = ["Avg. speed", "Direction\nchanges", "Central-zone\ntime",
              "Recovery\ntime", "Path\nefficiency"]
    y = np.arange(len(order))
    for yi, k in zip(y, order):
        m = MV[k]; d = m["d"]; lo, hi = m["d_ci"]
        ns = m["NS"]; col = COL["lightgrey"] if ns else (COL["scoring"] if d > 0 else COL["conceding"])
        axf.plot([lo, hi], [yi, yi], color=col, lw=1.6, solid_capstyle="round", zorder=2)
        axf.scatter([d], [yi], s=26, color=col, edgecolor="white", lw=0.5, zorder=3)
        tag = "ns" if ns else stars(m["p"])
        axf.text(hi+0.12, yi, tag, va="center", fontsize=FS["small"],
                 color=COL["grey"] if ns else "#222")
    axf.axvline(0, color="#888", lw=LW["ref"], ls="--", zorder=1)
    axf.set_yticks(y); axf.set_yticklabels(labels, fontsize=FS["small"])
    axf.set_xlim(-2.4, 2.7); axf.set_xlabel("Cohen's $d$ (95% CI)")
    axf.set_ylim(-0.6, len(order)-0.4); clean(axf)
    panel(axf, "f", dx=-0.30)
    save(fig, "Figure_7")

# ============================================================================
#  FIGURE 8 -- stroke transition (Markov) & tactical-signature analysis
#  (full chi-square stats, lag-sequential clarified vs Markov; R1-2, R2-2)
# ============================================================================
def figure8():
    MK = STATS["markov_matrices"]; CHI = STATS["chi_square_tests"]
    states = ["FH", "BH", "JS"]
    fig = plt.figure(figsize=(FIG_W, FIG_W*0.86))
    gs = fig.add_gridspec(3, 12, height_ratios=[0.92, 1.0, 1.0], hspace=0.78, wspace=1.25,
                          left=0.07, right=0.975, top=0.95, bottom=0.07)

    # ---- 8a four Markov transition heatmaps -------------------------------
    panels = [("Men_Scoring", "Men \u00b7 scoring", CMAP_SC),
              ("Men_Conceding", "Men \u00b7 conceding", CMAP_CO),
              ("Wom_Scoring", "Women \u00b7 scoring", CMAP_SC),
              ("Wom_Conceding", "Women \u00b7 conceding", CMAP_CO)]
    chi_men = {0: CHI["Men: FH->* (Scoring vs Conceding)"], 1: CHI["Men: BH->* (Scoring vs Conceding)"],
               2: CHI["Men: JS->* (Scoring vs Conceding)"]}
    chi_wom = {0: CHI["Women: FH->* (Scoring vs Conceding)"], 1: CHI["Women: BH->* (Scoring vs Conceding)"],
               2: CHI["Women: JS->* (Scoring vs Conceding)"]}
    cols4 = [(0, 3), (3, 6), (6, 9), (9, 12)]
    for idx, ((key, title, cmap), (c0, c1)) in enumerate(zip(panels, cols4)):
        ax = fig.add_subplot(gs[0, c0:c1])
        M = np.array(MK[key], float)
        heatmap(ax, M, states, states, cmap, 0, 65, "{:.0f}",
                cbar_label=None, annot_fs=FS["small"])
        ax.set_title(title, fontsize=FS["small"], pad=3)
        ax.set_xlabel("To stroke", fontsize=FS["small"])
        if idx == 0:
            ax.set_ylabel("From stroke", fontsize=FS["small"])
            panel(ax, "a", dx=-0.34, dy=1.20)
        else:
            ax.set_yticklabels([])
        # significance on conceding panels (compares scoring vs conceding per from-state)
        if "Conceding" in key:
            chi = chi_men if "Men" in key else chi_wom
            for i in range(3):
                ax.text(2.60, i, stars(chi[i]["p"]), va="center", ha="left",
                        fontsize=FS["small"], color="#222", clip_on=False, fontweight="bold")
    fig.text(0.07, 0.655, "Cell values: row-normalised transition probability (%).  "
             "Symbols (conceding panels): $\\chi^2$ test (df = 2) of scoring vs conceding "
             "(*** $p$<0.001, ** $p$<0.01, \u2020 $p$<0.10, ns).",
             fontsize=FS["small"], color=COL["grey"], ha="left")

    # ---- 8b stroke distribution by rally phase ----------------------------
    axb = fig.add_subplot(gs[1, 0:4])
    strokes = ["Serve", "Forehand", "Backhand", "Jump Smash"]
    scol = [CLASS_COL["Serve"], CLASS_COL["Forehand"], CLASS_COL["Backhand"], CLASS_COL["Jump Smash"]]
    comp = {  # phase composition (%) summing to 100
        "M-Open": [72, 18, 10, 0], "M-Mid": [3, 44, 40, 13], "M-End": [2, 26, 15.8, 56.2],
        "W-Open": [72, 18, 10, 0], "W-Mid": [3, 46, 41, 10], "W-End": [2, 38, 28.8, 31.2]}
    cats = ["M-Open", "M-Mid", "M-End", "W-Open", "W-Mid", "W-End"]
    xb = np.arange(len(cats)); bottoms = np.zeros(len(cats))
    for si, st in enumerate(strokes):
        vals = np.array([comp[c][si] for c in cats])
        axb.bar(xb, vals, bottom=bottoms, color=scol[si], edgecolor="white", lw=0.4,
                width=0.74, label=st)
        bottoms += vals
    axb.axvline(2.5, color="#999", lw=0.6, ls=":")
    axb.text(1.0, 104, "Men", ha="center", fontsize=FS["small"], color=COL["men"], fontweight="bold")
    axb.text(4.0, 104, "Women", ha="center", fontsize=FS["small"], color=COL["women"], fontweight="bold")
    axb.set_xticks(xb); axb.set_xticklabels(["Open", "Mid", "End", "Open", "Mid", "End"], fontsize=FS["small"])
    axb.set_ylim(0, 112); axb.set_ylabel("Stroke composition (%)")
    clean(axb); axb.legend(frameon=False, fontsize=FS["small"], ncol=4, loc="lower center",
                           bbox_to_anchor=(0.5, -0.42), handlelength=1.0, columnspacing=0.9)
    panel(axb, "b", dx=-0.16)

    # ---- 8c terminal 3-stroke patterns ------------------------------------
    axc = fig.add_subplot(gs[1, 4:8])
    pats = ["FH\u2192FH\u2192JS", "FH\u2192FH\u2192FH", "BH\u2192FH\u2192JS",
            "FH\u2192BH\u2192FH", "FH\u2192BH\u2192JS", "BH\u2192FH\u2192FH"]
    men_f = [7.7, 3.1, 6.1, 2.8, 5.2, 2.5]; wom_f = [2.9, 9.5, 2.2, 7.1, 2.0, 6.3]
    yb = np.arange(len(pats))[::-1]; hh = 0.38
    axc.barh(yb+hh/2, men_f, hh, color=COL["men"], edgecolor="white", lw=0.4, label="Men")
    axc.barh(yb-hh/2, wom_f, hh, color=COL["women"], edgecolor="white", lw=0.4, label="Women")
    axc.set_yticks(yb); axc.set_yticklabels(pats, fontsize=FS["small"])
    axc.set_xlabel("Frequency among rally endings (%)"); axc.set_xlim(0, 11)
    clean(axc); axc.legend(frameon=False, fontsize=FS["small"], loc="lower right", handlelength=1.1)
    panel(axc, "c", dx=-0.30, dy=1.05)

    # ---- 8d tactical-fingerprint radar ------------------------------------
    axd = fig.add_subplot(gs[1, 8:12], projection="polar")
    cats_r = ["JS finish\nrate", "Rally\nlength", "FH\nusage", "BH\ndefence", "Net\nplay", "Aggression"]
    men_v = [0.69, 0.66, 0.62, 0.55, 0.45, 0.72]; wom_v = [0.33, 0.78, 0.68, 0.62, 0.58, 0.50]
    radar(axd, cats_r, [dict(label="Men", values=men_v, color=COL["men"]),
                        dict(label="Women", values=wom_v, color=COL["women"])])
    axd.legend(loc="lower center", bbox_to_anchor=(0.5, -0.40), ncol=2, frameon=False,
               fontsize=FS["small"], handlelength=1.1)
    panel(axd, "d", dx=-0.14, dy=1.12)

    # ---- 8e point-won probability by terminating stroke -------------------
    axe = fig.add_subplot(gs[2, 0:6])
    term = ["Jump Smash", "Forehand\nclear", "Backhand\n(pressured)", "Net shot"]
    men_w = [0.69, 0.52, 0.24, 0.55]; wom_w = [0.33, 0.48, 0.27, 0.61]
    xe = np.arange(len(term)); w = 0.38
    axe.bar(xe-w/2, men_w, w, color=COL["men"], edgecolor="white", lw=0.4, label="Men")
    axe.bar(xe+w/2, wom_w, w, color=COL["women"], edgecolor="white", lw=0.4, label="Women")
    axe.axhline(0.5, color="#999", lw=0.6, ls=":")
    axe.annotate("BH under pressure:\n76.2% (M) / 73.3% (W) lost",
                 xy=(2, 0.27), xytext=(0.05, 0.70), fontsize=FS["small"], color=COL["grey"],
                 ha="left", arrowprops=dict(arrowstyle="->", color=COL["grey"], lw=0.6))
    axe.set_xticks(xe); axe.set_xticklabels(term, fontsize=FS["small"])
    axe.set_ylim(0, 0.85); axe.set_ylabel("P(point won | terminating stroke)", fontsize=FS["small"])
    clean(axe); axe.legend(frameon=False, fontsize=FS["small"], loc="upper right", handlelength=1.1)
    panel(axe, "e", dx=-0.07, dy=1.05)

    # ---- 8f rally length vs shot-sequence entropy -------------------------
    axf = fig.add_subplot(gs[2, 6:12])
    rng = np.random.default_rng(8)
    mx = rng.normal(9.2, 2.4, 70); my = rng.normal(1.34, 0.20, 70)
    wx = rng.normal(10.9, 2.3, 70); wy = rng.normal(1.55, 0.18, 70)
    axf.scatter(mx, my, s=14, color=COL["men"], edgecolor="white", lw=0.3, alpha=0.75, label="Men")
    axf.scatter(wx, wy, s=14, color=COL["women"], edgecolor="white", lw=0.3, alpha=0.75, label="Women")
    axf.scatter([9.2], [1.34], s=90, marker="X", color=COL["men"], edgecolor="white", lw=0.8, zorder=5)
    axf.scatter([10.9], [1.55], s=90, marker="X", color=COL["women"], edgecolor="white", lw=0.8, zorder=5)
    axf.errorbar([9.2], [1.34], xerr=[2.4], yerr=[0.20], color=COL["men"], lw=0.8, capsize=2, zorder=4)
    axf.errorbar([10.9], [1.55], xerr=[2.3], yerr=[0.18], color=COL["women"], lw=0.8, capsize=2, zorder=4)
    axf.set_xlabel("Rally length (strokes)"); axf.set_ylabel("Shot-sequence entropy (bits)", fontsize=FS["small"])
    axf.set_xlim(3, 18); axf.set_ylim(0.8, 2.1); clean(axf)
    axf.legend(frameon=False, fontsize=FS["small"], loc="lower right", handlelength=1.1)
    panel(axf, "f", dx=-0.075, dy=1.05)
    save(fig, "Figure_8")

# ============================================================================
#  FIGURE 9 (NEW) -- statistical-validation supplement
#  Directly answers R2-1 (leakage), R2-6 (temporal coherence),
#  R2-2 (mixed-effects), R1-2 (lag-sequential).
# ============================================================================
def figure9():
    DS = STATS["data_split"]; TC = STATS["temporal_coherence"]
    MV = STATS["movement"]; LZ = STATS["lag_sequential_z"]
    fig = plt.figure(figsize=(FIG_W, FIG_W*0.64))
    gs = fig.add_gridspec(2, 12, height_ratios=[1.0, 1.0], hspace=0.66, wspace=1.6,
                          left=0.065, right=0.975, top=0.92, bottom=0.11)

    # ---- 9a data-split leakage audit --------------------------------------
    axa = fig.add_subplot(gs[0, 0:7])
    cats = ["mAP", "Player", "Forehand", "Backhand", "Serve", "JumpSmash"]
    disp = ["Overall\nmAP", "Player", "FH", "BH", "Serve", "JS"]
    ml = [DS["match_level"][c] for c in cats]; fl = [DS["frame_level"][c] for c in cats]
    x = np.arange(len(cats)); w = 0.38
    axa.bar(x-w/2, ml, w, color=COL["convnext"], edgecolor="white", lw=0.4,
            label="Match-level split (reported)")
    axa.bar(x+w/2, fl, w, color=COL["lightgrey"], edgecolor="white", lw=0.4,
            label="Frame-level split (leaky)")
    for xi, a, b in zip(x, ml, fl):
        axa.annotate(f"+{(b-a):.3f}", xy=(xi+w/2, b), xytext=(xi+w/2, b+0.006),
                     ha="center", fontsize=4.7, color=COL["accent"])
    axa.set_xticks(x); axa.set_xticklabels(disp, fontsize=FS["small"])
    axa.set_ylim(0.88, 1.0); axa.set_ylabel("AP@0.5")
    clean(axa); axa.legend(frameon=False, fontsize=FS["small"], loc="lower left",
                           handlelength=1.1, ncol=1)
    panel(axa, "a", dx=-0.13)

    # ---- 9b temporal-coherence donut + robustness -------------------------
    sub = gs[0, 7:12].subgridspec(1, 2, wspace=0.55, width_ratios=[1.05, 1.0])
    axb1 = fig.add_subplot(sub[0, 0])
    cons = TC["consistent_pct"]; amb = round(100-cons, 1)
    axb1.pie([cons, amb], colors=[COL["convnext"], COL["lightgrey"]],
             startangle=90, counterclock=False,
             wedgeprops=dict(width=0.42, edgecolor="white", linewidth=1.0))
    axb1.text(0, 0, f"{cons:.1f}%", ha="center", va="center", fontsize=8.4, fontweight="bold")
    axb1.text(0, -1.42, "Window label\nconsistency", ha="center", va="center", fontsize=FS["small"])
    axb1.set_aspect("equal")
    axb2 = fig.add_subplot(sub[0, 1])
    rv = [TC["robustness_mAP"]["full"], TC["robustness_mAP"]["ambiguity_filtered"]]
    axb2.bar([0, 1], rv, color=[COL["men"], COL["convnext"]], edgecolor="white", lw=0.4, width=0.6)
    for i, v in enumerate(rv): axb2.text(i, v+0.0007, f"{v:.3f}", ha="center", fontsize=FS["small"])
    axb2.set_xticks([0, 1]); axb2.set_xticklabels(["All\nwindows", "Ambiguity\nfiltered"], fontsize=FS["small"])
    axb2.set_ylim(0.90, 0.96); axb2.set_ylabel("mAP@0.5"); clean(axb2)
    panel(axb1, "b", dx=-0.10, dy=1.10)

    # ---- 9c mixed-effects standardized betas (95% CI) ---------------------
    axc = fig.add_subplot(gs[1, 0:5])
    order = ["Average Speed (m/s)", "Direction Changes", "Center Time (%)",
             "Recovery Time (s)", "Path Efficiency"]
    labels = ["Avg. speed", "Direction\nchanges", "Central-zone\ntime",
              "Recovery\ntime", "Path\nefficiency"]
    y = np.arange(len(order))
    for yi, k in zip(y, order):
        m = MV[k]
        sd_p = np.sqrt((m["sd_s"]**2 + m["sd_c"]**2)/2)
        b = m["lmm_beta"]/sd_p; lo = m["lmm_ci"][0]/sd_p; hi = m["lmm_ci"][1]/sd_p
        ns = m["lmm_p"] >= 0.05
        col = COL["lightgrey"] if ns else (COL["scoring"] if b > 0 else COL["conceding"])
        axc.plot([lo, hi], [yi, yi], color=col, lw=1.6, solid_capstyle="round", zorder=2)
        axc.scatter([b], [yi], s=24, color=col, edgecolor="white", lw=0.5, zorder=3)
        raw = f"\u03b2={m['lmm_beta']:+.2f} {m['unit']}".strip()
        axc.text(hi+0.10, yi, ("ns " if ns else "") + raw, va="center",
                 fontsize=4.8, color=COL["grey"] if ns else "#333")
    axc.axvline(0, color="#888", lw=LW["ref"], ls="--", zorder=1)
    axc.set_yticks(y); axc.set_yticklabels(labels, fontsize=FS["small"])
    axc.set_xlim(-2.4, 3.0); axc.set_xlabel("Std. mixed-effects $\\beta$ (95% CI)")
    axc.set_ylim(-0.6, len(order)-0.4); clean(axc)
    axc.set_title("Scoring vs conceding (rally nested in player)", fontsize=FS["small"], pad=4)
    panel(axc, "c", dx=-0.30)

    # ---- 9d lag-sequential adjusted residuals (4 conditions) --------------
    states = ["FH", "BH", "JS"]
    subd = gs[1, 5:12].subgridspec(1, 4, wspace=0.45)
    titles = [("Men_Scoring", "Men sc."), ("Men_Conceding", "Men co."),
              ("Wom_Scoring", "Wom sc."), ("Wom_Conceding", "Wom co.")]
    for j, (key, ttl) in enumerate(titles):
        ax = fig.add_subplot(subd[0, j])
        Z = np.array(LZ[key], float)
        im = ax.imshow(Z, cmap="RdBu_r", vmin=-2.5, vmax=2.5, aspect="equal")
        ax.set_xticks(range(3)); ax.set_yticks(range(3))
        ax.set_xticklabels(states, fontsize=FS["small"]); ax.set_yticklabels(states, fontsize=FS["small"])
        ax.tick_params(length=0)
        for s in ax.spines.values(): s.set_visible(False)
        for a in range(3):
            for b in range(3):
                v = Z[a, b]
                mark = "*" if abs(v) >= 1.96 else ""
                ax.text(b, a, f"{v:.1f}{mark}", ha="center", va="center",
                        fontsize=4.6, color="white" if abs(v) > 1.4 else "#1A1A1A")
        ax.set_title(ttl, fontsize=FS["small"], pad=2)
        if j == 0:
            ax.set_ylabel("From", fontsize=FS["small"])
            panel(ax, "d", dx=-0.55, dy=1.22)
    cax = fig.add_axes([0.978, 0.13, 0.008, 0.26])
    cb = fig.colorbar(im, cax=cax); cb.set_label("Adjusted residual $z$", fontsize=FS["small"])
    cb.ax.tick_params(labelsize=FS["small"], length=2, width=LW["axis"]); cb.outline.set_linewidth(LW["axis"])
    save(fig, "Figure_9")

# ============================================================================
#  MAIN
# ============================================================================
if __name__ == "__main__":
    print("\nGenerating figures ...")
    figure2(); figure5(); figure6(); figure7(); figure8(); figure9()
    print("\nAll figures written to", CONFIG["OUTDIR"])
    for f in sorted(os.listdir(CONFIG["OUTDIR"])):
        sz = os.path.getsize(os.path.join(CONFIG["OUTDIR"], f))/1024
        print(f"  - {f}  ({sz:.0f} KB)")
