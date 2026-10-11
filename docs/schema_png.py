#!/usr/bin/env python3
"""
Render the OpenTransportData pipeline schema as a PNG.

    python schema_png.py        ->  schema.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = "schema.png"
FIG_W, FIG_H, DPI = 16, 13, 200

# palette: (fill, edge)
SRC  = ("#dbeafe", "#2563eb")   # data sources
ING  = ("#dcfce7", "#16a34a")   # ingest
BLD  = ("#fef3c7", "#d97706")   # curated build
CMP  = ("#ede9fe", "#7c3aed")   # compute engines
OUTP = ("#ccfbf1", "#0d9488")   # aggregation / report
ARROW = "#475569"

fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), dpi=DPI)
ax.set_xlim(0, 100)
ax.set_ylim(0, 134)
ax.axis("off")
fig.patch.set_facecolor("white")


def box(cx, cy, w, h, title, subs=(), palette=SRC, title_fs=11, sub_fs=9):
    """Draw a rounded box with a bold title and optional subtitle lines."""
    fc, ec = palette
    ax.add_patch(FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0,rounding_size=1.5",
        linewidth=1.6, edgecolor=ec, facecolor=fc, zorder=2))

    lines = [(title, title_fs, "bold", "#0f172a")]
    lines += [(s, sub_fs, "normal", "#334155") for s in subs]

    step = h * 0.8 / len(lines)          # evenly distribute vertically
    y = cy + h * 0.4 - step * 0.5
    for text, fs, weight, col in lines:
        ax.text(cx, y, text, ha="center", va="center",
                fontsize=fs, fontweight=weight, color=col, zorder=3)
        y -= step


def arrow(p1, p2):
    ax.add_patch(FancyArrowPatch(
        p1, p2, arrowstyle="-|>", mutation_scale=16, lw=1.6,
        color=ARROW, shrinkA=0, shrinkB=0, zorder=1))


def line(p1, p2):
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=ARROW, lw=1.6,
            solid_capstyle="round", zorder=1)


# ── sources ───────────────────────────────────────────────────────────
ax.text(12, 131, "opentransportdata.swiss", ha="center", va="center",
        fontsize=11.5, fontweight="bold", color="#1e3a8a")

box(12, 124, 22, 8, "daily CSV", ["(50-day window)"], SRC)
box(12, 112, 22, 8, "monthly ZIP archive", ["(backfill)"], SRC)

# ── ingest ────────────────────────────────────────────────────────────
box(38, 118, 26, 8, "ingest.py / backfill.py", (), ING, title_fs=11)
box(38, 102, 34, 10, "raw Parquet",
    ["one file per day, canonical schema"], ING)

arrow((23, 124), (25, 120))
arrow((23, 112), (25, 116))
arrow((38, 114), (38, 107))

# ── curated build ─────────────────────────────────────────────────────
box(38, 92, 24, 6, "normalize_raw.py", (), ING, title_fs=10)
box(38, 83.5, 22, 7, "build_curated.py", (), BLD, title_fs=11.5)
box(38, 71.5, 40, 12, "curated/",
    ["integer facts + append-only dimensions",
     "905M rows · 1.6 GB · loads in 0.2 s"], BLD)

arrow((38, 97), (38, 95))
arrow((38, 89), (38, 87))
arrow((38, 80), (38, 77.5))

# ── fan-out to the three engines ──────────────────────────────────────
FX, FY, FW, FH = (16, 50, 84), 52, 30, 16

box(FX[0], FY, FW, FH, "f2_curated.py",
    ["group statistics", "(CuPy engines)"], CMP)
box(FX[1], FY, FW, FH, "f3_bootstrap.py",
    ["quantile bootstrap", "(custom CUDA kernel)"], CMP)
box(FX[2], FY, FW, FH, "f4_yoy.py / f41_yoy_panel.py",
    ["day-block bootstrap,", "panel, FDR", "(cuBLAS SGEMM)"], CMP,
    title_fs=10)

BUS_OUT = 62
line((38, 65.5), (38, BUS_OUT))                 # drop from curated/
line((FX[0], BUS_OUT), (FX[-1], BUS_OUT))     # horizontal bus
for x in FX:
    arrow((x, BUS_OUT), (x, FY + FH / 2))

# ── fan-in ────────────────────────────────────────────────────────────
BUS_IN = 40
for x in FX:
    line((x, FY - FH / 2), (x, BUS_IN))
line((FX[0], BUS_IN), (FX[-1], BUS_IN))
arrow((50, BUS_IN), (50, 36.5))
# reporting policy sits between the engines and the aggregates

# ── aggregation + report ──────────────────────────────────────────────
box(50, 33, 34, 7, "f31_ci_rule.py", ["reporting policy: n floor, precision tier"], CMP, title_fs=10.5, sub_fs=8.5)
box(50, 23, 26, 7, "agg/*.parquet", (), OUTP, title_fs=11.5)
box(50, 13.5, 24, 6.5, "f5_report.py", (), OUTP, title_fs=11)
box(50, 4.5, 22, 6.5, "out/report.html", (), OUTP, title_fs=11)

arrow((50, 29.5), (50, 26.5))
arrow((50, 19.5), (50, 16.75))
arrow((50, 10.25), (50, 7.75))

fig.savefig(OUT, dpi=DPI, bbox_inches="tight", facecolor="white")
print(f"wrote {OUT}")