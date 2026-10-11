# Part of the ADMET-Stack revision analysis set.
# Fig. S6: QED against the aggregate ADMET risk-favorability score.
# Inputs: data/honest_selection.json
# -*- coding: utf-8 -*-
"""Fig. S6 - selection optimism against test-set size.

Design notes
------------
y is *relative* optimism, as a percentage of the best attainable score on the held-out half.
Absolute optimism cannot be plotted on one axis across these endpoints: five are scored by
MAE, whose values run to ~7 (PPBR_AZ), while the other seventeen are AUROC, AUPRC or
Spearman and are bounded by 1. An absolute axis would therefore mix two scales and let
PPBR_AZ dominate for a reason that has nothing to do with selection. Dividing by the oracle
score makes all 22 endpoints comparable and leaves the claim - that optimism falls with
test-set size - intact. Table S40 carries the absolute values.

One series, so no legend for the points; the axis titles name them. Log x, because the sizes
span 95 to 2,626. The 1/sqrt(n) reference curve makes the scaling claim visible rather than
asserted. Only the endpoints that carry the argument are labelled.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

SP = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = (r"E:\ADMET_Stack_project_for_review261003\ADMET-Stack_manuscript_submitted"
           r"\1991510_SupMaterial")

# ---------------------------------------------------------------- data
with open(os.path.join(SP, "data_honest_selection.json"), encoding="utf8") as fh:
    R = json.load(fh)

rows = []
for ep, d in R.items():
    lower = d["metric"] == "MAE"
    absolute = (d["sel"] - d["oracle"]) if lower else (d["oracle"] - d["sel"])
    relative = absolute / abs(d["oracle"]) * 100.0
    rows.append((ep, d["n_test"], absolute, relative, d["metric"]))
rows.sort(key=lambda r: r[1])

names = [r[0] for r in rows]
n = np.array([r[1] for r in rows], float)
rel = np.array([r[3] for r in rows], float)

# ---------------------------------------------------------------- style
INK, MUTED, GRID = "#1a1a1a", "#5c5c5c", "#dcdcdc"
ACCENT, REF = "#2a6f97", "#a8b2ba"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.5,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.linewidth": 0.8, "xtick.major.width": 0.8, "ytick.major.width": 0.8,
})

fig, ax = plt.subplots(figsize=(7.1, 4.0), dpi=600)

k = np.median(rel * np.sqrt(n))
xs = np.logspace(np.log10(70), np.log10(3400), 300)
ax.plot(xs, k / np.sqrt(xs), color=REF, lw=1.6, ls=(0, (5, 3)), zorder=1)
ax.annotate(r"$\propto 1/\sqrt{n}$", xy=(1500, k / np.sqrt(1500)),
            textcoords="offset points", xytext=(4, 9), fontsize=8, color=MUTED)

ax.scatter(n, rel, s=44, facecolor=ACCENT, edgecolor="white", linewidth=0.9, zorder=3)

# ---------------------------------------------------------------- labels
SHORT = {
    "CYP2C9_Substrate_CarbonMangels": "CYP2C9_Sub.",
    "CYP2D6_Substrate_CarbonMangels": "CYP2D6_Sub.",
    "CYP3A4_Substrate_CarbonMangels": "CYP3A4_Sub.",
    "Clearance_Hepatocyte_AZ": "Clearance_H.",
    "Clearance_Microsome_AZ": "Clearance_M.",
    "Half_Life_Obach": "Half_Life",
    "Bioavailability_Ma": "Bioavailability",
    "Lipophilicity_AstraZeneca": "Lipophilicity",
    "Solubility_AqSolDB": "Solubility",
    "Pgp_Broccatelli": "Pgp",
}
# (dx, dy) in points; chosen after rendering, to keep every label clear of a mark
OFFSET = {
    "CYP2C9_Substrate_CarbonMangels": (9, 2),
    "Half_Life_Obach": (9, -1),
    "CYP3A4_Substrate_CarbonMangels": (9, -2),
    "CYP2D6_Substrate_CarbonMangels": (-9, 4),
    "Clearance_Hepatocyte_AZ": (9, -2),
    "DILI": (9, 1),
    "hERG": (-9, 3),
}
for i, nm in enumerate(names):
    if nm not in OFFSET:
        continue
    dx, dy = OFFSET[nm]
    ax.annotate(SHORT.get(nm, nm), (n[i], rel[i]), textcoords="offset points",
                xytext=(dx, dy), fontsize=7.2, color=INK,
                ha="left" if dx > 0 else "right", va="center", zorder=4)

# ---------------------------------------------------------------- axes
ax.set_xscale("log")
ax.set_xlabel("Number of test compounds", labelpad=6)
ax.set_ylabel("Selection optimism\n(% of the best attainable score)", labelpad=6)
ax.set_xlim(70, 3400)
ax.set_ylim(-0.7, max(rel) * 1.22)
ax.set_xticks([100, 200, 500, 1000, 2000])
ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
ax.set_axisbelow(True)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)

small, large = rel[n < 300], rel[n >= 1000]
ax.annotate(f"n < 300  ({len(small)} endpoints)\nmedian {np.median(small):.1f}%",
            xy=(0.015, 0.97), xycoords="axes fraction", fontsize=7.6, color=MUTED,
            ha="left", va="top", linespacing=1.5)
ax.annotate(f"n $\\geq$ 1000  ({len(large)} endpoints)\nmedian {np.median(large):.1f}%",
            xy=(0.985, 0.97), xycoords="axes fraction", fontsize=7.6, color=MUTED,
            ha="right", va="top", linespacing=1.5)

rho, p = stats.spearmanr(n, rel)
ax.annotate(f"Spearman $\\rho$ = {rho:.2f},  p = {p:.3f}",
            xy=(0.52, 0.995), xycoords="axes fraction", fontsize=7.6, color=MUTED,
            ha="center", va="top")

fig.tight_layout()
for ext in ("png", "pdf"):
    path = os.path.join(OUT_DIR, f"Fig_S6_selection_optimism.{ext}")
    fig.savefig(path, dpi=600, bbox_inches="tight", facecolor="white")
    print("written:", path)

print(f"\nn range         : {int(n.min())} - {int(n.max())}")
print(f"relative optimism: {rel.min():.2f}% - {rel.max():.2f}%")
print(f"median n < 300   : {np.median(small):.2f}%   (absolute {np.median([r[2] for r in rows if r[1] < 300]):.4f})")
print(f"median n >= 1000 : {np.median(large):.2f}%   (absolute {np.median([r[2] for r in rows if r[1] >= 1000]):.4f})")
print(f"Spearman rho     : {rho:.3f}  p = {p:.4f}")
