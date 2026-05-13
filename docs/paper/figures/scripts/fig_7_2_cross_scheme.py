"""F7.2 — Cross-scheme comparison: learned dual-head selector across 4 geodesic schemes.

Cells used at w = 0.5: pl_learned_intp, mx_learned_intp, fr_learned_intp,
ot_learned_intp. Plus Wald and identity-WALDO baselines for reference.

Run from repo root:
    python docs/paper/figures/scripts/fig_7_2_cross_scheme.py
"""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[4]
w_focus = 0.5

SCHEMES = [
    ("pl_learned_intp",   "PL (e-geodesic)",      "#d62728",  "o"),
    ("mx_learned_intp",   "MX (m-geodesic)",      "#2ca02c",  "s"),
    ("fr_learned_intp",   "FR (Levi-Civita)",     "#7e3a93",  "^"),
    ("ot_learned_intp",   "OT (W₂)",              "#1f4e79",  "D"),
]


def load_width(cell):
    path = REPO / "results" / "wald_audit" / cell / "width" / "mean_width.csv"
    data = {}
    with path.open() as f:
        for r in csv.DictReader(f):
            data[(float(r["theta_true"]), float(r["w"]))] = (
                float(r["mean_width"]), float(r["width_se"])
            )
    return data


fig, ax = plt.subplots(figsize=(7.6, 4.6))

# Baselines.
wald = load_width("wald")
ts_wald = sorted({t for (t, w) in wald if abs(w - w_focus) < 1e-6})
ax.plot(ts_wald, [wald[(t, w_focus)][0] for t in ts_wald],
        color="#444", lw=1.8, ls="--", marker="x", markersize=6,
        label="Wald (baseline)")

waldo = load_width("waldo")
ts_waldo = sorted({t for (t, w) in waldo if abs(w - w_focus) < 1e-6})
ax.plot(ts_waldo, [waldo[(t, w_focus)][0] for t in ts_waldo],
        color="#888", lw=1.5, ls=":", marker="s", markersize=5,
        label="identity-WALDO (η=0)")

# Cross-scheme cells.
for cell, label, color, marker in SCHEMES:
    data = load_width(cell)
    ts = sorted({t for (t, w) in data if abs(w - w_focus) < 1e-6})
    means = np.array([data[(t, w_focus)][0] for t in ts])
    ses = np.array([data[(t, w_focus)][1] for t in ts])
    ax.errorbar(ts, means, yerr=2 * ses,
                fmt=f"-{marker}", lw=1.8, ms=6,
                color=color, ecolor=color, capsize=3, label=label)

ax.axvline(0.0, color="#888", lw=0.7, ls=":", alpha=0.4)

ax.set_xlabel(r"true parameter $\theta_{\rm true}$")
ax.set_ylabel(r"mean CI width at $\alpha = 0.05$")
ax.set_xlim(-3.4, 4.4)
ax.set_ylim(2.8, 6.0)
ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.30), frameon=False,
          fontsize=9, ncol=3)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.set_title(rf"Cross-scheme width: learned dual-head × 4 geodesics at $w = {w_focus}$",
             fontsize=12, pad=10)

out_path = Path(__file__).parent.parent / "fig_7_2_cross_scheme.png"
fig.tight_layout()
fig.savefig(out_path, dpi=180, bbox_inches="tight")
print(f"wrote {out_path}")
