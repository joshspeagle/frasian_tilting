"""F7.1 — PL selector ladder: width vs θ_true comparing four selectors at w = 0.5.

Cells used:
  pl_fixed0          identity-WALDO (η = 0)
  pl_numerical_intp  static-numerical (post-selection)
  pl_dyn_numerical   dynamic-numerical (calibrated)
  pl_learned_intp    learned dual-head (integrated_p loss)

Plus the Wald baseline (from results/wald_audit/wald) as a flat reference.

Run from repo root:
    python docs/paper/figures/scripts/fig_7_1_selector_ladder.py
"""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[4]
w_focus = 0.5

CELLS = [
    ("pl_fixed0",          "identity (η=0, WALDO)",         "#888",     "s"),
    ("pl_numerical_intp",  "static-numerical (η from D)",   "#d62728",  "o"),
    ("pl_dyn_numerical",   "dynamic-numerical (η(θ))",       "#1f4e79",  "^"),
    ("pl_learned_intp",    "learned dual-head",              "#2ca02c",  "D"),
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

# Wald reference.
wald = load_width("wald")
ts_wald = sorted({t for (t, w) in wald if abs(w - w_focus) < 1e-6})
wald_w = [wald[(t, w_focus)][0] for t in ts_wald]
ax.plot(ts_wald, wald_w, color="#444", lw=2.0, ls="--",
        marker="x", markersize=7, label="Wald (baseline)")

for cell, label, color, marker in CELLS:
    data = load_width(cell)
    ts = sorted({t for (t, w) in data if abs(w - w_focus) < 1e-6})
    means = np.array([data[(t, w_focus)][0] for t in ts])
    ses = np.array([data[(t, w_focus)][1] for t in ts])
    ax.errorbar(ts, means, yerr=2 * ses,
                fmt=f"-{marker}", lw=1.8, ms=6,
                color=color, ecolor=color, capsize=3, label=label)

ax.axvline(0.0, color="#888", lw=0.7, ls=":", alpha=0.4)
ax.text(0.1, 5.5, r"$\theta = \mu_0$", color="#888", fontsize=10)

ax.set_xlabel(r"true parameter $\theta_{\rm true}$")
ax.set_ylabel(r"mean CI width at $\alpha = 0.05$")
ax.set_xlim(-3.4, 4.4)
ax.set_ylim(2.8, 6.0)
ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.30), frameon=False,
          fontsize=9.5, ncol=3)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.set_title(rf"PL selector ladder: width vs $\theta_{{\rm true}}$ at $w = {w_focus}$",
             fontsize=12, pad=10)

out_path = Path(__file__).parent.parent / "fig_7_1_selector_ladder.png"
fig.tight_layout()
fig.savefig(out_path, dpi=180, bbox_inches="tight")
print(f"wrote {out_path}")
