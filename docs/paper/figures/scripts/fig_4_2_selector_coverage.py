"""F4.2 — Selector coverage comparison: static-numerical undercovers; dynamic and learned stay calibrated.

Reads audit data from four PL-tilting cells at w = 0.5:
  - pl_fixed0:        identity-WALDO baseline (η = 0 fixed)
  - pl_numerical_intp: static-numerical selector (η chosen from D — post-selection)
  - pl_dyn_numerical: dynamic-numerical selector (η(θ), not from D)
  - pl_learned_intp:  learned dual-head selector (EtaNet + ValidityNet)

Run from repo root:
    python docs/paper/figures/scripts/fig_4_2_selector_coverage.py
"""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[4]

CELLS = {
    "pl_fixed0":         {"label": "identity (η = 0, no selection)",          "color": "#444",     "marker": "s"},
    "pl_numerical_intp": {"label": "static-numerical (η from D)",             "color": "#d62728",  "marker": "o"},
    "pl_dyn_numerical":  {"label": "dynamic-numerical (η(θ), not from D)",    "color": "#1f4e79",  "marker": "^"},
    "pl_learned_intp":   {"label": "learned dual-head (EtaNet + ValidityNet)", "color": "#2ca02c", "marker": "D"},
}

w_focus = 0.5


def load_coverage(cell):
    path = REPO / "results" / "wald_audit" / cell / "coverage" / "coverage_rate.csv"
    data = {}
    with path.open() as f:
        for r in csv.DictReader(f):
            data[(float(r["theta_true"]), float(r["w"]))] = (
                float(r["coverage"]), float(r["coverage_se"])
            )
    return data


fig, ax = plt.subplots(figsize=(7.5, 4.4))

# Nominal coverage band ±2pp.
ax.axhline(0.95, color="#666", lw=1.2, ls="--", alpha=0.7,
           label=r"nominal $1 - \alpha = 0.95$")
ax.axhspan(0.93, 0.97, color="#666", alpha=0.08)

for cell, style in CELLS.items():
    data = load_coverage(cell)
    ts_w = sorted(t for (t, w) in data if abs(w - w_focus) < 1e-6)
    means = np.array([data[(t, w_focus)][0] for t in ts_w])
    ses = np.array([data[(t, w_focus)][1] for t in ts_w])
    ax.errorbar(ts_w, means, yerr=2 * ses,
                fmt=f"-{style['marker']}", lw=1.8, ms=6,
                color=style["color"], ecolor=style["color"], capsize=3,
                label=style["label"])

# Annotation for the post-selection bias.
ax.annotate(
    "post-selection bias:\n~2–3 pp undercoverage",
    xy=(3.0, 0.92), xytext=(-2.5, 0.895),
    fontsize=9.5, color="#d62728",
    arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.0),
)

ax.axvline(0.0, color="#2ca02c", lw=0.8, ls=":", alpha=0.4)

ax.set_xlabel(r"true parameter $\theta_{\rm true}$ (prior at $\mu_0 = 0$)")
ax.set_ylabel(r"empirical coverage")
ax.set_xlim(-3.4, 4.4)
ax.set_ylim(0.88, 1.02)
ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.34),
          frameon=False, fontsize=9, ncol=3)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.set_title(fr"Selector coverage on NN+Normal ($w = {w_focus}$, $\alpha = 0.05$)",
             fontsize=12, pad=10)

out_path = Path(__file__).parent.parent / "fig_4_2_selector_coverage.png"
fig.tight_layout()
fig.savefig(out_path, dpi=180, bbox_inches="tight")
print(f"wrote {out_path}")
