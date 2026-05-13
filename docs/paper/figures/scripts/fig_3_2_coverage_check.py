"""F3.2 — Coverage check for Wald and WALDO: both calibrated at nominal 1 - α.

Reads audit data from `results/wald_audit/{wald,waldo}/coverage/coverage_rate.csv`
and plots empirical coverage as a function of true θ for both statistics at
representative w values.

The conflict tax pays in width, not in coverage — this figure makes that
explicit (the coverage hovers near 0.95 across all θ_true).

Run from repo root:
    python docs/paper/figures/scripts/fig_3_2_coverage_check.py
"""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[4]


def load_coverage(name):
    path = REPO / "results" / "wald_audit" / name / "coverage" / "coverage_rate.csv"
    data = {}
    with path.open() as f:
        for r in csv.DictReader(f):
            data[(float(r["theta_true"]), float(r["w"]))] = (
                float(r["coverage"]), float(r["coverage_se"])
            )
    return data


wald = load_coverage("wald")
waldo = load_coverage("waldo")

ts = sorted({t for (t, w) in wald})
nominal = 0.95
alpha = 0.05

fig, ax = plt.subplots(figsize=(7.2, 4.0))

# Nominal coverage band: ±2 SE from MC estimate.
ax.axhline(nominal, color="#444", lw=1.2, ls="--", alpha=0.7,
           label=fr"nominal $1 - \alpha = {nominal}$")
ax.axhspan(nominal - 0.02, nominal + 0.02, color="#444", alpha=0.07)

# Wald is constant in w; plot once.
wald_mean = np.array([wald[(t, 0.5)][0] for t in ts])
wald_se = np.array([wald[(t, 0.5)][1] for t in ts])
ax.errorbar(ts, wald_mean, yerr=2 * wald_se, fmt="-s", lw=2.0, ms=7,
            color="#444", ecolor="#444", capsize=3, label="Wald")

# WALDO at three w values.
waldo_colors = {0.2: "#1f4e79", 0.5: "#5882c2", 0.8: "#a5c0e0"}
for w_val, color in waldo_colors.items():
    means = np.array([waldo[(t, w_val)][0] for t in ts])
    ses = np.array([waldo[(t, w_val)][1] for t in ts])
    # Small horizontal jitter to separate visually.
    jitter = 0.04 * (list(waldo_colors).index(w_val) - 1)
    ax.errorbar(np.array(ts) + jitter, means, yerr=2 * ses,
                fmt="-o", lw=1.5, ms=4.5,
                color=color, ecolor=color, capsize=2,
                label=fr"WALDO ($w = {w_val}$)", alpha=0.85)

# Prior center.
ax.axvline(0.0, color="#2ca02c", lw=1.0, ls=":", alpha=0.5)
ax.text(0.12, 0.99, r"$\theta = \mu_0$", color="#2ca02c", fontsize=10, va="top")

ax.set_xlabel(r"true parameter $\theta_{\rm true}$")
ax.set_ylabel(r"empirical coverage")
ax.set_xlim(-3.4, 4.4)
ax.set_ylim(0.88, 1.00)
ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.42), frameon=False,
          fontsize=9.5, ncol=3)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.set_title("Coverage check: both Wald and WALDO hit nominal $1 - \\alpha$",
             fontsize=12, pad=10)

out_path = Path(__file__).parent.parent / "fig_3_2_coverage_check.png"
fig.tight_layout()
fig.savefig(out_path, dpi=180, bbox_inches="tight")
print(f"wrote {out_path}")
