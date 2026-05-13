"""F3.1 — The conflict tax: WALDO CI width inflates with prior–data conflict, Wald doesn't.

Reads existing audit data from `results/wald_audit/{wald,waldo}/width/mean_width.csv`
and plots mean CI width as a function of true θ for both statistics, at three
prior strengths (w values).

Headline: Wald width is constant (no prior); WALDO is narrower than Wald at low
conflict (θ_true ≈ μ_0) and wider than Wald at high conflict.

Run from repo root:
    python docs/paper/figures/scripts/fig_3_1_width_vs_conflict.py
"""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[4]


def load_width(name):
    path = REPO / "results" / "wald_audit" / name / "width" / "mean_width.csv"
    data = {}
    with path.open() as f:
        for r in csv.DictReader(f):
            data[(float(r["theta_true"]), float(r["w"]))] = (
                float(r["mean_width"]), float(r["width_se"])
            )
    return data


wald = load_width("wald")
waldo = load_width("waldo")

ts = sorted({t for (t, w) in wald})

fig, ax = plt.subplots(figsize=(7.2, 4.4))

# Wald is constant in w (and θ_true).
wald_w = [wald[(t, 0.5)][0] for t in ts]
ax.plot(ts, wald_w, color="#444", lw=2.2, ls="--",
        marker="s", markersize=6, label="Wald (any $w$)")

# WALDO at three prior strengths.
waldo_colors = {0.2: "#1f4e79", 0.5: "#5882c2", 0.8: "#a5c0e0"}
for w_val, color in waldo_colors.items():
    means = np.array([waldo[(t, w_val)][0] for t in ts])
    ses = np.array([waldo[(t, w_val)][1] for t in ts])
    ax.errorbar(ts, means, yerr=2 * ses, fmt="-o", lw=2.0, ms=5,
                color=color, ecolor=color, capsize=3,
                label=fr"WALDO ($w = {w_val}$)")

# Shade prior center (mu_0 inferred = 0).
ax.axvline(0.0, color="#2ca02c", lw=1.0, ls=":", alpha=0.6)
ax.text(0.12, 5.3, r"$\theta = \mu_0$", color="#2ca02c", fontsize=10, va="top")

# Annotations.
ax.annotate(
    "WALDO narrower\n(prior helpful)",
    xy=(0.0, 3.4), xytext=(-2.8, 3.0),
    fontsize=10, color="#1f4e79",
    arrowprops=dict(arrowstyle="->", color="#1f4e79", lw=1.0),
)
ax.annotate(
    "WALDO wider\n(prior fights data)",
    xy=(4.0, 5.38), xytext=(1.6, 5.7),
    fontsize=10, color="#1f4e79",
    arrowprops=dict(arrowstyle="->", color="#1f4e79", lw=1.0),
)

ax.set_xlabel(r"true parameter $\theta_{\rm true}$ (with prior at $\mu_0 = 0$)")
ax.set_ylabel(r"mean CI width at $\alpha = 0.05$")
ax.set_xlim(-3.4, 4.4)
ax.set_ylim(2.7, 6.0)
ax.legend(loc="upper left", frameon=True, fontsize=9.5,
          framealpha=0.92, edgecolor="#bbb")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.set_title("The conflict tax inside Frasian inference",
             fontsize=12, pad=10)

out_path = Path(__file__).parent.parent / "fig_3_1_width_vs_conflict.png"
fig.tight_layout()
fig.savefig(out_path, dpi=180, bbox_inches="tight")
print(f"wrote {out_path}")
