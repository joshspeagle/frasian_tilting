"""F7.3 — Smoothness comparison: Lipschitz and total-variation metrics on η* and width across schemes.

Cells used: pl_dyn_numerical, mx_dyn_numerical, fr_dyn_numerical, ot_dyn_numerical.
The smoothness experiment measures the per-θ-optimal η*(|Δ|) curve and the
resulting width vs Δ curve. FR cell missing data flagged as N/A.

Run from repo root:
    python docs/paper/figures/scripts/fig_7_3_smoothness.py
"""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[4]

SCHEMES = [
    ("pl_dyn_numerical", "PL (e)",        "#d62728"),
    ("mx_dyn_numerical", "MX (m)",        "#2ca02c"),
    ("fr_dyn_numerical", "FR (Levi-C)",   "#7e3a93"),
    ("ot_dyn_numerical", "OT (W₂)",       "#1f4e79"),
]


def load_metrics(cell):
    path = REPO / "results" / "wald_audit" / cell / "smoothness" / "smoothness.csv"
    out = {}
    with path.open() as f:
        for r in csv.DictReader(f):
            try:
                out[r["metric"]] = float(r["value"])
            except ValueError:
                out[r["metric"]] = np.nan
    return out


all_data = {name: load_metrics(cell) for cell, name, _ in SCHEMES}

# Metric ordering for the bar chart.
metric_groups = [
    ("Lipschitz", ["lipschitz_eta", "lipschitz_width"]),
    ("Total variation", ["total_variation_eta", "total_variation_width"]),
]

fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))

bar_width = 0.18
x = np.arange(2)  # two subgroups per panel: η, width

for ax, (group_name, metrics) in zip(axes, metric_groups):
    for i, (cell, name, color) in enumerate(SCHEMES):
        vals = [all_data[name].get(m, np.nan) for m in metrics]
        offset = (i - 1.5) * bar_width
        bars = ax.bar(x + offset, vals, bar_width, color=color, label=name,
                      alpha=0.85, edgecolor="white", linewidth=0.7)
        # Mark missing as a hatched outline.
        for bar_idx, val in enumerate(vals):
            if np.isnan(val):
                ax.text(x[bar_idx] + offset, 0.05, "N/A", rotation=90,
                        fontsize=8.5, color=color, ha="center", va="bottom")

    ax.set_xticks(x)
    ax.set_xticklabels([r"on $\eta^*(\Delta)$", r"on $W_\alpha(\Delta)$"], fontsize=10)
    ax.set_ylabel(f"{group_name} metric")
    ax.set_title(f"{group_name} smoothness", fontsize=12, pad=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(0, max(ax.get_ylim()[1] * 1.1, 0.1))

axes[0].legend(loc="upper right", frameon=False, fontsize=9.5, title="Scheme",
               title_fontsize=10)

fig.suptitle(
    "Smoothness metrics for η*(|Δ|) and width vs Δ across schemes (dyn_numerical selector, w-grid)",
    fontsize=11.5, y=1.02,
)

out_path = Path(__file__).parent.parent / "fig_7_3_smoothness.png"
fig.tight_layout()
fig.savefig(out_path, dpi=180, bbox_inches="tight")
print(f"wrote {out_path}")
for cell, name, _ in SCHEMES:
    print(f"  {name}: {all_data[name]}")
