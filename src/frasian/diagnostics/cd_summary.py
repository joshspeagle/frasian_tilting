"""CDSummaryDiagnostic: 4-panel CD heatmap per cell.

For each (tilting, statistic) cell, four heatmaps over (θ_true, w):
  - CD-median
  - CD 95% interval width
  - W₁ distance to Wald reference CD
  - Fraction of replicates with non-monotone signed_confidence

Long-format DataFrame with one row per (cell, theta_true, w); CSV
columns include all four metrics + standard errors.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..experiments.base import RawResult
from .base import DiagnosticTable


@dataclass(frozen=True)
class CDSummaryDiagnostic:
    """Per-cell heatmaps of CD distributional summaries."""

    name: str = "cd_summary"

    def compute(self, raw: RawResult) -> DiagnosticTable:
        theta_grid = raw.arrays["theta_grid"]
        w_grid = raw.arrays["w_grid"]
        cd_median = raw.arrays["cd_median"]
        cd_median_se = raw.arrays["cd_median_se"]
        cd_width = raw.arrays["cd_width_95"]
        cd_width_se = raw.arrays["cd_width_95_se"]
        w1 = raw.arrays["w1_to_wald_cd"]
        w1_se = raw.arrays["w1_to_wald_cd_se"]
        nm = raw.arrays["nonmonotone_fraction"]

        records = []
        for i, theta in enumerate(theta_grid):
            for j, w in enumerate(w_grid):
                records.append({
                    "experiment": raw.experiment,
                    "tilting": raw.tilting,
                    "statistic": raw.statistic,
                    "theta_true": float(theta),
                    "w": float(w),
                    "cd_median": float(cd_median[i, j]),
                    "cd_median_se": float(cd_median_se[i, j]),
                    "cd_width_95": float(cd_width[i, j]),
                    "cd_width_95_se": float(cd_width_se[i, j]),
                    "w1_to_wald_cd": float(w1[i, j]),
                    "w1_to_wald_cd_se": float(w1_se[i, j]),
                    "nonmonotone_fraction": float(nm[i, j]),
                })
        df = pd.DataFrame.from_records(records)
        return DiagnosticTable(
            name=self.name,
            table=df,
            units={
                "theta_true": "param units", "w": "(0,1)",
                "cd_median": "param units",
                "cd_width_95": "param units",
                "w1_to_wald_cd": "param units (W₁)",
                "nonmonotone_fraction": "fraction in [0,1]",
            },
            metadata={
                "alpha": raw.metadata.get("alpha"),
                "n_reps": raw.metadata.get("n_reps"),
                "n_grid_cd": raw.metadata.get("n_grid_cd"),
            },
        )

    def render(self, table: DiagnosticTable, fig_dir: Path) -> Path:
        df = table.table
        if df.empty:
            raise ValueError("empty cd_summary table; cannot render")

        groups = list(df.groupby(["tilting", "statistic"], sort=False))
        n_cells = len(groups)

        # 4 metric columns x n_cells rows.
        metrics = [
            ("cd_median", "CD median", "viridis"),
            ("cd_width_95", "CD 95% width", "magma"),
            ("w1_to_wald_cd", "W₁ to Wald CD", "plasma"),
            ("nonmonotone_fraction", "non-monotone fraction", "cividis"),
        ]
        fig, axes = plt.subplots(
            n_cells, 4, figsize=(13, 3.0 * max(n_cells, 1)), squeeze=False,
        )

        # Common colour scale per metric across cells where possible.
        scales: dict[str, tuple[float, float]] = {}
        for col, *_ in metrics:
            vals = df[col][np.isfinite(df[col])]
            if vals.empty:
                scales[col] = (0.0, 1.0)
            elif col == "nonmonotone_fraction":
                scales[col] = (0.0, 1.0)
            else:
                scales[col] = (float(vals.min()), float(vals.max()))

        for r_idx, ((tilting, statistic), gdf) in enumerate(groups):
            theta_vals = np.sort(gdf["theta_true"].unique())
            w_vals = np.sort(gdf["w"].unique())
            extent = [w_vals[0], w_vals[-1], theta_vals[0], theta_vals[-1]]

            def _grid(col: str, _gdf=gdf, _theta=theta_vals, _w=w_vals) -> np.ndarray:
                return (_gdf.pivot(index="theta_true", columns="w", values=col)
                           .reindex(index=_theta, columns=_w)
                           .to_numpy())

            for c_idx, (col, title, cmap) in enumerate(metrics):
                ax = axes[r_idx][c_idx]
                vmin, vmax = scales[col]
                im = ax.imshow(_grid(col), aspect="auto", origin="lower",
                                extent=extent, cmap=cmap,
                                vmin=vmin, vmax=vmax)
                if r_idx == 0:
                    ax.set_title(title, fontsize=9)
                if c_idx == 0:
                    ax.set_ylabel(rf"$\theta_\mathrm{{true}}$", fontsize=8)
                if r_idx == n_cells - 1:
                    ax.set_xlabel("w", fontsize=8)
                # Cell-row label on the leftmost column.
                if c_idx == 0:
                    ax.text(-0.42, 0.5, f"{tilting}\n×{statistic}",
                            transform=ax.transAxes, fontsize=8,
                            va="center", ha="right")
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        fig.suptitle("Confidence-distribution summaries", fontsize=11)
        fig.tight_layout(rect=[0.06, 0, 1, 0.97])
        fig_dir.mkdir(parents=True, exist_ok=True)
        out = fig_dir / "cd_summary.png"
        fig.savefig(out, dpi=130)
        plt.close(fig)
        return out
