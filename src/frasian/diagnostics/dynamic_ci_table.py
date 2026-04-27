"""Diagnostic for DynamicCIExperiment: tidy table + 3-panel heatmap."""

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
class DynamicCITableDiagnostic:
    """Coverage, mean width, and mean region count for dynamic-eta CIs."""

    name: str = "dynamic_ci"

    def compute(self, raw: RawResult) -> DiagnosticTable:
        theta_grid = raw.arrays["theta_grid"]
        w_grid = raw.arrays["w_grid"]
        cov = raw.arrays["coverage"]
        cov_se = raw.arrays["coverage_se"]
        mw = raw.arrays["mean_width"]
        w_se = raw.arrays["width_se"]
        regs = raw.arrays["mean_regions"]
        records = []
        for i, theta in enumerate(theta_grid):
            for j, w in enumerate(w_grid):
                records.append({
                    "experiment": raw.experiment,
                    "tilting": raw.tilting,
                    "statistic": raw.statistic,
                    "theta_true": float(theta),
                    "w": float(w),
                    "coverage": float(cov[i, j]),
                    "coverage_se": float(cov_se[i, j]),
                    "mean_width": float(mw[i, j]),
                    "width_se": float(w_se[i, j]),
                    "mean_regions": float(regs[i, j]),
                })
        df = pd.DataFrame.from_records(records)
        return DiagnosticTable(
            name=self.name,
            table=df,
            units={"theta_true": "param units", "w": "(0,1)",
                   "coverage": "fraction", "mean_width": "param units",
                   "mean_regions": "count"},
            metadata={"alpha": raw.metadata.get("alpha"),
                       "n_reps": raw.metadata.get("n_reps")},
        )

    def render(self, table: DiagnosticTable, fig_dir: Path) -> Path:
        df = table.table
        if df.empty:
            raise ValueError("empty dynamic_ci table; cannot render")

        groups = list(df.groupby(["tilting", "statistic"], sort=False))
        n_cells = len(groups)
        # Three columns (coverage, width, regions); one row per cell.
        fig, axes = plt.subplots(
            n_cells, 3, figsize=(11, 3.0 * n_cells), squeeze=False,
        )
        alpha = table.metadata.get("alpha", 0.05)
        nominal = (1.0 - alpha) if alpha is not None else None

        # Common width scale across cells.
        finite_w = df["mean_width"][np.isfinite(df["mean_width"])]
        wmin = float(finite_w.min()) if not finite_w.empty else 0.0
        wmax = float(finite_w.max()) if not finite_w.empty else 1.0
        finite_r = df["mean_regions"][np.isfinite(df["mean_regions"])]
        rmin = float(finite_r.min()) if not finite_r.empty else 1.0
        rmax = max(2.0, float(finite_r.max()) if not finite_r.empty else 1.0)

        for r_idx, ((tilting, statistic), gdf) in enumerate(groups):
            theta_vals = np.sort(gdf["theta_true"].unique())
            w_vals = np.sort(gdf["w"].unique())

            def _grid(col):
                return (gdf.pivot(index="theta_true", columns="w", values=col)
                           .reindex(index=theta_vals, columns=w_vals).to_numpy())

            extent = [w_vals[0], w_vals[-1], theta_vals[0], theta_vals[-1]]

            # Coverage.
            ax = axes[r_idx][0]
            im = ax.imshow(_grid("coverage"), aspect="auto", origin="lower",
                            extent=extent, vmin=0.0, vmax=1.0, cmap="viridis")
            ax.set_title(f"{tilting} x {statistic}: coverage", fontsize=9)
            ax.set_xlabel("w"); ax.set_ylabel(r"$\theta_\mathrm{true}$")
            if nominal is not None:
                ax.text(0.02, 0.95, f"nominal {nominal:.0%}", color="w",
                         transform=ax.transAxes, fontsize=8, va="top")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            # Width.
            ax = axes[r_idx][1]
            im = ax.imshow(_grid("mean_width"), aspect="auto", origin="lower",
                            extent=extent, vmin=wmin, vmax=wmax, cmap="magma")
            ax.set_title("mean width", fontsize=9)
            ax.set_xlabel("w"); ax.set_ylabel(r"$\theta_\mathrm{true}$")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            # Regions.
            ax = axes[r_idx][2]
            im = ax.imshow(_grid("mean_regions"), aspect="auto", origin="lower",
                            extent=extent, vmin=rmin, vmax=rmax, cmap="cividis")
            ax.set_title("mean region count", fontsize=9)
            ax.set_xlabel("w"); ax.set_ylabel(r"$\theta_\mathrm{true}$")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        fig.suptitle("Dynamic-η CI diagnostics", fontsize=11)
        fig.tight_layout()
        fig_dir.mkdir(parents=True, exist_ok=True)
        out = fig_dir / "dynamic_ci.png"
        fig.savefig(out, dpi=130)
        plt.close(fig)
        return out
