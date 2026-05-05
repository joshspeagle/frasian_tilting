"""Coverage-rate diagnostic: tidy DataFrame + heatmap figure."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..experiments.base import RawResult
from .base import DiagnosticTable


@dataclass(frozen=True)
class CoverageRateDiagnostic:
    """Compute and render the coverage-rate table."""

    name: ClassVar[str] = "coverage_rate"

    def compute(self, raw: RawResult) -> DiagnosticTable:
        theta_grid = raw.arrays["theta_grid"]
        w_grid = raw.arrays["w_grid"]
        coverage = raw.arrays["coverage"]
        se = raw.arrays["coverage_se"]
        records = []
        for i, theta in enumerate(theta_grid):
            for j, w in enumerate(w_grid):
                records.append(
                    {
                        "experiment": raw.experiment,
                        "tilting": raw.tilting,
                        "statistic": raw.statistic,
                        "theta_true": float(theta),
                        "w": float(w),
                        "coverage": float(coverage[i, j]),
                        "coverage_se": float(se[i, j]),
                    }
                )
        df = pd.DataFrame.from_records(records)
        return DiagnosticTable(
            name=self.name,
            table=df,
            units={
                "theta_true": "param units",
                "w": "(0,1)",
                "coverage": "fraction",
                "coverage_se": "fraction",
            },
            metadata={"alpha": raw.metadata.get("alpha"), "n_reps": raw.metadata.get("n_reps")},
        )

    def render(self, table: DiagnosticTable, fig_dir: Path) -> Path:
        df = table.table
        if df.empty:
            raise ValueError("empty coverage table; cannot render")

        # One subplot per (tilting, statistic).
        groups = df.groupby(["tilting", "statistic"], sort=False)
        n = len(groups)
        ncols = min(n, 3)
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 3.2 * nrows), squeeze=False)
        alpha_raw = table.metadata.get("alpha", 0.05)
        alpha = float(alpha_raw) if alpha_raw is not None else None  # type: ignore[arg-type]
        nominal = 1.0 - alpha if alpha is not None else None

        for ax_idx, ((tilting, statistic), gdf) in enumerate(groups):
            r, c = divmod(ax_idx, ncols)
            ax = axes[r][c]
            theta_vals = np.sort(gdf["theta_true"].unique())
            w_vals = np.sort(gdf["w"].unique())
            grid = (
                gdf.pivot(index="theta_true", columns="w", values="coverage")
                .reindex(index=theta_vals, columns=w_vals)
                .to_numpy()
            )
            im = ax.imshow(
                grid,
                aspect="auto",
                origin="lower",
                extent=[w_vals[0], w_vals[-1], theta_vals[0], theta_vals[-1]],
                vmin=0.0,
                vmax=1.0,
                cmap="viridis",
            )
            ax.set_title(f"{tilting} x {statistic}", fontsize=9)
            ax.set_xlabel("w")
            ax.set_ylabel(r"$\theta_\mathrm{true}$")
            if nominal is not None:
                ax.text(
                    0.02,
                    0.95,
                    f"nominal {nominal:.0%}",
                    color="w",
                    transform=ax.transAxes,
                    fontsize=8,
                    va="top",
                )
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        for k in range(n, nrows * ncols):
            r, c = divmod(k, ncols)
            axes[r][c].axis("off")

        fig.suptitle("Empirical coverage rate", fontsize=11)
        fig.tight_layout()
        fig_dir.mkdir(parents=True, exist_ok=True)
        out = fig_dir / "coverage_rate.png"
        fig.savefig(out, dpi=130)
        plt.close(fig)
        return out
