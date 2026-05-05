"""Mean-CI-width diagnostic: tidy DataFrame + heatmap figure."""

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
class MeanWidthDiagnostic:
    """Compute and render the mean-CI-width table."""

    name: ClassVar[str] = "mean_width"

    def compute(self, raw: RawResult) -> DiagnosticTable:
        theta_grid = raw.arrays["theta_grid"]
        w_grid = raw.arrays["w_grid"]
        mean_width = raw.arrays["mean_width"]
        se = raw.arrays["width_se"]
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
                        "mean_width": float(mean_width[i, j]),
                        "width_se": float(se[i, j]),
                    }
                )
        df = pd.DataFrame.from_records(records)
        return DiagnosticTable(
            name=self.name,
            table=df,
            units={
                "theta_true": "param units",
                "w": "(0,1)",
                "mean_width": "param units",
                "width_se": "param units",
            },
            metadata={"alpha": raw.metadata.get("alpha"), "n_reps": raw.metadata.get("n_reps")},
        )

    def render(self, table: DiagnosticTable, fig_dir: Path) -> Path:
        df = table.table
        if df.empty:
            raise ValueError("empty width table; cannot render")
        groups = df.groupby(["tilting", "statistic"], sort=False)
        n = len(groups)
        ncols = min(n, 3)
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 3.2 * nrows), squeeze=False)

        # Common colour scale.
        finite = df["mean_width"][np.isfinite(df["mean_width"])]
        vmin = float(finite.min()) if not finite.empty else 0.0
        vmax = float(finite.max()) if not finite.empty else 1.0

        for ax_idx, ((tilting, statistic), gdf) in enumerate(groups):
            r, c = divmod(ax_idx, ncols)
            ax = axes[r][c]
            theta_vals = np.sort(gdf["theta_true"].unique())
            w_vals = np.sort(gdf["w"].unique())
            grid = (
                gdf.pivot(index="theta_true", columns="w", values="mean_width")
                .reindex(index=theta_vals, columns=w_vals)
                .to_numpy()
            )
            im = ax.imshow(
                grid,
                aspect="auto",
                origin="lower",
                extent=[w_vals[0], w_vals[-1], theta_vals[0], theta_vals[-1]],
                vmin=vmin,
                vmax=vmax,
                cmap="magma",
            )
            ax.set_title(f"{tilting} x {statistic}", fontsize=9)
            ax.set_xlabel("w")
            ax.set_ylabel(r"$\theta_\mathrm{true}$")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        for k in range(n, nrows * ncols):
            r, c = divmod(k, ncols)
            axes[r][c].axis("off")

        fig.suptitle("Mean CI width", fontsize=11)
        fig.tight_layout()
        fig_dir.mkdir(parents=True, exist_ok=True)
        out = fig_dir / "mean_width.png"
        fig.savefig(out, dpi=130)
        plt.close(fig)
        return out
