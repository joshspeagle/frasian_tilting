"""Smoothness diagnostic: quantify η*(|Δ|) and CI-endpoint regularity.

Operates on a fine sweep of |Δ| (from `SmoothnessExperiment`) and emits:
  - Local Lipschitz estimate L_hat = max_i |Δη*/Δ|Δ||
  - Total variation TV(η*) = Σ |η*_{i+1} - η*_i|
  - Discontinuity count (intervals where |2nd diff| > k * MAD-σ)
  - Spectral roughness (ratio of high-freq to low-freq FFT power)

Power-law tilting is expected to spike the Lipschitz / discontinuity
metrics at low |Δ|; smoother schemes (OT, geodesic) should not.
"""

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

_DISCONTINUITY_K = 3.0  # MAD-sigma threshold for second-difference outliers


def _local_lipschitz(x: np.ndarray, y: np.ndarray) -> float:
    dx = np.diff(x)
    dy = np.diff(y)
    finite = np.isfinite(dx) & np.isfinite(dy) & (dx > 0)
    if not finite.any():
        return float("nan")
    return float(np.max(np.abs(dy[finite] / dx[finite])))


def _total_variation(y: np.ndarray) -> float:
    finite = y[np.isfinite(y)]
    if finite.size < 2:
        return float("nan")
    return float(np.sum(np.abs(np.diff(finite))))


def _discontinuity_count(y: np.ndarray, k: float = _DISCONTINUITY_K) -> int:
    """Count second-difference outliers via robust MAD-based threshold."""
    finite_mask = np.isfinite(y)
    if finite_mask.sum() < 4:
        return 0
    y_f = y[finite_mask]
    d2 = np.diff(y_f, n=2)
    if d2.size == 0:
        return 0
    median = np.median(d2)
    mad = np.median(np.abs(d2 - median)) + 1e-12
    threshold = k * 1.4826 * mad  # 1.4826 normalises MAD to sigma for Gaussian.
    return int(np.sum(np.abs(d2 - median) > threshold))


def _spectral_roughness(y: np.ndarray) -> float:
    """High-frequency / low-frequency FFT power ratio.

    Larger values indicate jagged behaviour over smooth trends.
    """
    finite = y[np.isfinite(y)]
    if finite.size < 8:
        return float("nan")
    # Detrend mean.
    y_c = finite - finite.mean()
    spectrum = np.abs(np.fft.rfft(y_c))
    n = spectrum.size
    if n < 4:
        return float("nan")
    cutoff = max(1, n // 4)
    low = float(np.sum(spectrum[:cutoff] ** 2))
    high = float(np.sum(spectrum[cutoff:] ** 2))
    if low <= 0:
        return float("inf") if high > 0 else float("nan")
    return high / low


@dataclass(frozen=True)
class SmoothnessDiagnostic:
    """Compute and render smoothness metrics on η*(|Δ|) and CI endpoints."""

    name: ClassVar[str] = "smoothness"

    def compute(self, raw: RawResult) -> DiagnosticTable:
        delta = np.asarray(raw.arrays["abs_delta_grid"], dtype=np.float64)
        eta_star = np.asarray(raw.arrays["eta_star"], dtype=np.float64)
        ci_lo = np.asarray(raw.arrays["ci_lower"], dtype=np.float64)
        ci_hi = np.asarray(raw.arrays["ci_upper"], dtype=np.float64)
        width = ci_hi - ci_lo

        records = [
            {
                "experiment": raw.experiment,
                "tilting": raw.tilting,
                "statistic": raw.statistic,
                "metric": "lipschitz_eta",
                "value": _local_lipschitz(delta, eta_star),
            },
            {
                "experiment": raw.experiment,
                "tilting": raw.tilting,
                "statistic": raw.statistic,
                "metric": "total_variation_eta",
                "value": _total_variation(eta_star),
            },
            {
                "experiment": raw.experiment,
                "tilting": raw.tilting,
                "statistic": raw.statistic,
                "metric": "discontinuity_count_eta",
                "value": _discontinuity_count(eta_star),
            },
            {
                "experiment": raw.experiment,
                "tilting": raw.tilting,
                "statistic": raw.statistic,
                "metric": "spectral_roughness_eta",
                "value": _spectral_roughness(eta_star),
            },
            {
                "experiment": raw.experiment,
                "tilting": raw.tilting,
                "statistic": raw.statistic,
                "metric": "lipschitz_width",
                "value": _local_lipschitz(delta, width),
            },
            {
                "experiment": raw.experiment,
                "tilting": raw.tilting,
                "statistic": raw.statistic,
                "metric": "total_variation_width",
                "value": _total_variation(width),
            },
        ]
        df = pd.DataFrame.from_records(records)
        return DiagnosticTable(
            name=self.name,
            table=df,
            units={"value": "depends on metric"},
            metadata={
                "n_grid": int(delta.size),
                "abs_delta_min": float(delta.min()),
                "abs_delta_max": float(delta.max()),
                "w": raw.metadata.get("w"),
                "alpha": raw.metadata.get("alpha"),
            },
        )

    def render(self, table: DiagnosticTable, fig_dir: Path) -> Path:
        df = table.table
        if df.empty:
            raise ValueError("empty smoothness table; cannot render")
        # Bar chart per (tilting, statistic) cell, one panel per metric.
        metrics = list(df["metric"].unique())
        cells = list(df.groupby(["tilting", "statistic"], sort=False).groups)
        fig, axes = plt.subplots(2, 3, figsize=(11, 6), squeeze=False)
        for k, metric in enumerate(metrics):
            r, c = divmod(k, 3)
            ax = axes[r][c]
            sub = df[df["metric"] == metric]
            labels = [f"{t}/{s}" for t, s in cells]
            values = [
                (
                    float(sub[(sub["tilting"] == t) & (sub["statistic"] == s)]["value"].iloc[0])
                    if not sub[(sub["tilting"] == t) & (sub["statistic"] == s)].empty
                    else float("nan")
                )
                for t, s in cells
            ]
            ax.bar(labels, values, color="#2E86AB")
            ax.set_title(metric, fontsize=9)
            ax.tick_params(axis="x", labelrotation=30, labelsize=7)
            ax.tick_params(axis="y", labelsize=8)
        for k in range(len(metrics), 6):
            r, c = divmod(k, 3)
            axes[r][c].axis("off")
        fig.suptitle(
            "Smoothness metrics (lower = smoother; spikes = " "discontinuity-prone)",
            fontsize=11,
        )
        fig.tight_layout()
        fig_dir.mkdir(parents=True, exist_ok=True)
        out = fig_dir / "smoothness_metrics.png"
        fig.savefig(out, dpi=130)
        plt.close(fig)
        return out
