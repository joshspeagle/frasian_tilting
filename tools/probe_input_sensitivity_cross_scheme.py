"""Cross-scheme input-sensitivity probe for Phase G v4 fixtures.

Diagnostic tool for Stage C.5 of the Fisher-Rao PR. The framework's CLAUDE.md
row 13b documents a "near-constant per-cell" / "input-insensitivity" finding
attributed to the EtaNet/ValidityNet architecture. The Fisher-Rao
characterization in `docs/notes/2026-05-11-fisher-rao-cd-var-hyperparams.md`
hypothesised that this is in fact **loss-specific to integrated_p**, not
architecture-wide.

This probe loads each of the 12 Phase G v4 fixtures (4 schemes x 3 losses)
on disk and evaluates per-cell std + cross-cell spread + η range over a
representative grid. The same probe configuration is used as in
`tests/regression/test_fisher_rao_v4_fixture.py` so numbers are directly
comparable to that test's reference values.

Run as `python tools/probe_input_sensitivity_cross_scheme.py`. Saves the
formatted summary table to /tmp/probe_input_sensitivity_output.txt.
"""

from __future__ import annotations

# JAX x64 MUST be enabled before any jax.numpy import.
from jax import config as _jax_config
_jax_config.update("jax_enable_x64", True)

from pathlib import Path

import numpy as np

from frasian._registry_bootstrap import bootstrap

bootstrap()

from frasian.learned.eta_artifact import EtaArtifact
from frasian.models.distributions import NormalDistribution
from frasian.models.normal_normal import NormalNormalModel


_ARTIFACT_DIR = Path("artifacts")
_OUTPUT_PATH = Path("/tmp/probe_input_sensitivity_output.txt")

# (scheme_name_for_display, filename_scheme_token)
_SCHEMES = [
    ("power_law", "powerlaw"),
    ("ot", "ot"),
    ("mixture", "mixture"),
    ("fisher_rao", "fisher_rao"),
]
_LOSSES = ("integrated_p", "cd_variance", "static_width")

# Three hyperparam cells matching test_fisher_rao_v4_fixture.py.
_CELLS = [
    (0.0, 0.5, 1.0),
    (1.0, 2.0, 1.5),
    (-1.0, 1.0, 0.5),
]

_THETA_GRID = np.linspace(-3.0, 3.0, 51)


def _artifact_path(scheme_token: str, loss: str) -> Path:
    return _ARTIFACT_DIR / (
        f"learned_eta_canonical_normal_normal_{scheme_token}_phaseC_{loss}_v4.eqx"
    )


def probe_fixture(path: Path) -> dict | None:
    """Load fixture and compute per-cell + cross-cell sensitivity stats.

    Returns dict with keys:
      - per_cell_std:      list of std(eta over theta) per cell
      - per_cell_mean:     list of mean(eta over theta) per cell
      - cross_cell_spread: max(per_cell_mean) - min(per_cell_mean)
      - eta_min, eta_max:  global range across all cells x theta
    Returns None if the fixture cannot be loaded.
    """
    if not path.exists():
        return None
    try:
        art = EtaArtifact(artifact_path=path)
        art.load()
    except Exception as exc:  # pragma: no cover - diagnostic
        return {"error": f"load failed: {exc}"}

    per_cell_std: list[float] = []
    per_cell_mean: list[float] = []
    all_eta: list[np.ndarray] = []

    for mu0, sigma0, sigma in _CELLS:
        prior = NormalDistribution(loc=mu0, scale=sigma0)
        model = NormalNormalModel(sigma=sigma)
        try:
            eta = art.predict_eta(
                _THETA_GRID, prior.hyperparams(), model.hyperparams()
            )
        except Exception as exc:  # pragma: no cover - diagnostic
            return {"error": f"predict_eta failed at cell ({mu0},{sigma0},{sigma}): {exc}"}
        per_cell_std.append(float(np.std(eta)))
        per_cell_mean.append(float(np.mean(eta)))
        all_eta.append(np.asarray(eta))

    flat = np.concatenate(all_eta)
    return {
        "per_cell_std": per_cell_std,
        "per_cell_mean": per_cell_mean,
        "cross_cell_spread": float(max(per_cell_mean) - min(per_cell_mean)),
        "eta_min": float(np.min(flat)),
        "eta_max": float(np.max(flat)),
    }


def fmt_row(scheme: str, loss: str, stats: dict | None) -> str:
    """Format one fixture's stats into a single tabular row."""
    if stats is None:
        return f"{scheme:>10} {loss:>14} | MISSING"
    if "error" in stats:
        return f"{scheme:>10} {loss:>14} | ERROR: {stats['error']}"

    std0, std1, std2 = stats["per_cell_std"]
    mean0, mean1, mean2 = stats["per_cell_mean"]
    spread = stats["cross_cell_spread"]
    lo = stats["eta_min"]
    hi = stats["eta_max"]
    # Use scientific for std (dynamic range across losses ~5e-4 to ~5e-1).
    return (
        f"{scheme:>10} {loss:>14} | "
        f"std=({std0:>9.2e}, {std1:>9.2e}, {std2:>9.2e})  "
        f"mean=({mean0:>+7.3f}, {mean1:>+7.3f}, {mean2:>+7.3f})  "
        f"spread={spread:>6.3f}  "
        f"range=[{lo:>+7.3f}, {hi:>+7.3f}]"
    )


def main() -> None:
    lines: list[str] = []

    def emit(s: str = "") -> None:
        print(s)
        lines.append(s)

    emit("Cross-scheme input-sensitivity probe (Stage C.5)")
    emit("=" * 116)
    emit(f"Theta grid: 51 points in [-3, 3]")
    emit(f"Cells: (mu0, sigma0, sigma) in {_CELLS}")
    emit(f"Per-cell std = std(eta over theta), one value per cell.")
    emit(f"Per-cell mean = mean(eta over theta), one value per cell.")
    emit(f"Cross-cell spread = max(mean) - min(mean) across cells.")
    emit(f"Range = [min, max] of eta across all cells x theta.")
    emit("")

    header = (
        f"{'scheme':>10} {'loss':>14} | "
        f"{'per-cell std (cell0, cell1, cell2)':>43}  "
        f"{'per-cell mean (cell0, cell1, cell2)':>33}  "
        f"{'spread':>13}  {'range':>20}"
    )
    emit(header)
    emit("-" * 116)

    all_results: dict[tuple[str, str], dict | None] = {}
    for scheme_display, scheme_token in _SCHEMES:
        for loss in _LOSSES:
            path = _artifact_path(scheme_token, loss)
            stats = probe_fixture(path)
            all_results[(scheme_display, loss)] = stats
            emit(fmt_row(scheme_display, loss, stats))
        emit("-" * 116)

    # Loss-specificity summary: aggregate per-cell std by loss, across schemes.
    emit("")
    emit("Loss-specificity summary (median per-cell std across schemes x cells):")
    for loss in _LOSSES:
        stds: list[float] = []
        for scheme_display, _ in _SCHEMES:
            stats = all_results.get((scheme_display, loss))
            if stats and "per_cell_std" in stats:
                stds.extend(stats["per_cell_std"])
        if stds:
            arr = np.asarray(stds)
            emit(
                f"  {loss:>14}: median={float(np.median(arr)):.3e}  "
                f"min={float(np.min(arr)):.3e}  max={float(np.max(arr)):.3e}  "
                f"(n={len(stds)} cells)"
            )
        else:
            emit(f"  {loss:>14}: no fixtures loaded")

    emit("")
    emit("Cross-cell spread summary (per loss, across schemes):")
    for loss in _LOSSES:
        spreads: list[float] = []
        for scheme_display, _ in _SCHEMES:
            stats = all_results.get((scheme_display, loss))
            if stats and "cross_cell_spread" in stats:
                spreads.append(stats["cross_cell_spread"])
        if spreads:
            arr = np.asarray(spreads)
            emit(
                f"  {loss:>14}: median={float(np.median(arr)):.3f}  "
                f"min={float(np.min(arr)):.3f}  max={float(np.max(arr)):.3f}  "
                f"(n={len(spreads)} schemes)"
            )

    emit("")
    emit(f"Output saved to {_OUTPUT_PATH}")
    _OUTPUT_PATH.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
