"""L2 regression: cross-scheme smoothness orderings (Stage D).

Pins a small number of QUALITATIVE orderings on the metrics computed by
``scripts.compare_geodesic_smoothness`` — NOT bit-equal numbers, which
drift with v4 retrains and JAX MC seeds. The orderings were verified
against the headline run committed in ``output/diagnostics/
compare_smoothness_<sha>.csv`` (Stage D.1).

To keep the test runtime under 60s, the grids are coarsened to
``n_theta=60, n_D=60`` (vs. the headline's 200/200) and only the
relevant subset of cells is evaluated. The orderings tested here have
been verified to hold at both coarse and headline-fidelity grids.

Skips gracefully if any required Phase G v4 fixture or DynamicNumerical
disk-cache file is missing — the orderings depend on real artifacts +
caches being present.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

# Reuse the headline script's machinery so the test computes the SAME
# numbers (no parallel re-implementation that could drift).
from scripts.compare_geodesic_smoothness import (
    _build_tilting,
    _compute_metrics_row,
    _learned_fixture_path,
    _per_cell_eta_curve,
    _per_cell_width_curve,
)


# ---------------------------------------------------------------------------
# Skip guards: the fixtures + caches required for the orderings.
# ---------------------------------------------------------------------------

_REQUIRED_LEARNED = [
    ("power_law", "learned_intp"),
    ("ot", "learned_intp"),
    ("mixture", "learned_intp"),
    ("fisher_rao", "learned_intp"),
    ("fisher_rao", "learned_cd_var"),
]


def _missing_learned() -> list[str]:
    return [
        f"{s}/{sel}"
        for s, sel in _REQUIRED_LEARNED
        if not _learned_fixture_path(s, sel).exists()
    ]


_REPO_ROOT = Path(__file__).resolve().parents[2]
_ETA_LOOKUP_DIR = _REPO_ROOT / "artifacts" / "eta_lookups"


def _disk_cache_present() -> bool:
    """At least 4 (one per scheme) production dyn_numerical cache files."""
    if not _ETA_LOOKUP_DIR.exists():
        return False
    files = list(_ETA_LOOKUP_DIR.glob("dyn_numerical_*.npz"))
    # Filter out tmp_ atomic-write leftovers.
    real = [p for p in files if not p.name.startswith("dyn_numerical_tmp")]
    return len(real) >= 4


_pytestmark_skip = pytest.mark.skipif(
    not _disk_cache_present() or _missing_learned(),
    reason=(
        "Stage D smoothness comparison requires the 4 dyn_numerical disk "
        "caches AND the 5 learned fixtures listed in _REQUIRED_LEARNED. "
        "Run `PYTHONHASHSEED=0 python -m scripts.regen_headline` to populate "
        "the disk caches and the Phase G v4 training pipeline to produce "
        "the learned fixtures."
    ),
)


# ---------------------------------------------------------------------------
# One-shot fixture: compute the smoothness metrics matrix for the
# minimal set of cells needed by the orderings below. The expensive bit
# is per-D CI-width sweeps; we use n_D=60 to keep this < 60s.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def smoothness_metrics():
    """Returns dict[(scheme, selector, metric_target)] -> {metric_name: value}.

    Only computes the cells exercised by the test orderings.
    """
    pytest.importorskip("jax")
    pytest.importorskip("equinox")

    from frasian._registry_bootstrap import bootstrap
    from frasian.models.distributions import NormalDistribution
    from frasian.models.normal_normal import NormalNormalModel
    from frasian.statistics.waldo import WaldoStatistic

    bootstrap()

    cells_needed = [
        # CI-width orderings: dyn_numerical for all 4 schemes
        ("power_law", "dyn_numerical"),
        ("ot", "dyn_numerical"),
        ("mixture", "dyn_numerical"),
        ("fisher_rao", "dyn_numerical"),
        # learned_intp η-curve orderings + the FR cd_var pathology check
        ("power_law", "learned_intp"),
        ("fisher_rao", "learned_cd_var"),
    ]

    sigma, sigma0, mu0, alpha = 1.0, 1.0, 0.0, 0.05
    model = NormalNormalModel(sigma=sigma)
    prior = NormalDistribution(loc=mu0, scale=sigma0)
    waldo = WaldoStatistic()

    n_theta, n_D = 60, 60
    theta_grid = np.linspace(-6.0, 6.0, n_theta)
    D_grid = np.linspace(-6.0, 6.0, n_D)

    out: dict[tuple[str, str, str], dict[str, float]] = {}
    for scheme, selector in cells_needed:
        tilting, label = _build_tilting(scheme, selector)
        if tilting is None:
            pytest.skip(f"required cell {label} missing")
        # Prime cache.
        try:
            tilting.confidence_interval(alpha, np.array([0.0]), model, prior, waldo)
        except Exception:
            pass

        eta_curve = _per_cell_eta_curve(
            tilting=tilting, selector_name=label, theta_grid=theta_grid,
            model=model, prior=prior, alpha=alpha, statistic=waldo,
        )
        width_curve = _per_cell_width_curve(
            tilting=tilting, label=label, D_grid=D_grid,
            model=model, prior=prior, alpha=alpha, statistic=waldo,
            n_jobs=1,
        )
        eta_row = _compute_metrics_row(
            scheme=scheme, selector=selector, metric_target="eta",
            x=theta_grid, y=eta_curve,
        )
        width_row = _compute_metrics_row(
            scheme=scheme, selector=selector, metric_target="ci_width",
            x=D_grid, y=width_curve,
        )
        out[(scheme, selector, "eta")] = eta_row
        out[(scheme, selector, "ci_width")] = width_row

    return out


# ---------------------------------------------------------------------------
# Ordering tests. All marked L2 / regression. Each pins a qualitative
# property; tolerances absorb MC drift but reject a sign flip / collapse.
# ---------------------------------------------------------------------------


@_pytestmark_skip
@pytest.mark.L2
@pytest.mark.regression
class TestSmoothnessOrderings:
    """Stage D headline orderings on the (4 schemes × 4 selectors) matrix."""

    # The framework's central hypothesis: alternative geodesics produce
    # smoother CI-width(D) families than the e-geodesic (power_law).
    # Verified at both coarse (n=60) and headline (n=200) grids:
    #   FR : Lipschitz 0.67, TV 5.81
    #   MX : Lipschitz 0.84, TV 7.07
    #   OT : Lipschitz 1.46, TV 8.07
    #   PL : Lipschitz 4.80, TV 9.50
    # FR is dramatically smoother than PL by Lipschitz (~7×); MX and OT
    # are both clearly smoother than PL.
    def test_fr_ci_width_lipschitz_smoother_than_pl(self, smoothness_metrics):
        """FR's CI-width(D) Lipschitz ≤ PL's at dyn_numerical
        (central hypothesis: Riemannian geodesic smoother than e-geodesic)."""
        fr = smoothness_metrics[("fisher_rao", "dyn_numerical", "ci_width")]
        pl = smoothness_metrics[("power_law", "dyn_numerical", "ci_width")]
        assert fr["lipschitz"] < pl["lipschitz"], (
            f"expected FR CI-width Lipschitz < PL's at dyn_numerical; "
            f"got FR={fr['lipschitz']:.3f}, PL={pl['lipschitz']:.3f}. "
            f"This sign flip would refute the Stage D headline finding."
        )

    def test_mx_and_ot_ci_width_tv_smoother_than_pl(self, smoothness_metrics):
        """Both MX and OT have CI-width(D) total variation ≤ PL's at
        dyn_numerical. Mixture (m-geodesic) and OT (W2 geodesic) are
        both expected to oscillate less than PL through the conflict band."""
        pl_tv = smoothness_metrics[("power_law", "dyn_numerical", "ci_width")]["tv"]
        mx_tv = smoothness_metrics[("mixture", "dyn_numerical", "ci_width")]["tv"]
        ot_tv = smoothness_metrics[("ot", "dyn_numerical", "ci_width")]["tv"]
        assert mx_tv < pl_tv, (
            f"expected MX CI-width TV < PL's; got MX={mx_tv:.3f}, PL={pl_tv:.3f}"
        )
        assert ot_tv < pl_tv, (
            f"expected OT CI-width TV < PL's; got OT={ot_tv:.3f}, PL={pl_tv:.3f}"
        )

    # FR with dyn_numerical collapses to bare WALDO (per-θ static optimum
    # is η=0, the no-tilt point). Verify the η-curve is exactly constant
    # for FR — this is a documented degeneracy, not a smoothness win.
    def test_fr_dyn_numerical_eta_curve_is_constant(self, smoothness_metrics):
        """At w=0.5, FR's per-θ static optimum η is η=0 (the Riemannian
        no-tilt point), so the dyn_numerical η-curve is exactly constant.
        This produces TV=0 / discontinuity_count=0 — a documented
        degeneracy that flags the cell as not actually exercising the
        Fisher-Rao geometry. Pinned so a regression that breaks the
        η=0 collapse (e.g. accidentally introducing a non-trivial
        admissibility shift) is caught."""
        fr_eta = smoothness_metrics[("fisher_rao", "dyn_numerical", "eta")]
        assert fr_eta["tv"] == pytest.approx(0.0, abs=1e-10), (
            f"FR η-curve at dyn_numerical should be flat (η ≡ 0); "
            f"got TV={fr_eta['tv']:.6g}"
        )
        assert fr_eta["discontinuity_count"] == 0, (
            f"FR η-curve at dyn_numerical should have 0 discontinuities; "
            f"got {fr_eta['discontinuity_count']}"
        )

    # FR cd_var is known pathological: per Stage C.4 note
    # (`2026-05-11-fisher-rao-cd-var-hyperparams.md`) and the headline
    # run, FR cd_var produces η reaching ~-2.0 with a 1.43 cross-cell
    # spread, leading to inflated CI widths. The CI-width Lipschitz at
    # FR learned_cd_var is ~94 vs ~4 at PL learned_intp — a ~20x ratio.
    # Pin the qualitative gap so a future training fix that closes it
    # is detected.
    def test_fr_learned_cd_var_ci_width_pathology(self, smoothness_metrics):
        """FR learned_cd_var produces a much rougher CI-width(D) than
        PL learned_intp — by design, the cd_var loss on FR drives η to
        large negative values and inflates the CI. Pinned so a future
        FR cd_var training fix (sharper hyperparams or boundary-aware
        loss) is observable as this gap closing."""
        fr_cd = smoothness_metrics[("fisher_rao", "learned_cd_var", "ci_width")]
        pl_intp = smoothness_metrics[("power_law", "learned_intp", "ci_width")]
        # Pin a 5x ratio (much weaker than the observed ~20x); tolerates
        # MC / retrain drift but flags a fundamental change in the
        # FR cd_var's pathology.
        assert fr_cd["lipschitz"] > 5.0 * pl_intp["lipschitz"], (
            f"expected FR learned_cd_var CI-width Lipschitz to dominate "
            f"PL learned_intp (Stage C.4 pathology); got "
            f"FR_cd={fr_cd['lipschitz']:.3f}, PL_intp={pl_intp['lipschitz']:.3f}"
        )
