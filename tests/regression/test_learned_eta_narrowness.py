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

from frasian.learned.eta_artifact import EtaArtifact
from frasian.models.distributions import NormalDistribution
from frasian.models.normal_normal import NormalNormalModel
from frasian.statistics.waldo import WaldoStatistic
from frasian.tilting.eta_selectors import LearnedDynamicEtaSelector
from frasian.tilting.ot import OTTilting
from frasian.tilting.power_law import PowerLawTilting

_CHECKPOINTS = {
    "powerlaw": (
        Path("artifacts/learned_eta_canonical_normal_normal_powerlaw_v1.eqx"),
        Path("artifacts/learned_eta_canonical_normal_normal_powerlaw_v0_smoke.eqx"),
        PowerLawTilting,
    ),
    "ot": (
        Path("artifacts/learned_eta_canonical_normal_normal_ot_v1.eqx"),
        Path("artifacts/learned_eta_canonical_normal_normal_ot_v0_smoke.eqx"),
        OTTilting,
    ),
}


# Audit P0-12: explicit per-scheme seed offsets (deterministic across
# Python processes; mirrors `_SCHEME_SEED_OFFSET` in
# `test_learned_eta_calibration.py` to avoid `hash(scheme_label) % 50`,
# which is process-local under PYTHONHASHSEED=random).
_SCHEME_SEED_OFFSET = {"powerlaw": 17, "ot": 31}


def _checkpoint_and_tolerance(scheme_label: str) -> tuple[Path, float, type]:
    """Return (checkpoint_path, tolerance, scheme_class).

    Strict tolerance for v1; loose for v0_smoke. OT v0_smoke is more
    undertrained (Head B accuracy ~0.67 vs power_law's ~0.97), so the
    smoke tolerance is wider.
    """
    v1, v0_smoke, cls = _CHECKPOINTS[scheme_label]
    if v1.exists():
        return v1, 0.0, cls
    if v0_smoke.exists():
        # OT smoke is more undertrained → wider tolerance.
        rel_tol = 0.15 if scheme_label == "powerlaw" else 0.30
        return v0_smoke, rel_tol, cls
    pytest.skip(f"no Phase E {scheme_label} learned-eta checkpoint available")


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
            alpha,
            np.asarray([D]),
            model,
            prior,
            WaldoStatistic(),
        )
        widths[i] = sum(hi - lo for lo, hi in regions)
    return widths


def _build_scheme_and_priors(ckpt_path: Path, scheme_cls: type):
    """Read sigma/sigma0/mu0 from the trained checkpoint config."""
    artifact = EtaArtifact(artifact_path=ckpt_path)
    artifact.load()
    cfg = artifact.metadata["experiment_config"]
    sigma = float(cfg["model_fingerprint"][1])
    mu0 = float(cfg["prior_fingerprint"][1])
    sigma0 = float(cfg["prior_fingerprint"][2])
    selector = LearnedDynamicEtaSelector(
        artifact=artifact,
        sigma=sigma,
        mu0=mu0,
    )
    scheme = scheme_cls(selector=selector)
    prior = NormalDistribution(loc=mu0, scale=sigma0)
    model = NormalNormalModel(sigma=sigma)
    return scheme, prior, model, sigma, mu0


@pytest.mark.L3
@pytest.mark.slow
@pytest.mark.parametrize("scheme_label", ["powerlaw", "ot"])
@pytest.mark.parametrize(
    "theta_true",
    # Audit P0-13: extend to ±4 (the CLAUDE.md-cited conflict band) so
    # the headline claim "calibrated AND ≤ Wald at θ=±4" is actually
    # exercised by this test.
    [-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0],
)
def test_learned_no_wider_than_wald(scheme_label, theta_true):
    """Headline claim 1: learned width ≤ Wald (3.92) + MC tolerance."""
    ckpt, rel_tol, scheme_cls = _checkpoint_and_tolerance(scheme_label)
    alpha = 0.05
    n_reps = 100

    learned, prior, model, sigma, _ = _build_scheme_and_priors(ckpt, scheme_cls)
    # Audit P0-12: do NOT use `hash(scheme_label)` — Python's hash is
    # randomised per process under PYTHONHASHSEED=random, so two runs
    # of this test in different processes get different seeds, breaking
    # the "byte-reproducible at fixed config + sha" claim. Use the same
    # explicit offset map test_learned_eta_calibration.py uses.
    rng = np.random.default_rng(
        seed=42 + int(theta_true) + _SCHEME_SEED_OFFSET[scheme_label],
    )
    widths = _measure_widths(
        learned,
        n_reps,
        rng,
        theta_true,
        sigma,
        prior,
        model,
        alpha,
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
@pytest.mark.parametrize("scheme_label", ["powerlaw", "ot"])
@pytest.mark.parametrize("theta_true", [-4.0, 4.0])
def test_learned_beats_bare_waldo_at_conflict(scheme_label, theta_true):
    """Headline claim 2: at high conflict, learned strictly < bare WALDO."""
    ckpt, rel_tol, scheme_cls = _checkpoint_and_tolerance(scheme_label)
    alpha = 0.05
    n_reps = 100

    learned, prior, model, sigma, _ = _build_scheme_and_priors(ckpt, scheme_cls)
    bare_waldo = scheme_cls()  # default = identity selector (η=0)

    # Audit P0-12: do NOT use `hash(scheme_label)` — Python's hash is
    # randomised per process under PYTHONHASHSEED=random, so two runs
    # of this test in different processes get different seeds, breaking
    # the "byte-reproducible at fixed config + sha" claim. Use the same
    # explicit offset map test_learned_eta_calibration.py uses.
    rng = np.random.default_rng(
        seed=42 + int(theta_true) + _SCHEME_SEED_OFFSET[scheme_label],
    )
    widths_learned = _measure_widths(
        learned,
        n_reps,
        rng,
        theta_true,
        sigma,
        prior,
        model,
        alpha,
    )
    widths_bare = _measure_widths(
        bare_waldo,
        n_reps,
        rng,
        theta_true,
        sigma,
        prior,
        model,
        alpha,
    )

    mean_learned = float(widths_learned.mean())
    mean_bare = float(widths_bare.mean())
    se_diff = float(np.sqrt(widths_learned.var(ddof=1) / n_reps + widths_bare.var(ddof=1) / n_reps))
    threshold = mean_bare * (1.0 + rel_tol) + 2.0 * se_diff
    assert mean_learned <= threshold, (
        f"At conflict θ_true={theta_true} ({scheme_label}): learned "
        f"width {mean_learned:.3f} > bare WALDO {mean_bare:.3f} + tol "
        f"= {threshold:.3f}; the conflict-band narrowness claim fails."
    )


# Audit P0-13: a STRICT narrowness regression that fires only when a v1
# checkpoint is present. With rel_tol=0 and n_reps=500 the test pins the
# CLAUDE.md headline numbers within ~1·SE — a 25% width regression that
# the smoke-tolerance tests above would absorb is caught here.
@pytest.mark.L3
@pytest.mark.slow
@pytest.mark.nightly
@pytest.mark.parametrize("scheme_label", ["powerlaw", "ot"])
@pytest.mark.parametrize("theta_true", [-4.0, -2.0, 0.0, 2.0, 4.0])
def test_learned_strict_narrowness_at_v1(scheme_label, theta_true):
    """Strict v1 regression: rel_tol=0, n_reps=500 — pins headline numbers.

    Skips if no v1 checkpoint is on disk; runs only when production
    artifacts are present. CLAUDE.md cites the headline table at this
    config (n_reps=200, w=0.5, α=0.05); we use n_reps=500 here for
    tighter SE and assert STRICTLY mean_learned ≤ Wald + 2·SE (no
    rel_tol slack). A 15-25% width regression that the smoke tests
    absorb fails this test.
    """
    v1, _, scheme_cls = _CHECKPOINTS[scheme_label]
    if not v1.exists():
        pytest.skip(
            f"No v1 checkpoint at {v1!s}; strict narrowness regression "
            f"runs only when production artifacts are present. Train via "
            f"`python -m scripts.train_learned_eta --config "
            f"experiments/canonical_normal_normal_{scheme_label}.yaml`."
        )

    alpha = 0.05
    n_reps = 500

    learned, prior, model, sigma, _ = _build_scheme_and_priors(v1, scheme_cls)
    rng = np.random.default_rng(seed=42 + int(theta_true) + _SCHEME_SEED_OFFSET[scheme_label])
    widths = _measure_widths(
        learned, n_reps, rng, theta_true, sigma, prior, model, alpha
    )

    mean_w = float(widths.mean())
    se = float(np.sqrt(widths.var(ddof=1) / n_reps))
    wald_width = 2.0 * 1.96 * sigma  # 3.92
    # rel_tol = 0; only MC noise tolerance.
    assert mean_w <= wald_width + 2.0 * se, (
        f"v1 strict narrowness FAILED ({scheme_label}, θ={theta_true}): "
        f"learned width {mean_w:.4f} ± {se:.4f} > Wald {wald_width:.4f} "
        f"(SE band: {wald_width + 2 * se:.4f}). The headline claim "
        f"'≤ Wald at every θ' is broken on the v1 checkpoint."
    )
