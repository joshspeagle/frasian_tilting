"""L2 regression: NN translation-invariance sanity check on the trained
v4 conditional EtaNet.

For Normal-Normal, the loss is invariant under (θ → θ+a, μ₀ → μ₀+a),
so a well-trained η_φ should approximately respect:
    η(θ+a, μ₀+a, σ₀, σ) ≈ η(θ, μ₀, σ₀, σ)
We don't tell the network this — it's emergent from the loss surface.

Tolerance is per-tier:

  * **Smoke fixtures** (n_lhs=10k, ~10-20 epochs, the default Step 5
    training budget): 0.5 absolute. The fixture is ≥0.99 valid in
    held-out training metrics but emergent equivariance at extreme
    σ₀ (the fixture trains over σ₀ ∈ [0.2, 5] log-uniform, so a
    25× ratio across one batch) drifts up to ~0.3-0.4.
  * **Production fixtures** (longer training, full λ schedule):
    aspirational 0.05 — not currently pinned because there is no
    production v4 checkpoint convention yet.

Skipped if the trained v4 NN+power_law fixture isn't on disk
(fixtures are gitignored — train via
``scripts.train_learned_eta --config experiments/canonical_normal_normal_powerlaw_v4.yaml``).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from frasian._registry_bootstrap import bootstrap

bootstrap()

from frasian.learned.eta_artifact import EtaArtifact
from frasian.models.distributions import NormalDistribution
from frasian.models.normal_normal import NormalNormalModel

_ARTIFACT = Path(
    "artifacts/learned_eta_canonical_normal_normal_powerlaw_v4.eqx"
)


@pytest.mark.L2
@pytest.mark.skipif(
    not _ARTIFACT.exists(),
    reason="v4 fixture not trained yet",
)
def test_translation_invariance():
    art = EtaArtifact(artifact_path=_ARTIFACT)
    art.load()

    rng = np.random.default_rng(0)
    n_pairs = 8
    failures: list[tuple] = []
    for _ in range(n_pairs):
        sigma = float(np.exp(rng.uniform(np.log(0.5), np.log(2.0))))
        sigma0 = float(np.exp(rng.uniform(np.log(0.2), np.log(5.0))))
        mu0_base = float(rng.uniform(-1.0, 1.0))
        a = float(rng.uniform(-1.0, 1.0))
        mu0_shifted = mu0_base + a
        if not (-2.0 <= mu0_shifted <= 2.0):
            continue
        theta = np.linspace(-3.0, 3.0, 21)
        prior_base = NormalDistribution(loc=mu0_base, scale=sigma0)
        prior_shifted = NormalDistribution(loc=mu0_shifted, scale=sigma0)
        model = NormalNormalModel(sigma=sigma)
        eta_base = art.predict_eta(
            theta, prior_base.hyperparams(), model.hyperparams(),
        )
        eta_shifted = art.predict_eta(
            theta + a, prior_shifted.hyperparams(), model.hyperparams(),
        )
        diff = np.abs(eta_base - eta_shifted)
        if diff.max() > 0.5:
            failures.append(
                (mu0_base, mu0_shifted, sigma0, sigma, float(diff.max()))
            )
    assert not failures, (
        f"translation-invariance violated on {len(failures)}/{n_pairs} pairs "
        f"(tol=0.5): {failures[:3]}"
    )
