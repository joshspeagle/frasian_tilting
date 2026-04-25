"""WALDO statistic — Bayesian-frequentist hybrid via the posterior mean.

The p-value formula (Theorem 3 in the legacy derivations, ported verbatim
from `legacy/src/frasian/waldo.py:115`):

  a(theta) = |mu_n - theta| / (w * sigma)
  b(theta) = (1 - w) * (mu0 - theta) / (w * sigma)
  p(theta) = Phi(b - a) + Phi(-a - b)

Specializes on `NormalNormalModel` and a `NormalDistribution` prior.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import stats

from .._registry import register_statistic
from ..models.base import Model, Prior
from ..models.distributions import NormalDistribution
from ..models.normal_normal import NormalNormalModel, posterior_params
from .base import AsymptoticDistribution


def _require_normal_normal(model: Model, prior: Prior | None
                           ) -> tuple[NormalNormalModel, NormalDistribution]:
    if not isinstance(model, NormalNormalModel):
        raise NotImplementedError(
            f"WaldoStatistic currently requires NormalNormalModel; "
            f"got {type(model).__name__!r}."
        )
    if not isinstance(prior, NormalDistribution):
        raise NotImplementedError(
            f"WaldoStatistic requires a NormalDistribution prior; "
            f"got {type(prior).__name__!r}."
        )
    return model, prior


def _pvalue_components(theta: NDArray[np.float64], mu_n: float, mu0: float,
                       w: float, sigma: float
                       ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """a(theta), b(theta) from the WALDO p-value formula."""
    a = np.abs(mu_n - theta) / (w * sigma)
    b = (1.0 - w) * (mu0 - theta) / (w * sigma)
    return a, b


@register_statistic(name="waldo", brief="docs/methods/waldo.md")
@dataclass(frozen=True)
class WaldoStatistic:
    """WALDO statistic for the Normal-Normal model."""

    name: str = "waldo"
    asymptotic_null: AsymptoticDistribution = AsymptoticDistribution(
        family="weighted_chi2", df=1, scale=1.0,
        description="(mu_n - theta)^2 / sigma_n^2 ~ w * ncx2_1(lambda(theta)).",
    )

    def evaluate(self, theta0: ArrayLike, data: NDArray[np.float64],
                 model: Model, prior: Prior | None = None
                 ) -> NDArray[np.float64]:
        m, pi = _require_normal_normal(model, prior)
        D = float(np.atleast_1d(np.asarray(data, dtype=np.float64)).mean())
        mu_n, sigma_n, _ = posterior_params(D, pi.loc, m.sigma, pi.scale)
        diff = np.asarray(mu_n - np.asarray(theta0, dtype=np.float64),
                          dtype=np.float64)
        return np.asarray(diff * diff / (sigma_n ** 2), dtype=np.float64)

    def pvalue(self, theta0: ArrayLike, data: NDArray[np.float64],
               model: Model, prior: Prior | None = None
               ) -> NDArray[np.float64]:
        m, pi = _require_normal_normal(model, prior)
        D = float(np.atleast_1d(np.asarray(data, dtype=np.float64)).mean())
        mu_n, _, w = posterior_params(D, pi.loc, m.sigma, pi.scale)
        theta_arr = np.asarray(theta0, dtype=np.float64)
        a, b = _pvalue_components(theta_arr, float(mu_n), pi.loc, w, m.sigma)
        return np.asarray(stats.norm.cdf(b - a) + stats.norm.cdf(-a - b),
                          dtype=np.float64)

    def acceptance_region(self, alpha: float, theta0: ArrayLike,
                          model: Model, prior: Prior | None = None
                          ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Acceptance region in `D`-space at fixed theta0.

        Solves p(theta0; D) = alpha for D — the dual of the CI construction
        used by the legacy `confidence_interval`.
        """
        # The dual problem (acceptance region in D-space) is non-trivial in
        # closed form because mu_n depends on D. Step 4's CoverageExperiment
        # constructs this region numerically via the CI inversion. Until then,
        # we expose the contract but defer the implementation.
        raise NotImplementedError(
            "WaldoStatistic.acceptance_region: numerical inversion lands with "
            "the CoverageExperiment in Step 4."
        )
