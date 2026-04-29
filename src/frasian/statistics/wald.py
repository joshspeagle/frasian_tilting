"""Wald test statistic — pure-likelihood, prior-ignoring CI.

  tau_Wald(theta) = ((D - theta) / sigma)^2
  CI:           D ± z_{1-alpha/2} * sigma

Specializes on `NormalNormalModel` (port from legacy `waldo.py:wald_ci`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import stats

from .._registry import register_statistic
from ..models._dispatch import require_model
from ..models.base import Model, Prior
from ..models.normal_normal import NormalNormalModel
from .base import AsymptoticDistribution


def _require_normal_normal(model: Model) -> NormalNormalModel:
    return require_model(model, NormalNormalModel, caller="WaldStatistic")


@register_statistic(name="wald", brief="docs/methods/wald.md")
@dataclass(frozen=True)
class WaldStatistic:
    """Wald statistic for the Normal location family."""

    name: str = "wald"
    asymptotic_null: AsymptoticDistribution = AsymptoticDistribution(
        family="chi2", df=1, scale=1.0,
        description="(D - theta)^2 / sigma^2 ~ chi^2_1 under H0.",
    )

    def evaluate(self, theta0: ArrayLike, data: NDArray[np.float64],
                 model: Model, prior: Prior | None = None
                 ) -> NDArray[np.float64]:
        m = _require_normal_normal(model)
        D = float(np.atleast_1d(np.asarray(data, dtype=np.float64)).mean())
        z = (D - np.asarray(theta0, dtype=np.float64)) / m.sigma
        return np.asarray(z * z, dtype=np.float64)

    def pvalue(self, theta0: ArrayLike, data: NDArray[np.float64],
               model: Model, prior: Prior | None = None
               ) -> NDArray[np.float64]:
        m = _require_normal_normal(model)
        D = float(np.atleast_1d(np.asarray(data, dtype=np.float64)).mean())
        z = np.abs(D - np.asarray(theta0, dtype=np.float64)) / m.sigma
        # Two-sided p-value: 2 * (1 - Phi(|z|))
        return np.asarray(2.0 * stats.norm.sf(z), dtype=np.float64)

    def acceptance_region(self, alpha: float, theta0: ArrayLike,
                          model: Model, prior: Prior | None = None
                          ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return (D_lo, D_hi) such that Wald accepts H0 iff D in [D_lo, D_hi]."""
        m = _require_normal_normal(model)
        z_crit = stats.norm.ppf(1.0 - alpha / 2.0)
        theta_arr = np.asarray(theta0, dtype=np.float64)
        return (
            np.asarray(theta_arr - z_crit * m.sigma, dtype=np.float64),
            np.asarray(theta_arr + z_crit * m.sigma, dtype=np.float64),
        )

    def confidence_interval(self, alpha: float, data: NDArray[np.float64],
                            model: Model, prior: Prior | None = None
                            ) -> tuple[float, float]:
        """Closed-form Wald CI: D ± z_{1-alpha/2} * sigma."""
        m = _require_normal_normal(model)
        D = float(np.atleast_1d(np.asarray(data, dtype=np.float64)).mean())
        z = stats.norm.ppf(1.0 - alpha / 2.0)
        half = z * m.sigma
        return (D - half, D + half)
