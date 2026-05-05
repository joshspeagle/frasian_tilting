"""Regression: Phase 4 skeptic §1 — antithetic flag is restricted to ``static_width``.

The antithetic ``2θ − D`` pairing reduces variance only on losses with
odd Taylor structure in ``D − θ``. ``integrated_pvalue_loss`` and
``cd_variance_loss`` are even in ``D − θ`` over a θ-symmetric grid,
so paired and IID estimators are algebraically identical at the same
batch size. Phase 4 §1 restricts the user-facing flag to
``loss_kind == "static_width"`` and warns when set with a
non-applicable loss kind.

We pin two properties:

1. ``fit_eta_artifact(loss_kind="integrated_p", antithetic=True)`` emits
   a ``UserWarning`` with explanatory text and proceeds with
   ``effective_antithetic=False``.
2. ``fit_eta_artifact(loss_kind="static_width", antithetic=True)`` emits
   no warning and the data sampler reaches ``compose_width_loss``
   with a ``(2N,)`` D batch (verified via the saved checkpoint's
   ``antithetic`` metadata key).

Both tests are torch-gated; the audit env without torch skips them.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from frasian.learned.training.sampling import (  # noqa: E402
    ExperimentConfig,
    UniformThetaDistribution,
)
from frasian.learned.training.train import fit_eta_artifact  # noqa: E402
from frasian.models.distributions import NormalDistribution  # noqa: E402
from frasian.models.normal_normal import NormalNormalModel  # noqa: E402


def _tiny_config() -> ExperimentConfig:
    return ExperimentConfig(
        scheme_name="power_law",
        statistic_name="waldo",
        prior=NormalDistribution(loc=0.0, scale=1.0),
        model=NormalNormalModel(sigma=1.0),
        theta_distribution=UniformThetaDistribution(low=-3.0, high=3.0),
        n_grid=33,
        n_lhs=64,
        eta_explore_box=(-2.0, 2.0),
        seed=0,
    )


@pytest.mark.L4
@pytest.mark.slow
def test_antithetic_with_integrated_p_warns_and_falls_back(
    tmp_path: Path, bootstrapped_registry: object
) -> None:
    """``antithetic=True`` with ``loss_kind="integrated_p"`` warns + falls back."""
    out_path = tmp_path / "ckpt.pt"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = fit_eta_artifact(
            config=_tiny_config(),
            out_path=out_path,
            loss_kind="integrated_p",
            n_epochs=2,
            batch_size=16,
            n_aux=16,
            patience=2,
            antithetic=True,
            verbose=False,
        )
    msgs = [str(w.message) for w in caught if issubclass(w.category, UserWarning)]
    assert any("antithetic=True is a no-op" in m for m in msgs), (
        f"expected antithetic-no-op UserWarning; got {msgs!r}"
    )
    # Saved checkpoint records the *effective* antithetic value, which
    # must be False once the gating warning fires.
    assert result.metadata["antithetic"] is False


@pytest.mark.L4
@pytest.mark.slow
def test_antithetic_with_static_width_no_warning(
    tmp_path: Path, bootstrapped_registry: object
) -> None:
    """``antithetic=True`` with ``loss_kind="static_width"`` is silent +
    is recorded as ``True`` in the saved checkpoint."""
    out_path = tmp_path / "ckpt_static.pt"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = fit_eta_artifact(
            config=_tiny_config(),
            out_path=out_path,
            loss_kind="static_width",
            alpha=0.05,
            n_epochs=2,
            batch_size=16,
            n_aux=16,
            patience=2,
            antithetic=True,
            verbose=False,
        )
    no_op_msgs = [
        str(w.message)
        for w in caught
        if issubclass(w.category, UserWarning) and "antithetic=True is a no-op" in str(w.message)
    ]
    assert not no_op_msgs, f"unexpected no-op warning: {no_op_msgs!r}"
    assert result.metadata["antithetic"] is True
