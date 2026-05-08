"""Helpers for clear error messages when a method only supports certain models.

The framework's tilting schemes and test statistics specialise on a
specific (Model, Prior) combination — usually `(NormalNormalModel,
NormalDistribution)`. Hand-rolled `isinstance` checks proliferate; this
module collapses them into one helper so the error message is uniform
and the caller's intent is explicit.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar

from .base import Model, Prior

if TYPE_CHECKING:
    pass


T = TypeVar("T")


def require_model(model: Model, expected: type[T], *, caller: str) -> T:
    """Assert `model` is an instance of `expected`, else raise a clear error.

    `caller` is the name of the method that's restricting (e.g.
    "WaldoStatistic.pvalue"); used in the error message so users know
    where to look. Returns `model` cast to `expected` for type-checker
    convenience.
    """
    if not isinstance(model, expected):
        raise NotImplementedError(
            f"{caller} currently requires {expected.__name__}; "
            f"got {type(model).__name__!r}. Generalising the implementation "
            f"to other models is tracked as Phase-3 follow-up work."
        )
    return model


def require_prior(prior: Prior | None, expected: type[T], *, caller: str) -> T:
    """Assert `prior` is an instance of `expected`, else raise."""
    if prior is None:
        raise NotImplementedError(f"{caller} requires a {expected.__name__} prior; got None.")
    if not isinstance(prior, expected):
        raise NotImplementedError(
            f"{caller} currently requires a {expected.__name__} prior; "
            f"got {type(prior).__name__!r}."
        )
    return prior


def is_normal_normal(model: Model) -> bool:
    """True iff `model.fingerprint()[0] == "normal_normal"`.

    Audit P1 G.5: prefer this over `isinstance(model, NormalNormalModel)`
    so a future wrapper (or numerically-equivalent reimplementation)
    can opt into the closed-form Normal-Normal dispatch path by
    declaring its fingerprint, without inheriting from a specific
    class. The fingerprint contract is the single source of truth for
    "what kind of model is this" — the registry, cache, and learned-η
    loader all key on it.
    """
    fp = getattr(model, "fingerprint", None)
    if fp is None:
        return False
    try:
        tup = fp()
    except Exception:
        return False
    return bool(tup) and tup[0] == "normal_normal"
