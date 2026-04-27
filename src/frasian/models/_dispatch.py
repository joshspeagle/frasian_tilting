"""Helpers for clear error messages when a method only supports certain models.

The framework's tilting schemes and test statistics specialise on a
specific (Model, Prior) combination — usually `(NormalNormalModel,
NormalDistribution)`. Hand-rolled `isinstance` checks proliferate; this
module collapses them into one helper so the error message is uniform
and the caller's intent is explicit.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Type, TypeVar

from .base import Model, Prior

if TYPE_CHECKING:
    pass


T = TypeVar("T")


def require_model(model: Model, expected: Type[T], *, caller: str) -> T:
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
    return model  # type: ignore[return-value]


def require_prior(prior: Prior | None, expected: Type[T], *, caller: str) -> T:
    """Assert `prior` is an instance of `expected`, else raise."""
    if prior is None:
        raise NotImplementedError(
            f"{caller} requires a {expected.__name__} prior; got None."
        )
    if not isinstance(prior, expected):
        raise NotImplementedError(
            f"{caller} currently requires a {expected.__name__} prior; "
            f"got {type(prior).__name__!r}."
        )
    return prior  # type: ignore[return-value]
