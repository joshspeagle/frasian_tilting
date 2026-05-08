"""Parallel-map helper used by experiment per-replicate inner loops.

Provides a single function `parallel_map(fn, items, n_jobs)` that:

  * `n_jobs == 1` (default): plain list comprehension, zero extra
    dependencies imported, byte-identical results to pre-parallelism.
  * `n_jobs > 1` or `n_jobs == -1`: dispatches via `joblib.Parallel`
    with the loky (process-based) backend. Process workers keep JAX
    state per-worker so per-replicate JAX traces don't collide; the
    cost is a first-call import + JAX-warmup of ~1-2 s per worker,
    amortised across the per-cell `n_reps` work.

`fn` must be picklable: top-level functions or `functools.partial`
instances bind cleanly. Lambdas and nested closures DO NOT pickle and
will raise — we deliberately don't smooth that over because the right
fix is a top-level function with explicit arguments.

`items` is an iterable; results returned in input order. `joblib`'s
default `batch_size='auto'` adaptively chunks dispatches to amortise
per-call pickling cost — important when `fn` is fast (~ms) since
otherwise dispatch overhead dominates.

Reproducibility: `n_jobs` does NOT enter `Config.fingerprint()`. The
per-replicate `D` values come from a pre-generated raw stream (see
`simulation.raw.generate_normal_D_samples`) so parallel dispatch
reorders only the *evaluation*, not the *seeding*. Numerical result
is byte-identical at any `n_jobs >= 1` — pinned by
`tests/regression/test_parallelism_bitwise.py`.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import TypeVar

T = TypeVar("T")
R = TypeVar("R")


def parallel_map(
    fn: Callable[[T], R],
    items: Iterable[T],
    *,
    n_jobs: int = 1,
) -> list[R]:
    """Map `fn` over `items`, optionally in parallel.

    Parameters
    ----------
    fn
        Picklable callable. Top-level functions or `functools.partial`
        instances bind cleanly. Closures over module-level state (e.g.
        a partial that has captured a frozen-dataclass model + prior)
        also pickle fine.
    items
        Iterable of arguments. `fn(item)` is called once per element.
    n_jobs
        `1` (default) → serial list comprehension. `n > 1` → that many
        worker processes via joblib loky. `-1` → all available cores.
        `0` is treated as `1`.

    Returns
    -------
    list[R]
        Results in input order.
    """
    items_list = list(items)
    if n_jobs in (0, 1):
        return [fn(x) for x in items_list]
    # Lazy import: keeps `joblib` an optional runtime dep for callers
    # that never set `n_jobs > 1`.
    from joblib import Parallel, delayed

    return list(
        Parallel(n_jobs=n_jobs, prefer="processes", batch_size="auto")(
            delayed(fn)(x) for x in items_list
        )
    )
