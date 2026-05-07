# JAX Style Guide

This guide is the source of truth for JAX conventions across
`src/frasian/`. Every porting agent reads this before touching code,
and every PR reviewer checks against it.

## Setup

- **Always** rely on `frasian._jax_setup` being imported. It enables
  `jax_enable_x64` so all `jnp` arrays default to `float64` and
  match the existing `atol ≤ 1e-12` baselines. Submodules that import
  JAX directly should keep `from frasian import _jax_setup as _x64`
  near the top (pull the side effect explicitly so static-analysis
  doesn't strip the import).

## Imports

```python
import jax
import jax.numpy as jnp
import jax.scipy.special as jsp_special
import jax.scipy.stats as jsp_stats
```

- Never alias `jax.numpy` as `np`. The codebase still has `numpy`
  imports in I/O / persistence layers; mixing the two under `np`
  invites bugs.
- `numpy` itself stays as `np` and is used wherever an array crosses
  a non-JAX boundary: file I/O (`.npz` baselines), `pandas`, `h5py`,
  matplotlib draw arguments. Convert with `np.asarray(jnp_array)` at
  the boundary; convert back with `jnp.asarray(np_array)`.

## Array creation

- Prefer `jnp.asarray(x, dtype=jnp.float64)` to `jnp.array(x)` when
  the input might already be a JAX array — `asarray` is a no-op in
  that case and avoids an extra copy.
- Default float dtype is `jnp.float64` (set by `_jax_setup`). Do not
  hardcode `float32` anywhere; if you need a specific bit width,
  spell it out so a future review can flag it.

## Mutation

JAX arrays are immutable. Replace numpy in-place idioms:

| numpy | JAX |
|-------|-----|
| `a[i] = x` | `a = a.at[i].set(x)` |
| `a[i] += x` | `a = a.at[i].add(x)` |
| `a[mask] = x` | `a = jnp.where(mask, x, a)` |
| `a[i:j] = b` | `a = a.at[i:j].set(b)` |
| `np.put(a, idx, vals)` | `a = a.at[idx].set(vals)` |

If you find yourself reaching for `a.at[...]` more than twice in a
function, that's a sign the algorithm should be reformulated as a
single `jnp.where` / `jnp.select` / `jax.lax.scan`.

## Random numbers

JAX RNG is explicit and stateless. Replace numpy's mutable
`np.random.RandomState`:

| numpy | JAX |
|-------|-----|
| `rng = np.random.default_rng(seed)` | `key = jax.random.PRNGKey(seed)` |
| `rng.normal(size=n)` | `jax.random.normal(key, (n,))` |
| Reuse `rng` | `key, sub = jax.random.split(key); jax.random.normal(sub, ...)` |

Functions that consume randomness take a `key: jax.Array` argument.
Never reuse a key — always split.

For framework code that interacts with the existing
`Config.rng_seed` / numpy RNGs, derive the JAX key at the boundary:
`key = jax.random.PRNGKey(int(seed))`. Persist `seed` (an int), not
the key, to keep cache fingerprints stable.

## scipy → jax.scipy

Map scipy calls to their JAX equivalents wherever available:

| scipy | jax.scipy |
|-------|-----------|
| `scipy.stats.norm.cdf(x)` | `jax.scipy.stats.norm.cdf(x)` |
| `scipy.stats.norm.sf(x)` | `1 - jax.scipy.stats.norm.cdf(x)` (no `sf` in jax.scipy.stats) |
| `scipy.stats.norm.ppf(q)` | `jax.scipy.stats.norm.ppf(q)` |
| `scipy.stats.chi2.sf(x, df)` | `1 - jax.scipy.stats.chi2.cdf(x, df)` |
| `scipy.special.erf(x)` | `jax.scipy.special.erf(x)` |
| `scipy.special.gammaln(x)` | `jax.scipy.special.gammaln(x)` |

Keep using **scipy** for distributions where JAX has no equivalent,
or where the call is not on a hot/differentiable path (e.g. test
fixtures, one-shot CDF inversions inside non-traced code). Annotate
each scipy call left in place with a short `# scipy: <reason>`
comment so a future cleanup agent knows whether to chase it.

## Optimisation / root-finding

`scipy.optimize.brentq` has no JAX equivalent. The framework already
has `tilting/_solvers.py::brentq_with_doubling`; port it to JAX
using `jax.lax.while_loop` only if the bracketing logic is on a
traced path. For non-traced callers (CI inversion at the public
boundary, where we just need a scalar root), keep the existing
scipy implementation. Don't introduce `jaxopt` unless a real need
materialises.

## jit / vmap / grad

- **Don't `jit` defensively.** Add `@jax.jit` only after profiling
  shows it pays off. Most numerics paths are fast enough without
  it; over-jitting blows up tracing time during tests.
- `jax.vmap` is the right answer when a numpy loop iterates over
  the leading dim of an array.
- `jax.grad` and `jax.hessian` are the unifying autodiff tools — use
  them for `Model.fisher_information` defaults, MC pvalue gradient
  paths, and learned-η loss gradients. Never compute Hessians by
  finite differences.

## PyTree discipline

When we get to `learned/`, MLP modules become Equinox modules
(`eqx.Module` subclasses). They are PyTrees of arrays. Conventions:

- Static fields use `eqx.field(static=True)` (e.g. layer sizes).
- Optimiser state from `optax.adam(...)` is also a PyTree; thread
  it through training loops explicitly.
- Save/load via `eqx.tree_serialise_leaves` / `tree_deserialise_leaves`,
  not `pickle`. The serialiser is forward-compatible across JAX
  versions.

## File splitting

Don't introduce new files just because of the port. Exceptions:

- An Equinox module file naturally splits when a `nn.Module` had
  multiple architectural variants in one file. Then one file per
  variant is fine.
- A standalone JAX-numerics utility (e.g. a generic bisection
  in `lax.while_loop` form) deserves its own file iff it's reused
  by 3+ callers.

## Tests

- Existing baselines stay green at `atol ≤ 1e-12` on `float64`. Do
  not loosen tolerances. If a baseline drifts at ULP scale,
  re-pin and document the reason in the test docstring.
- Add nothing to test surface during the port itself — tests for
  new generic pathways come in Phases 1–4.
- Hypothesis tests in `tests/properties/` may need explicit
  `np.asarray(...)` at array boundaries; the strategies still emit
  numpy.

## What NOT to do during the port

- Do not change algorithmic behaviour. The port is mechanical.
- Do not rename public APIs. Method names, signatures, and module
  paths are stable.
- Do not introduce `@jax.jit` speculatively.
- Do not collapse multiple modules together.
- Do not move scipy fallbacks into hidden helper modules; keep
  them visible at the call site with the `# scipy:` comment.
- Do not touch `legacy/`.
