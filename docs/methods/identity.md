# identity

> Status: `implemented`

## Summary

The no-op tilting scheme. `tilt(...)` returns the input posterior
unchanged regardless of η, and `confidence_interval(...)` delegates
straight to the bare statistic. Provides the canonical "no prior
reweighting" baseline cell against which every adjusted scheme
(`power_law`, `ot_normal`, `geodesic_normal`, `mixture`, `exp_family`)
is compared.

## Motivation

The framework is structured as a `(TiltingScheme × TestStatistic)`
matrix. Without an identity tilting, statistics that ignore the prior
(e.g. Wald) had no natural cell to live in — the legacy code papered
over this by silently routing `(power_law, wald)` to the same number.
Making the no-op tilting a first-class plugin lets the runner enumerate
cells uniformly and lets statistics gate themselves to the identity
cell only (see `WaldStatistic.accepts_tilting`).

## Definition

For any prior `π`, likelihood `L`, posterior `p`,

  identity.tilt(p, π, L; η) = p,    for every η.

For any test statistic `T` admitting a CI,

  identity.confidence_interval(α, D; T) = T.confidence_interval(α, D).

## Derivation

Trivial: the identity tilting is the no-op map. Numerical equality with
the bare statistic CI is verified in
`tests/properties/test_identity_tilting_invariants.py`.

## Predicted behavior

- `(identity, wald)` cell reproduces `D ± z_{1-α/2} σ` exactly.
- `(identity, waldo)` cell reproduces the standard WALDO CI (no tilting).
- Every `coverage` / `width` measurement on `(identity, ·)` matches the
  pre-refactor `coverage` / `width` cell exactly (regression-tested).

## Failure modes

None internal to the tilting itself. A statistic that does not implement
`confidence_interval` will surface its own `NotImplementedError` from
the delegation.

## Invariants

- `tilt(...)` returns the input posterior object verbatim (identity by
  reference, not just by value).
- `is_identity(η)` returns `True` for every η.
- `confidence_interval(α, D, model, prior, T)` is exactly equal to
  `T.confidence_interval(α, D, model, prior)`.

## Literature

Not applicable — this is a structural baseline, not a published method.
The framework's two-axis setup is consistent with how Bayesian and
frequentist CI machinery is typically compared on conjugate sandboxes
(see e.g. Gelman et al., *Bayesian Data Analysis* §4 for the
conjugate-Normal pedagogy).

## Links

- Implementation: `src/frasian/tilting/identity.py`
- Property tests: `tests/properties/test_identity_tilting_invariants.py`
- Illustration: `src/frasian/experiments/illustrations/identity_demo.py`

## Status notes

The identity tilting and the `(power_law, FixedEtaSelector(0.0))` cell
are numerically equivalent on the conjugate-Normal sandbox; the runner
emits only the former by default and leaves `power_law` to contribute
the dynamic-η cell.
