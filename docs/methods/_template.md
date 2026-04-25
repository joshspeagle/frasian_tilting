# <Method name>

> Status: `stub | implemented | deprecated`

## Summary

One- to three-sentence description: what is the method, what does it
compute, and what role does it play in the framework?

## Motivation

Why this method? What gap does it fill in the existing toolbox? What is the
research question that motivated adding it? Cite paper(s) if known to the
proposer.

## Definition

Formal mathematical statement. Use display math for the definition; define
every symbol used in the formula on first appearance.

## Derivation

Step-by-step derivation of the headline formula or property. This section
is filled by `/derive` (the `deriver` agent) and verified symbolically and
numerically on the conjugate-Normal sandbox before merge.

## Predicted behavior

What should the method do on the standard sandbox? Expected limits (e.g.
"recovers Wald as eta → 1"), expected qualitative behavior versus benchmarks,
and expected coverage / width characteristics.

## Failure modes

Known boundaries, regions of poor behavior, numerical hazards (e.g. small
denominators, NaN propagation, slow convergence). Each item here should
correspond to a property test or regression test.

## Invariants

Mirrored as property tests in `tests/properties/test_<name>_*.py`. Each
bullet is one property:

- Property A — what it asserts, in plain language.
- Property B — ...

## Literature

BibTeX-quality references. Each cited claim in the brief above must
appear here. The `literature-reviewer` agent fills/extends this section.

## Links

- Implementation: `src/frasian/<...>/<name>.py`
- Property tests: `tests/properties/test_<name>_*.py`
- Regression tests: `tests/regression/test_<name>*.py` (if any)
- Illustration: `src/frasian/experiments/illustrations/<name>_demo.py`
- Generated figure: `output/illustrations/<name>_demo.png`

## Status notes

Free-form: deprecation reasons, known issues, planned changes.
