# Trained checkpoints (Phase G conditional learned-η)

This directory holds trained `EtaArtifact` checkpoints used by
`LearnedDynamicEtaSelector`. Phase G ships **checkpoint format v4**
— a self-describing Equinox `.eqx` container with the keys listed in
`src/frasian/learned/eta_artifact.py:_REQUIRED_KEYS`.

Phase G checkpoints are **conditional**: each one is trained over a
*range* of `(prior_hp, lik_hp)` (the
`hyperparam_distribution` block in the experiment YAML) instead of a
single fixed prior + likelihood. At inference time the selector reads
`prior.hyperparams()` and `model.hyperparams()` and threads them into
`predict_eta(theta, prior_hp, lik_hp)`. The selector refuses inference
when the observed `(prior_class, model_class)` pair doesn't match the
trained classes, or when the observed hyperparams fall outside the
trained range.

## On-disk format (v4)

```
[4 bytes BE uint32: len(metadata_json)]
[metadata_json bytes]                          # JSON metadata header
[equinox.tree_serialise_leaves(eta_net)]       # variable length
[equinox.tree_serialise_leaves(val_net)]       # variable length
```

The metadata header carries `equinox_version` + `jax_version`,
`arch_sha`, `experiment_config` (with `prior_class`, `model_class`,
serialised `hyperparam_distribution`, theta_distribution fingerprint),
all training hyperparameters, and per-epoch metrics. The trained net
leaves follow in the same order (`eta_net`, then `val_net`).

## What's NOT committed

Unlike pre-Phase-G v0_smoke fixtures, **no v4 checkpoints are
committed to git**: the conditional architecture's wider training
distribution makes binaries larger and the YAML-driven training is
fast enough to make local regen the better option. The `.eqx` files
are gitignored via `artifacts/learned_eta_*_v4.eqx`.

## Training a fixture

```bash
python -m scripts.train_learned_eta \
    --config experiments/canonical_normal_normal_powerlaw_v4.yaml \
    --out artifacts/learned_eta_canonical_normal_normal_powerlaw_v4.eqx \
    --n-epochs 30 --batch-size 256

python -m scripts.train_learned_eta \
    --config experiments/canonical_normal_normal_ot_v4.yaml \
    --out artifacts/learned_eta_canonical_normal_normal_ot_v4.eqx \
    --n-epochs 30 --batch-size 256

python -m scripts.train_learned_eta \
    --config experiments/canonical_bernoulli_powerlaw_v4.yaml \
    --out artifacts/learned_eta_canonical_bernoulli_powerlaw_v4.eqx \
    --n-epochs 30 --batch-size 256
```

For production-grade checkpoints, increase `--n-lhs`, `--n-epochs`,
and tune the `--lambda-*` schedule (see
`docs/methods/learned_eta.md`).

Expected runtime on CPU: a few minutes per fixture.

## Resolution order

`_default_cells._make_learned_selector(scheme)` looks up
`learned_eta_<config_name>.eqx` where `config_name` is the YAML stem
(e.g. `canonical_normal_normal_powerlaw_v4`). If the file is absent,
`FileNotFoundError` is raised with a hint to run the training script.

## Activating the learned selector globally

```bash
export FRASIAN_DEFAULT_DYNAMIC_ETA=learned
python -m scripts.run --fast experiment=coverage
```

Default is `numerical` (legacy `DynamicNumericalEtaSelector`).

## Retraining after framework changes

If you change any of:

- `src/frasian/learned/training/architecture.py` (architecture)
- `src/frasian/learned/training/pvalue_jax.py` (JAX p-value)
- `src/frasian/learned/training/losses.py` (boundary penalty / width)
- `src/frasian/learned/training/sampling.py` (`ExperimentConfig` schema)
- `src/frasian/learned/training/hyperparam_distribution.py` (sampler)

you must retrain. The v4 selector refuses to load mismatching
class fingerprints and warns on `equinox_version` / `arch_sha` drift.

## Format history

| Version | Period | Container | Notes |
|---|---|---|---|
| v2 | Phase E | torch `.pt` (pickled state dicts) | Legacy; not loadable post-port |
| v3 | Phase F port commit 3+ | Equinox `.eqx`; fixed-prior architecture | Replaced by v4 |
| v4 | Phase G | Equinox `.eqx`; conditional architecture (`prior_hp` + `lik_hp` inputs) | Current |

See `docs/methods/learned_eta.md` for the full method brief and
`src/frasian/learned/eta_artifact.py` for the documented checkpoint
format spec.
