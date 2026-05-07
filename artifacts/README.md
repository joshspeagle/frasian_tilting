# Trained checkpoints (Phase E / Phase F-ported)

This directory holds trained Phase E `EtaArtifact` checkpoints used by
`LearnedDynamicEtaSelector`. After the Phase F port commit 3 the
checkpoint format is **v3** (Equinox/Optax/JAX) — a self-describing
binary `.eqx` container with the keys listed in
`src/frasian/learned/eta_artifact.py:_REQUIRED_KEYS`.

Phase E checkpoints are **per-experiment**: each one is trained at one
specific (model, prior, scheme, statistic) configuration recorded in
its `experiment_config` metadata. The selector compares the trained
configuration's fingerprints against inference-time `model.fingerprint()`
and `prior.fingerprint()`; mismatch raises `MissingArtifactError`.

## On-disk format (v3)

```
[4 bytes BE uint32: len(metadata_json)]
[metadata_json bytes]                          # JSON metadata header
[equinox.tree_serialise_leaves(eta_net)]       # variable length
[equinox.tree_serialise_leaves(val_net)]       # variable length
```

The metadata header carries `equinox_version` + `jax_version`,
`arch_sha`, `experiment_config` (with prior / model / theta_distribution
fingerprints), all training hyperparameters and per-epoch metrics.
The trained net leaves follow in the same order
(`eta_net`, then `val_net`).

## What's committed

Only the **v0_smoke** test fixtures are committed to git:

| File | Config | Scheme | Purpose |
|---|---|---|---|
| `learned_eta_canonical_normal_normal_powerlaw_v0_smoke.eqx` | `canonical_normal_normal_powerlaw.yaml` | `power_law` | L2/L3/L4 test fixture |
| `learned_eta_canonical_normal_normal_ot_v0_smoke.eqx` | `canonical_normal_normal_ot.yaml` (with widened θ-range) | `ot` | L2/L3/L4 test fixture |
| `learned_eta_canonical_bernoulli_powerlaw_v0_smoke.eqx` | `canonical_bernoulli_powerlaw.yaml` | `power_law` | Phase 4 generic-grid kernel fixture; first non-Normal-Normal checkpoint. Regen: `python -m scripts.regen_bernoulli_smoke` |

These are intentionally small (64×64 hidden, ~5–10K LHS, ~30–120
epochs) — enough to pass calibration + narrowness regressions at
the trained w but not for production use. The committed OT smoke
trains on a widened `theta_distribution=Uniform[-10, 10]` (vs
the canonical YAML's `[-5, 5]`) so the runtime `dynamic_ci_scan`'s
extrapolation to `θ ≈ μ₀ ± 2|Δ|` stays inside the trained support
at the conflict band.

## Production checkpoints (NOT committed)

Production-grade checkpoints (larger LHS, more epochs, full λ
schedule) are not committed because they're larger and re-trainable.
Train them locally:

```bash
python -m scripts.train_learned_eta \
    --config experiments/canonical_normal_normal_powerlaw.yaml \
    --n-lhs 10000 --n-epochs 100 --batch-size 256 --n-aux 256 \
    --lambda-max 10.0 --lambda-warmup-frac 0.3 \
    --patience 15 --min-delta 1e-4 \
    --out artifacts/learned_eta_canonical_normal_normal_powerlaw_v1.eqx --version v1

python -m scripts.train_learned_eta \
    --config experiments/canonical_normal_normal_ot.yaml \
    --n-lhs 10000 --n-epochs 100 --batch-size 256 --n-aux 256 \
    --lambda-max 10.0 --lambda-warmup-frac 0.3 \
    --patience 15 --min-delta 1e-4 \
    --out artifacts/learned_eta_canonical_normal_normal_ot_v1.eqx --version v1
```

Expected runtime on CPU (post-Phase-F port; JAX/Equinox/Optax):
~5–15 minutes per scheme with early stopping. The JAX path is
roughly 2–3× faster than the legacy torch path on the same machine
because the per-step jit-compiled forward + grad fuses what was
previously three eager ops (forward / backward / opt.step).

## Resolution order

`_default_cells._make_learned_selector(scheme)` tries checkpoints in
this order:

1. `learned_eta_<config_name>_v1.eqx`         (production)
2. `learned_eta_<config_name>_v0_smoke.eqx`   (committed test fixture)

The first existing file is loaded. The mapping from `scheme` to
`config_name` is in `_PHASE_E_CHECKPOINT_FOR_SCHEME` in
`src/frasian/_default_cells.py`. If neither exists,
`MissingArtifactError` is raised with a hint to run the training
script.

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
- `src/frasian/learned/training/sampling.py` (ExperimentConfig schema)

you must retrain. The checkpoint format records most of these; the
v3 selector refuses to load mismatching fingerprints and warns
on `equinox_version` / `arch_sha` drift.

## Format history

| Version | Period | Container | Notes |
|---|---|---|---|
| v2 | Phase E | torch `.pt` (pickled state dicts) | Legacy; not loadable post-port |
| v3 | Phase F port commit 3+ | Equinox `.eqx` (length-prefixed JSON header + serialised leaves) | Current |

See `docs/methods/learned_eta.md` for the full method brief and
`src/frasian/learned/eta_artifact.py` for the documented checkpoint
format spec.
