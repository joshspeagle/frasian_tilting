# Trained checkpoints (Phase E)

This directory holds trained Phase E `EtaArtifact` checkpoints used by
`LearnedDynamicEtaSelector`. Each checkpoint is a self-describing
`.pt` file with the keys listed in
`src/frasian/learned/eta_artifact.py:_REQUIRED_KEYS` (format v2).

Phase E checkpoints are **per-experiment**: each one is trained at one
specific (model, prior, scheme, statistic) configuration recorded in
its `experiment_config` metadata. The selector compares the trained
configuration's fingerprints against inference-time `model.fingerprint()`
and `prior.fingerprint()`; mismatch raises `MissingArtifactError`.

## What's committed

Only the **v0_smoke** test fixtures are committed to git:

| File | Config | Scheme | Purpose |
|---|---|---|---|
| `learned_eta_canonical_normal_normal_powerlaw_v0_smoke.pt` | `canonical_normal_normal_powerlaw.yaml` | `power_law` | L2/L3/L4 test fixture |
| `learned_eta_canonical_normal_normal_ot_v0_smoke.pt` | `canonical_normal_normal_ot.yaml` | `ot` | L2/L3/L4 test fixture |

These are intentionally small (64×64 hidden, 1024 LHS, ~30 epochs)
— enough to pass calibration + narrowness regressions at the trained
w but not for production use.

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
    --out artifacts/learned_eta_canonical_normal_normal_powerlaw_v1.pt --version v1

python -m scripts.train_learned_eta \
    --config experiments/canonical_normal_normal_ot.yaml \
    --n-lhs 10000 --n-epochs 100 --batch-size 256 --n-aux 256 \
    --lambda-max 10.0 --lambda-warmup-frac 0.3 \
    --patience 15 --min-delta 1e-4 \
    --out artifacts/learned_eta_canonical_normal_normal_ot_v1.pt --version v1
```

Expected runtime on CPU: ~10–30 minutes per scheme with early
stopping.

## Resolution order

`_default_cells._make_learned_selector(scheme)` tries checkpoints in
this order:

1. `learned_eta_<config_name>_v1.pt`         (production)
2. `learned_eta_<config_name>_v0_smoke.pt`   (committed test fixture)

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
- `src/frasian/learned/training/pvalue_torch.py` (torch p-value)
- `src/frasian/learned/training/losses.py` (boundary penalty / width)
- `src/frasian/learned/training/sampling.py` (ExperimentConfig schema)

you must retrain. The checkpoint format records most of these; the
v2 selector refuses to load mismatching fingerprints.

## Format details

See `docs/methods/learned_eta.md` for the brief, and
`src/frasian/learned/eta_artifact.py` for the documented checkpoint
format v2 spec.
