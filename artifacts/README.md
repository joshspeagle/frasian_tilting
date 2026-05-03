# Trained checkpoints

This directory holds trained `MonotonicEtaArtifact` checkpoints used by
`LearnedDynamicEtaSelector`. Each checkpoint is a self-describing
`.pt` file with the keys listed in
`src/frasian/learned/monotonic_eta.py:_REQUIRED_KEYS`.

## What's committed

Only the **v0_smoke** test fixtures are committed to git:

| File | Size | Scheme | Purpose |
|---|---|---|---|
| `learned_eta_power_law_v0_smoke.pt` | ~18 KB | `power_law` | L2/L3/L4 test fixture |
| `learned_eta_ot_v0_smoke.pt` | ~17 KB | `ot` | L2/L3/L4 test fixture |

These are intentionally small (32×32 hidden, 1024 LHS, 30 epochs,
n_mc=4) — enough to pass calibration + narrowness regressions but
not for production use. They make the test suite reproducible without
torch-training every CI run.

## Production checkpoints (NOT committed)

Production-grade checkpoints (64×64+ hidden, n_lhs=10000, n_epochs
up to 200 with early-stopping) are not committed because they're
larger and not needed for tests. Train them locally:

```bash
# Power-law, default: integrated_p loss, α-marginalised, 5σ training range
python -m scripts.train_learned_eta \
    --scheme power_law --loss integrated_p \
    --n-lhs 10000 --n-mc 8 --n-epochs 200 \
    --shared-sizes 64 64 --mono-sizes 64 64 \
    --patience 15 --min-delta 1e-4 \
    --out artifacts/learned_eta_power_law_v1.pt --version v1

# OT
python -m scripts.train_learned_eta \
    --scheme ot --loss integrated_p \
    --n-lhs 10000 --n-mc 8 --n-epochs 200 \
    --shared-sizes 64 64 --mono-sizes 64 64 \
    --patience 15 --min-delta 1e-4 \
    --out artifacts/learned_eta_ot_v1.pt --version v1
```

Expected runtime on CPU: ~5–15 minutes per scheme with early stopping
(patience=15 typically halts within 30–60 epochs).

## Resolution order

`LearnedDynamicEtaSelector` (and the env-var switch in
`_default_cells._make_learned_selector`) tries checkpoints in this
order:

1. `learned_eta_<scheme>_v1.pt`        (production, if locally trained)
2. `learned_eta_<scheme>_v0_smoke.pt`  (committed test fixture)

The first existing file is loaded. If neither exists,
`MissingArtifactError` is raised with a hint to run the training script.

## Activating the learned selector globally

```bash
export FRASIAN_DEFAULT_DYNAMIC_ETA=learned
python -m scripts.run --fast experiment=coverage
```

Default is `numerical` (the legacy `DynamicNumericalEtaSelector`) for
backwards compatibility. Once the v1 production checkpoints are
available, set `FRASIAN_DEFAULT_DYNAMIC_ETA=learned` to opt in.

## Retraining after framework changes

If you change any of:

- `src/frasian/learned/training/architecture.py` (architecture)
- `src/frasian/learned/training/pvalue_torch.py` (torch p-value)
- `src/frasian/learned/transforms.py` (η/Δ transforms)
- `src/frasian/learned/training/sampling.py` (training distribution)

you must retrain. The checkpoint format records all of these; loading
a stale checkpoint either fails immediately (`architecture_kwargs`
mismatch → state_dict load error, or `checkpoint_format_version`
bump) or silently mismatches (transforms drift). The pre-commit hook
doesn't currently auto-retrain — do it manually.

## Format details

See `docs/methods/learned_eta.md` for the brief, and
`src/frasian/learned/monotonic_eta.py` for the documented checkpoint
format spec (REQUIRED + OPTIONAL keys).
