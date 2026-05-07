# Frasian Inference framework — audit & teardown report

**Branch:** `claude/audit-framework-c9DQT`
**Methodology:** 12 parallel sub-agents (5 deep on the four framework targets + math cross-check, 7 broad sweep across modules / tests / docs). Audit-only — no code changes in this pass. Each agent returned numbered findings with file:line, severity (P0/P1/P2), and a one-line recommendation. Findings below are de-duplicated and re-prioritized by impact.

**Severity legend:**
- **P0** — correctness bug or unstated invariant violation reachable from the canonical sandbox; block on these.
- **P1** — silent failure mode, undertested guarantee, or design-contract gap; ship-blockable but not currently producing wrong numbers in the headline experiments.
- **P2** — drift, dead code, polish, doc rot.

---

## Executive verdict per the four framework targets

| Target | Holds? | Status |
|---|---|---|
| 1. Test stat depending on (likelihood, prior, θ) | **Partial** | WALDO closed form is correct on n=1 NN+Normal; generic path silently disagrees with closed form for `data.size > 1` (P0-1) and has a variance-floor inconsistency that produces degenerate CI collapse (P0-2). |
| 2. Tilting interpolation with extrapolation beyond {likelihood, posterior} | **Fails** | Power_law allows extrapolation closed-form only; `NumericalEtaSelector` silently caps η at 0.999 (P0-5); OT extrapolation is structurally forbidden at three independent gates (P0-4); `MixtureTilting` endpoint convention contradicts both its own brief and the OT brief table (P0-3). |
| 3. η(θ) parameterization via uniform `EtaSelector` contract | **Fails (signature-uniform, semantically split)** | `IdentityTilting` ignores its selector entirely (P0-6). `NumericalEtaSelector.select` and `DynamicNumericalEtaSelector.select` consume `D = data.mean()`, contradicting the protocol's "θ-only model-agnostic" docstring (P0-7). The contract is signature-uniform but split semantically into "θ-only / calibrated" vs "D-conditional / post-selection" with no distinguishing flag. |
| 4. NN fits η(θ) targeting CI-width / integrated-α / calibration objective | **Partial** | Training loop implements documented dual-head objective; JAX kernel agrees with Theorem 6 at atol 1e-10. But narrowness regression's smoke tolerance (rel_tol=0.15-0.30) cannot detect a 25% width regression (P0-13); `regen_headline.py` uses non-stable `hash()` seed under `PYTHONHASHSEED=random` (P0-12); `learned_eta.md` brief is pre-JAX-port and lists files / artifacts that no longer exist (P0-11). |

**Math cross-check (deriver agent):** All five (analytic ↔ generic) cells were verified mathematically and are exercised by active L2/L3 regression tests at documented tolerances. The only caveat is the *deliberately-biased* training surrogate `generic_grid_tilted_pvalue` vs Theorem 8 (`test_grid_surrogate_vs_theorem8.py` pins the bias as a regression). Inference paths route around it via `power_law_tilted_pvalue_jax`. **Agreement-verified for all 5 cells.**

---

## P0 — block before next checkpoint regen / paper run

**P0-1. WALDO generic CI silently mis-sizes the MC reference for n>1 Normal-Normal data.**
`src/frasian/statistics/waldo.py:338` (also `normal_normal.py:106`).
The Normal-Normal closed-form path collapses `data` to the scalar mean `D` (line 145–146). The generic path uses `n_obs = int(data_arr.size)`. For `data` of size n>1, the closed form treats it as a single sufficient stat with variance σ², while the generic resamples `n_obs=n` draws and runs `posterior(D')` on an n-vector (variance σ²/n). The "generic == closed form on Normal-Normal" guarantee silently breaks for any `data.size > 1`. The cross-check test only ever passes `np.asarray([D])` so the bug is undiscovered.
**Fix:** assert `data.size == 1` in the closed-form path, or fix the closed form to use σ/√n; mirror in tests.

**P0-2. `WaldoStatistic` variance-floor inconsistency produces degenerate-posterior CI collapse.**
`src/frasian/statistics/waldo.py:244` vs `:301-305`.
`_generic_evaluate` floors variance at `1e-300` → returns `(diff)² / 1e-300` (huge but finite). `_generic_mc_reference` uses `if var <= 0: t_samples[i] = 0.0` → forces non-rejection. At degenerate posterior variance, `t_obs` is enormous and `t_ref` is all zero — the empirical p-value collapses to `1/(n_mc+1)` regardless of θ, producing a CI that pins to the support boundary.
**Fix:** make the two floors agree (both clamp; or both → 0), or raise on degenerate variance.

**P0-3. `MixtureTilting` endpoint convention contradicts brief + sibling-scheme briefs.**
`src/frasian/tilting/mixture.py:36-39`; `docs/methods/mixture.md:65-67`; `docs/methods/ot.md:30-31`.
Code: `eta_identity=1.0` with description `"0=prior, 1=posterior"`. Brief: `q = (1-η)·post + η·L/Z_L`, η=0 recovers posterior. Class docstring: `q(θ;η) = (1-η)·π + η·post`. OT brief table: `mixture: η=0 → posterior, η=1 → likelihood`. **Three mutually contradictory definitions.** `is_identity(1.0) → True` under current code, breaking the framework's anchor that "all schemes have η=0 at posterior."
**Fix:** pick one (the brief; matches `power_law` and `ot`), set `eta_default=0.0, eta_identity=0.0`, rewrite the class docstring.

**P0-4. OT extrapolation beyond {likelihood, posterior} is structurally forbidden at three independent gates.**
`src/frasian/tilting/quantile_mixture.py:62-64`, `src/frasian/tilting/ot.py:509-510, 597`.
The W2 geodesic is in fact well-defined for `t ∈ ℝ` (a *line* in Wasserstein space, not just the geodesic segment), but `QuantileMixturePath.__init__` raises on `t ∉ [0, 1]`, and both `OTTilting.tilt` and `OTTilting.tilted_pvalue` have the same hard guard. The framework's stated goal of "extrapolation beyond bounds" is silently blocked for OT.
**Fix:** relax to `np.isfinite(t)`, OR document explicitly in `ot.md` "OT is interval-only by policy" (the latter is a doc fix, not code).

**P0-5. `NumericalEtaSelector` silently caps `eta_hi` at 1.0 - 1e-3 even when admissible range goes higher.**
`src/frasian/tilting/eta_selectors.py:251`.
Lines 218-219 compute the correct admissible `(eta_lo, eta_hi) = (-w/(1-w)+buf, 1/(1-w)-buf)`. Line 251 then unconditionally narrows: `eta_hi = min(eta_hi, 1.0 - self.eta_min_buffer)`. For `w=0.2`, the admissible top is `1/0.8 = 1.25`, but the selector cannot pick anything above 0.999. No comment, no warning. The brief says η>1 "oversharpens"; the selector is forbidden to discover that.
**Fix:** remove the line-251 cap or comment its rationale and pin via a regression.

**P0-6. `IdentityTilting` bypasses the selector entirely; `EtaSelector` contract is not enforced.**
`src/frasian/tilting/identity.py:74-83, 100-101, 110-112`.
`confidence_interval` / `confidence_regions` / `pvalue` all `return statistic.confidence_interval(...)` — never call `self.selector.select(...)`. The `selector` field exists and defaults to `FixedEtaSelector(eta=0.0)` but is structurally dead. A user passing `IdentityTilting(selector=NumericalEtaSelector())` thinking it tilts gets a silent no-op.
**Fix:** drop `selector` field on `IdentityTilting`, or assert the resolved η is `is_identity(η)`.

**P0-7. `NumericalEtaSelector.select` consumes `D = data.mean()` — protocol contract is not "θ-only / model-agnostic."**
`src/frasian/tilting/base.py:155-176` (protocol docstring) vs `eta_selectors.py:222-247`.
The protocol docstring says "the call surface is θ-space and model-agnostic. `data` … passed at call time; selectors do not consume Normal-Normal-specific |Δ| summaries." But `NumericalEtaSelector.select` does `D = data.mean()` and minimizes width *at that D* — η is a function of D, not θ. This is the documented post-selection inference path, but the contract claim and the implementation diverge silently. **Related:** `DynamicNumericalEtaSelector.select(data=[D])` (single-context API) reduces to the same D-conditional behavior, contradicting CLAUDE.md's "DynamicNumerical is θ-only and calibrated" claim for the `select` entry-point.
**Fix:** add an `is_post_selection: bool` flag the runner gates on; document the asymmetry in the protocol docstring; force callers of dynamic selectors to use `select_grid` (not `select`).

**P0-8. `_dynamic.py` strict-inequality crossing detector misses tangential α-touches.**
`src/frasian/tilting/_dynamic.py:254`.
`if diff[i] * diff[i + 1] < 0.0:` uses strict `<`, so a grid point where `p_theta[i] == alpha` and the neighbour stays positive yields no crossing. At dynamic-WALDO multimodal regimes, p often grazes α; the brentq refinement is then skipped, the region boundary is missed, and downstream the stitch-by-pairs `for i in range(0, len(entries) - 1, 2)` may fold two regions into one.
**Fix:** use `<= 0.0` plus an explicit `diff[i] == 0` guard, or test `np.sign` transitions including zero.

**P0-9. `_dynamic.py` region-stitching can silently drop the trailing crossing on odd-parity input.**
`src/frasian/tilting/_dynamic.py:281-289`.
`for i in range(0, len(entries) - 1, 2)` silently drops the trailing crossing when `len(entries)` is odd (e.g. exactly 3 grid sign-flips because of MC noise / multi-modal selectors). The dropped crossing is the right edge of a real region, so the union-width is under-reported. No warning or assertion.
**Fix:** assert even parity post-padding, or pad with `theta_hi` when `p_theta[-1] > 0` even by a hair.

**P0-10. Stub tilting schemes (`MixtureTilting`, `FisherRaoTilting`) raise `AttributeError`, not `NotImplementedError`.**
`src/frasian/tilting/mixture.py:31-56`, `fisher_rao.py:42-69`.
They declare only `tilt`, `path`, `is_identity` — no `confidence_interval`, `confidence_regions`, `pvalue`. The `TiltingScheme` Protocol is `runtime_checkable` (Protocols don't check signatures), so `isinstance(scheme, TiltingScheme)` returns True. The runner enqueues these cells; first access raises `AttributeError`. `_runner.py:160-199` catches it as `status="error"` (traceback) rather than `status="incompatible"` (clean skip). **Same problem applies to stub statistics** (`LRTStatistic`, `SignedRootStatistic`, `BartlettStatistic`): runner records `status="error"`. The runner's stub-detection should consult `registry.entries[name].status == "stub"` before invocation.
**Fix:** add stub `confidence_regions` / `pvalue` / `evaluate` etc. that raise `NotImplementedError`; gate on `registry.status == "stub"` in the runner before calling.

**P0-11. `learned_eta.md` brief is pre-JAX/Equinox port — entire torch surface is wrong.**
`docs/methods/learned_eta.md`.
Brief still references `torch.autograd`, `torch.func.functional_call`, `torch.save`, `torch.trapezoid`, `pvalue_torch.py`, `cd_torch.py`, `BCEWithLogitsLoss`, `.pt` artifacts. Codebase is now Equinox/Optax (`pvalue_jax.py`, `cd_jax.py`, `.eqx` artifacts). Brief lists `tests/regression/test_torch_pvalue_matches_numpy.py` and `test_torch_cd_matches_numpy.py` — neither exists. Headline numbers in brief (3.67 / 3.71 / 3.80) **do not match CLAUDE.md** (3.63 / 3.64 / 3.68 / 3.75 / 3.82). Bernoulli smoke checkpoint is undocumented. **The single highest-leverage doc fix in the whole repo.**
**Fix:** rewrite §3 Definition, §7 Invariants, §9 Links; reconcile headline table with CLAUDE.md.

**P0-12. `regen_headline.py` and `test_learned_eta_narrowness.py` use `hash()`-derived seed under `PYTHONHASHSEED=random`.**
`tests/regression/test_learned_eta_narrowness.py:123, 161`; `scripts/regen_headline.py:130`.
The narrowness regression's seed is `hash(scheme_label) % 50`, which is process-local under default Python. The exact same bug was previously found and fixed in the calibration test (which now uses `_SCHEME_SEED_OFFSET`); the comment at `test_learned_eta_calibration.py:95-100` even names the issue. Narrowness still has the bug. `regen_headline.py` does not pin `PYTHONHASHSEED`. Net effect: the headline reproducibility claim "byte-reproducible on a clean tree" does not hold for narrowness.
**Fix:** apply `_SCHEME_SEED_OFFSET` to the narrowness test; pin `PYTHONHASHSEED=0` in `regen_headline.py`.

**P0-13. Smoke-mode narrowness tolerance is 0.78 wide → 25% width regressions pass silently.**
`tests/regression/test_learned_eta_narrowness.py:113, 151`.
`test_learned_no_wider_than_wald` only sweeps θ ∈ [-3, 3]; the CLAUDE.md narrowness claim is at θ=±4. `test_learned_beats_bare_waldo_at_conflict` runs at θ=±4 with `n_reps=100` and threshold "≤ Wald + 2·SE + rel_tol·Wald(0.15-0.30)" — for the OT smoke checkpoint (rel_tol=0.30), that's 0.78 wide. **A 25% inflation passes.** Smoke regression cannot fail unless something is catastrophically broken.
**Fix:** add a v1-only strict assertion (rel_tol=0.0, n_reps≥500); document smoke as advisory.

**P0-14. `train_learned_eta.py` accepts `--device cuda` but stores `"cuda"` while JAX expects `"gpu"`.**
`scripts/train_learned_eta.py:62`; `src/frasian/learned/training/_setup.py:32-44`.
A user passing `--device cuda` gets a CPU run while the metadata claims `cuda`. `resolve_device` only rewrites `"auto"`.
**Fix:** rewrite `"cuda"` → `"gpu"` in the resolver, or drop `cuda` from CLI choices.

**P0-15. `train_learned_eta.py` silently overwrites existing committed v0_smoke checkpoints.**
`scripts/train_learned_eta.py:127-147`; `_checkpoint._write_eqx_file` uses `os.replace` unconditionally.
A second invocation with the same `--out` path overwrites the committed artifact with no prompt. CLAUDE.md notes v0_smoke is committed and v1 retraining drifts within ~1× SE — so overwrite + commit silently moves the headline.
**Fix:** refuse if `out.exists()` unless `--force` is passed.

**P0-16. `_check_alpha` accepts marginalised checkpoints silently when `alpha=None` at inference.**
`src/frasian/tilting/eta_selectors.py:600-606`.
`_check_alpha` returns early when `meta.get("alpha") is None`. A user training with `loss_kind="static_width", alpha=0.05` and accidentally swapping in a `cd_variance` checkpoint at inference gets no failure. There's no `alpha_mode in {"marginalised","fixed_<a>"}` field — the contract overloads `None`.
**Fix:** add explicit `alpha_mode` metadata field; refuse on overload.

**P0-17. Bernoulli coverage test is vacuous — only Bernoulli framework-correctness check.**
`tests/regression/test_bernoulli_coverage.py:84-88`; `tests/integration/test_bernoulli_end_to_end.py:74-78`.
Asserts `0.0 <= empirical <= 1.0` (vacuous) and `lo <= MLE <= hi` on a single fixed dataset. The Bernoulli generic path could under-cover by 50 points and these tests pass. The CLAUDE.md headline claim "WALDO uses an MC reference distribution under H_0 sampled via `model.sample_data` … coverage at nominal level" is **unverified**.
**Fix:** add a nightly L3 test at `n_reps>=300` checking `|empirical - (1-alpha)| < 3·SE`.

**P0-18. Stub property tests are silent placeholders with no assertion.**
`tests/properties/test_{lrt,mixture,signed_root,fisher_rao,bartlett}_invariants.py`.
All 23 `@pytest.mark.skip` methods have empty bodies (just docstrings). No test verifies the stubs raise `NotImplementedError`. If `MixtureTilting.tilt()` silently starts returning a wrong value, no test catches it.
**Fix:** add one un-skipped sanity test per stub: `with pytest.raises(NotImplementedError): MixtureTilting().tilt(...)`.

---

## P1 — significant gaps; address before paper-grade reproducibility claim

### Test-statistic / WALDO

- **P1.** WALDO/Wald cross-check tests bypass public dispatch — they call `stat._closed_form_pvalue(...)` and `stat._generic_pvalue(...)` directly. A regression that breaks `_is_normal_normal_pair` would not be caught. `tests/regression/test_waldo_generic_matches_closed_form.py:46,67`; `test_wald_generic_matches_closed_form.py:34-35,49-50`.
- **P1.** No conflict-regime testing in WALDO generic agreement (`|Δ| ≤ 0.75` only). Same file lines 30-37.
- **P1.** CRN seed includes `alpha`, so `stat.pvalue(...)` and `stat.confidence_interval(...)` use disjoint MC streams. `waldo.py:341,378,265-271`.
- **P1.** `WaldoStatistic._generic_confidence_interval` swallows `BracketingFailed` and substitutes the support boundary with no metadata flag. `waldo.py:401-420`.
- **P1.** Wald/WALDO model dispatch uses `isinstance(model, NormalNormalModel)` — anti-pattern flagged in CLAUDE.md ("never import `frasian.models.normal_normal` outside `frasian/models/`"). `wald.py:34-35, 28`; `waldo.py:73-74, 64`.
- **P1.** `WaldStatistic.acceptance_region` is part of the protocol but raises `NotImplementedError` for non-Normal — implementation/protocol mismatch. `wald.py:191-195`; `waldo.py:443-448`.
- **P1.** WALDO MC reference biases p-values upward when posterior variance degenerates (sets `t_samples[i] = 0`). `waldo.py:301-305`.

### Tilting / interpolation

- **P1.** `power_law._generic_tilt` does not enforce admissible-η range — silently extrapolates beyond closed-form domain. `power_law.py:264-316`.
- **P1.** Phase-E checkpoint cross-experiment refusal has a silent-skip path: when `model_fingerprint is None` AND the trained checkpoint lacks `model_class`, both gates pass with no check. `eta_selectors.py:651, 658, 670`.
- **P1.** `_check_experiment` does NOT compare `n_data` or `theta_distribution_fingerprint` — a checkpoint trained at `n_data=16` will silently load and run inference for `n_data=1`. `eta_selectors.py:614+`.
- **P1.** `_dynamic.py` whole-window-accept case mis-classifies as `hit_boundary=True`, triggering 2× retry then `BracketingFailed`. `_dynamic.py:275-279`.
- **P1.** `_solvers.brentq_with_doubling` cannot detect a flat function; `f(midpoint) == 0` exactly leads to scipy accepting a no-sign-flip bracket. `_solvers.py:64-78`.
- **P1.** `OTTilting.tilt` Gaussian fast path accepts `prior` but silently ignores it. `ot.py:513-519`.

### CD / statistics

- **P1.** `cd/grid.py` `quantile` claim of "leftmost match" via `jnp.interp` is folklore — `np.interp` interpolates linearly across plateaus. `grid.py:138-154`.
- **P1.** `cd/grid.py:127-136` `cdf(+∞) ≠ 1` when the grid truncates mass; `quantile(q > cdf_values[-1])` silently clips, hiding the deficit.
- **P1.** `cd/grid.py:181-209` `secondary_modes` has off-by-one bounds; `pdf[i:].min()` includes the peak itself, returning `pdf[i]` for monotone-decreasing tails.
- **P1.** `from_pvalue.build_cd_from_pvalue` passes a length-1 `data=[D]` array, so WaldoStatistic._generic_pvalue uses `n_obs=1`, producing the wrong reference distribution for any non-Gaussian generic model. `from_pvalue.py:142-145`.

### Learned-η / training

- **P1.** `cd_variance` loss can NaN at near-zero p-curves (collapse mode); `_masked_mean` masks but `n_valid` skews. `losses.py:99`; `cd_jax.py:97-98`.
- **P1.** `extract_normal_normal_params` `_W_EPS = 1e-3` rejects `w ∉ (0.001, 0.999)` — undocumented hard limit on prior strength. `_losses_compose.py:96-103`.
- **P1.** `pvalue_jax.power_law_tilted_pvalue_jax:86` clamp `denom = max(1−η(1−w), 1e-6)` is differentiable but produces a flat plateau where width-loss gradient points the wrong way; only the boundary penalty saves training, and `lambda_warmup_frac=0.3` has Head B off for the first 30%.
- **P1.** `_call_normal_normal_pvalue` raises `NotImplementedError` on `D.ndim != 1` from inside the jit boundary instead of pre-flighting in `ExperimentConfig.__post_init__`. `_losses_compose.py:139-146`.
- **P1.** `precompute_generic_grids` is called twice (in `_make_step_fns` and `_make_eval_fn`) for non-NN configs — duplicate work. `_train_loop.py:231,287`.
- **P1.** Validity Head B trains on ~all-True labels for non-NN models (fast path returns `p ≡ 0.5`); inference-time `predict_validity` then returns trivial values. `validity.py:275-321`.
- **P1.** `_generic_grid_tilted_moments:181` `var = max(m2 - mu², 1e-12)` silently floors and disables gradient — Head A stops learning with no diagnostic counter.
- **P1.** Headline narrowness claim only tested at θ ∈ [-3, 3], not the CLAUDE.md-cited |θ|=4 conflict band. `test_learned_eta_narrowness.py:113`.
- **P1.** `_training_step` exception handler does not catch JAX `FloatingPointError` — NaN gradient propagates into optimizer state silently. `_train_loop.py:382-396`.

### Experiments / runner

- **P1.** Coverage and width experiments have partial-row hazards: a cell that succeeds on rep 0 but fails on rep 7 writes garbage into reps 0..6 then `break`s. `coverage.py:148,154`; `width.py:111-119`.
- **P1.** `coverage_se = sqrt(max(p(1-p),1e-12)/n_reps)` floors SE, silently hiding p=0/p=1 regimes. `coverage.py:154`.
- **P1.** Runner persists `out_dir` even when *all* cells fail; `figures.py` then crashes on empty raw_results. `_runner.py:222-253`; `figures.py:33-34`.
- **P1.** Coverage / width experiments are hard-wired to `NormalNormalModel`; they pass `data=np.asarray([D])` even when the YAML declares `n_data=16` (Bernoulli). Document or generalize. `coverage.py:105,137`; `width.py:105`.

### Cache / reproducibility

- **P1.** `runner.py:43` `extra = dict(raw_result.metadata)` dumps the entire computed metadata into the cache key — including `raw_fingerprint` (already a separate key) and any future timestamp / runtime telemetry → silent invalidations and JSON-serialization crashes on numpy types.
- **P1.** `digest()` uses `json.dumps` without a custom encoder — crashes on `np.float64` / `np.ndarray` / tuples / `Path` in metadata. `cache.py:46-58`.
- **P1.** `storage.save_result` is not concurrency-safe under `pytest -n auto` — the rename gap between `path → backup` and `tmp → path` is observable. `storage.py:55-86`.
- **P1.** `git_sha()` is `lru_cache(maxsize=1)`-memoized and never invalidated; "dirty trees never hit cache" silently violated mid-run. `cache.py:65-101`.
- **P1.** `git status --porcelain` ignores gitignored paths — `artifacts/learned_eta_*.eqx` mutations don't flip the tree to dirty. Cache invalidation hinges on `selector_artifact_fingerprint` plumbing only. `cache.py:88-97`.
- **P1.** `selector_artifact_fingerprint` plumbing has a single point of failure: `getattr` chain returns `None` and `except (AttributeError, TypeError)` swallows shape mismatches. v0/v1 checkpoints can collide. `runner.py:44-58`.
- **P1.** `Config.fingerprint()` excludes `numpy.__version__` / `jax.__version__` — a `pip install -U numpy` silently invalidates the documented numbers without bumping the cache key. `config.py:106-123`.
- **P1.** `RawSamples.fingerprint` hashes `D.tobytes()` ignoring dtype/shape — reshape-equivalent grids would collide. `raw.py:39-48`.
- **P1.** `get_or_compute` does not validate that loaded `_cache_key` matches the requested key. `cache.py:124-141`.

### Tests

- **P1.** `tests/integration/test_empty_registry.py` and `test_registry.py` carry **no** `@pytest.mark.LX` — `-m L0`/`-m L4` filters silently miss them.
- **P1.** Hypothesis `deadline=None` everywhere; no `--hypothesis-show-statistics`. Pathological inputs taking 30+ s won't be flagged.
- **P1.** `test_runner_resilience.py` covers only `RuntimeError` from `run_cell`; no KeyboardInterrupt / OOM / partial-write / signal-mid-write tests.
- **P1.** `_isolated_registry` autouse fixture (conftest.py:31) doesn't reset `_BOOTSTRAPPED` flag — tests calling `bootstrap()` directly after `registry.clear()` get an empty registry.
- **P1.** No EtaSelector property tests in `tests/properties/` — the protocol's `is_dynamic` / per-θ `select_grid` semantics, FixedEtaSelector identity, DynamicNumerical θ-only invariant are unpinned.
- **P1.** `IdentityTilting` lacks an L2 regression file (only property tests).
- **P1.** Power_law property tests miss the η=1 endpoint (`test_ot_invariants.py:55` covers it for OT).

### Docs

- **P1.** Broken `tests/...` links across `wald.md`, `waldo.md`, `power_law.md`, `ot.md` — references like `test_statistic_invariants.py`, `test_power_law_generic_matches_closed_form.py`, `test_ot_regression.py` do not exist. `tools/check_method_completeness.py` does not catch this.
- **P1.** `tools/check_method_completeness.py` only verifies section *headers* exist, never that bodies are non-empty — a brief with empty `## Definition\n## Derivation` passes.

---

## P2 — polish, dead code, doc rot

- `_admissible.py` is referenced in CLAUDE.md project structure but does not exist on disk; docstrings in `base.py:161` and `eta_selectors.py:207` still reference dropped `TiltingContext` / `admissible_range`.
- `power_law._stable_tilted_pvalue_seed` / `_resolve_support` are vestigial back-compat aliases.
- `NumericalEtaSelector.sigma` / `mu0` and `DynamicNumericalEtaSelector.sigma` / `mu0` are dead fields flagged "removed in commit 3a-3" but still present.
- `LearnedDynamicEtaSelector` and `DynamicNumericalEtaSelector` are dataclasses with mutable counters (`_clamped_calls`, `_last_clamped_fraction`); breaks the implicit "selectors are values" invariant.
- `__init__.py` for `tilting/` does not export concrete scheme classes — intentional but undocumented.
- `# type: ignore[return-value]` masks a real protocol violation: `power_law.tilted_pvalue` and `ot.tilted_pvalue` scalar fast paths return Python floats while protocol declares `jax.Array`. `power_law.py:895`, `ot.py:616`.
- `scripts/run.py:1-6` docstring is stale ("Step-1 implementation … will print a stub message"). Step 4 done.
- `scripts/regen_headline.py:11-13, 85-86` — torch references / legacy headline numbers, post-Phase-F.
- `_default_cells.py:60` env-var dispatch reads env every call; flipping mid-process produces fresh `EtaArtifact` with cold caches.
- `EtaArtifact._read_metadata_only` does not cap the 4-byte length prefix → adversarial `.eqx` could allocate 4 GB. `eta_artifact.py:228-239`.
- `_jax_setup.py:22` `enable_x64()` only fires if some `frasian` module imports it first; no defensive `assert jax.config.read("jax_enable_x64")` in loss kernels.
- `LearnedDynamicEtaSelector` accepts checkpoint format `v in (2, 3)` but `EtaArtifact.load` only knows v3 — v2 leaks past one gate then fails the other. `eta_selectors.py:568`.
- Hardcoded constants without YAML / CLI override: `N_MC_TRAIN=8`, `_N_GRID_GENERIC_TRAINING=512`, `_N_MC_VALIDITY=32`, `_CLAMP_FAIL_THRESHOLD=0.20`, `_FP_SLACK=1e-9`.
- CLAUDE.md cites "37 stub-skips" but actual count is 23 — off by 14 (likely stale doc, but worth confirming no skips are silently disabled elsewhere).
- `bernoulli.md` self-contradicts: Summary says "all pairings raise NotImplementedError" while later sections describe Wald/WALDO running generically.
- `test_smoothness_metrics.py` is mis-tagged `@pytest.mark.properties` but uses no hypothesis.
- `test_post_selection_coverage.py` should also carry `@pytest.mark.slow` (n_reps=600 × 3 schemes).
- `test_post_selection_coverage.py` pins direction (`< 0.945`) but not magnitude (CLAUDE.md says "~2 points"); a drift to 0.94 still passes.
- `clear_cache` doesn't recurse — `IsADirectoryError` on any future subdirectory in a result dir.
- No cache size cap or LRU pruning; `clear_cache` is all-or-nothing.
- `ProcessedResult` schema versioning is missing; `SCHEMA_VERSION` lives in `storage.py` only.
- `test_jax_determinism.py` only pins serialised-leaves equality, not end-to-end pipeline byte-reproducibility.

---

## Cross-cutting recommendations (prioritized triage list)

1. **Fix the four contract bugs** (P0-3, P0-4, P0-5, P0-6, P0-7): they make the framework's *advertised* uniformity false. None are hard fixes; mostly removing silent caps and fixing one endpoint convention.
2. **Fix the WALDO generic-vs-closed-form inconsistencies** (P0-1, P0-2): they invalidate the "agreement-verified" claim under any non-trivial sample size.
3. **Fix the multi-region CI subtle bugs** (P0-8, P0-9): they bias union widths in conflict regimes — exactly the regime the framework exists to study.
4. **Fix the stub-handling cleanly** (P0-10, P0-18): make stubs raise `NotImplementedError` at the protocol level *and* test that they do.
5. **Rewrite `learned_eta.md`** (P0-11): single highest-leverage doc fix; brief is currently a torch document for a JAX codebase.
6. **Pin reproducibility properly** (P0-12, P0-15, P1 cache): `PYTHONHASHSEED=0`, refuse-on-overwrite, library-version in fingerprint, key-validation on load.
7. **Tighten the smoke regressions to be useful** (P0-13, P0-17, P1 narrowness band): the current tolerances let regressions through.
8. **Add the missing property tests** (P1 EtaSelector invariants, P1 stub-raises tests, P1 marker tags on integration files).
9. **Decide and document the extrapolation policy** (P0-4, P0-5): if OT is interval-only by design, say so in `ot.md`; if not, relax the gates. Currently the framework promises one thing and silently does the other.
10. **Document the n=1 / n>1 sandbox boundary** (P0-1, P1 build_cd_from_pvalue): the framework operates on n=1 but several code paths claim generic-n support without testing it.

---

## Appendix — agent inventory

| ID | Role | Verdict |
|---|---|---|
| A1 | Target 1: Test stat using (lik, prior, θ) | block — P0-1, P0-2 |
| A2 | Target 2: Tilting interpolation + extrapolation | block — P0-3, P0-4, P0-5 |
| A3 | Target 3: η(θ) parameterization | block — P0-6, P0-7 + cross-experiment refusal hole |
| A4 | Target 4: NN fitting of η | block — P0-12, P0-13 + smoke tolerance |
| A5 | Math cross-check (analytic ↔ generic) | all 5 cells agreement-verified |
| B1 | tilting/ module health | block — P0-8, P0-9, P0-10 |
| B2 | statistics/ + cd/ module health | block — stub gating + degenerate-variance bias |
| B3 | experiments/diagnostics/scripts | partial-row hazards + stale torch refs |
| B4 | simulation/ + cache + reproducibility | accept-with-caveats — race + library-version blindness |
| B5 | learned/ + training/ JAX port | block — P0-14, P0-15, P0-16 |
| B6 | tests/ layering + coverage gaps | block — P0-17, P0-18 + unmarked integrations |
| B7 | docs/methods briefs vs implementation | block — P0-11 (learned_eta.md pre-port) |

**Total findings:** 18 P0, ~55 P1, ~30 P2 (de-duplicated; some agents overlapped on P0-1/P0-2/P0-8/P0-9/P0-13).

**Suggested first PR:** address P0-1, P0-2, P0-3, P0-4, P0-5, P0-6, P0-7, P0-10, P0-11 (the contract / closed-form-vs-generic / docs cluster). That alone is ~10-15 small fixes and one doc rewrite; everything else can land incrementally.
