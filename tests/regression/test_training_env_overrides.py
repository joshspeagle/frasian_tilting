"""Regression: training hardcoded constants honour env-var overrides.

Audit P2 (Cluster G) flagged five hardcoded knobs in
``learned/training/`` and ``tilting/eta_selectors.py`` as having no
YAML / CLI override path. Three of those knobs are precision /
wallclock tradeoffs and were exposed via env vars in this cluster:

  FRASIAN_N_MC_TRAIN              → _train_loop.N_MC_TRAIN
  FRASIAN_N_GRID_GENERIC_TRAINING → _losses_compose._N_GRID_GENERIC_TRAINING
  FRASIAN_N_MC_VALIDITY           → validity._N_MC_VALIDITY

The remaining two (``_FP_SLACK`` and ``_CLAMP_FAIL_THRESHOLD``) are
correctness-tuning knobs and were kept private by design — both are
documented in their respective module-level comments.

Reads happen at module-import time. To exercise the env-var path
in a single test process, we monkey-patch the env var and call the
``_resolve_*`` helpers directly (rather than reload the module).
"""

from __future__ import annotations

import pytest

# Import the resolvers at collection time so the env-var flips inside
# each test don't trigger module-level evaluation (the modules eagerly
# call the resolver at import time to set the public constant). The
# resolver functions themselves are pure and re-read the env var on
# every call, so per-test monkey-patching exercises the override path
# without re-importing.
from frasian.learned.training._losses_compose import (
    _resolve_n_grid_generic_training,
)
from frasian.learned.training._train_loop import _resolve_n_mc_train
from frasian.learned.training.validity import _resolve_n_mc_validity


@pytest.mark.L2
class TestEnvVarOverrides:
    def test_n_mc_train_default(self, monkeypatch):
        monkeypatch.delenv("FRASIAN_N_MC_TRAIN", raising=False)
        assert _resolve_n_mc_train() == 8

    def test_n_mc_train_overridden(self, monkeypatch):
        monkeypatch.setenv("FRASIAN_N_MC_TRAIN", "32")
        assert _resolve_n_mc_train() == 32

    def test_n_mc_train_rejects_zero(self, monkeypatch):
        monkeypatch.setenv("FRASIAN_N_MC_TRAIN", "0")
        with pytest.raises(ValueError, match=">= 1"):
            _resolve_n_mc_train()

    def test_n_mc_train_rejects_garbage(self, monkeypatch):
        monkeypatch.setenv("FRASIAN_N_MC_TRAIN", "not-a-number")
        with pytest.raises(ValueError, match="positive int"):
            _resolve_n_mc_train()

    def test_n_grid_generic_training_default(self, monkeypatch):
        monkeypatch.delenv("FRASIAN_N_GRID_GENERIC_TRAINING", raising=False)
        assert _resolve_n_grid_generic_training() == 512

    def test_n_grid_generic_training_overridden(self, monkeypatch):
        monkeypatch.setenv("FRASIAN_N_GRID_GENERIC_TRAINING", "1024")
        assert _resolve_n_grid_generic_training() == 1024

    def test_n_grid_generic_training_rejects_too_small(self, monkeypatch):
        monkeypatch.setenv("FRASIAN_N_GRID_GENERIC_TRAINING", "8")
        with pytest.raises(ValueError, match=">= 16"):
            _resolve_n_grid_generic_training()

    def test_n_mc_validity_default(self, monkeypatch):
        monkeypatch.delenv("FRASIAN_N_MC_VALIDITY", raising=False)
        assert _resolve_n_mc_validity() == 32

    def test_n_mc_validity_overridden(self, monkeypatch):
        monkeypatch.setenv("FRASIAN_N_MC_VALIDITY", "200")
        assert _resolve_n_mc_validity() == 200

    def test_n_mc_validity_rejects_zero(self, monkeypatch):
        monkeypatch.setenv("FRASIAN_N_MC_VALIDITY", "0")
        with pytest.raises(ValueError, match=">= 1"):
            _resolve_n_mc_validity()
