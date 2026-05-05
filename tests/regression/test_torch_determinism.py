"""Regression: ``fit_eta_artifact`` enables ``torch.use_deterministic_algorithms``.

Closes audit finding 1.2-NN4. The orchestrator should set the
deterministic flag (and ``torch.manual_seed`` + ``np.random.seed``)
as the first thing it does, so two runs at the same seed are
byte-reproducible on the same torch + GPU build.

We do *not* re-train end-to-end here (that's an L3/slow concern);
we just call the orchestrator's helper and verify the flag is set.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from frasian.learned.training.train import _enable_determinism


def test_enable_determinism_sets_flag_and_seeds():
    """``_enable_determinism(seed)`` must set the torch and cudnn flags
    and seed both RNGs.

    ``torch.are_deterministic_algorithms_enabled()`` is the canonical
    API to check the global flag. We don't directly check the cudnn
    backend on CPU-only torch builds.
    """
    _enable_determinism(seed=12345)

    assert torch.are_deterministic_algorithms_enabled(), (
        "torch.use_deterministic_algorithms(True) was not set by the orchestrator."
    )
    if hasattr(torch.backends, "cudnn"):
        assert torch.backends.cudnn.deterministic is True
        assert torch.backends.cudnn.benchmark is False

    # CUBLAS workspace config must be set so CUDA matmul is deterministic.
    assert os.environ.get("CUBLAS_WORKSPACE_CONFIG", "") != ""

    # Seeds: drawing from torch and numpy after _enable_determinism(seed)
    # twice in a row (with the same seed) should be byte-equal.
    _enable_determinism(seed=42)
    a_t = torch.randn(8)
    a_n = np.random.randn(8)
    _enable_determinism(seed=42)
    b_t = torch.randn(8)
    b_n = np.random.randn(8)
    assert torch.equal(a_t, b_t), "torch.manual_seed not honoured by _enable_determinism."
    np.testing.assert_array_equal(a_n, b_n)
