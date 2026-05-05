"""Regression: EtaArtifact warns on torch_version / arch_sha mismatch.

Closes audit findings 1.4-S3 and 1.2-NN3 (the metadata side).

We directly drive ``warn_on_metadata_mismatch`` with mocked
checkpoint-state dicts so this test doesn't require torch installed.
The end-to-end ``EtaArtifact.load`` plumbing is exercised by the
torch-gated calibration regression tests.
"""

from __future__ import annotations

import warnings

import pytest

from frasian.learned.training._checkpoint import (
    arch_spec_sha,
    warn_on_metadata_mismatch,
)


def _matching_state(torch_version: str = "2.0.0") -> dict:
    eta_kwargs = {"theta_dim": 1, "hidden_sizes": (64, 64)}
    validity_kwargs = {"theta_dim": 1, "hidden_sizes": (64, 64)}
    return {
        "torch_version": torch_version,
        "arch_sha": arch_spec_sha(eta_kwargs, validity_kwargs),
        "eta_architecture_kwargs": eta_kwargs,
        "validity_architecture_kwargs": validity_kwargs,
    }


def test_no_warning_when_metadata_matches(monkeypatch):
    """Matching torch_version + arch_sha → silent."""
    monkeypatch.setattr(
        "frasian.learned.training._checkpoint._torch_version",
        lambda: "2.0.0",
    )
    state = _matching_state(torch_version="2.0.0")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warn_on_metadata_mismatch(state, artifact_path="/dummy/foo.pt")
    assert not caught, f"unexpected warnings: {[str(w.message) for w in caught]}"


def test_warns_when_torch_version_mismatches(monkeypatch):
    """Torch-version mismatch → RuntimeWarning with both versions named."""
    monkeypatch.setattr(
        "frasian.learned.training._checkpoint._torch_version",
        lambda: "9.9.9",
    )
    state = _matching_state(torch_version="2.0.0")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warn_on_metadata_mismatch(state, artifact_path="/dummy/foo.pt")
    msgs = [str(w.message) for w in caught if issubclass(w.category, RuntimeWarning)]
    assert any(
        "torch version mismatch" in m and "2.0.0" in m and "9.9.9" in m for m in msgs
    ), f"expected a torch-version mismatch warning; got {msgs!r}"


def test_warns_when_arch_sha_mismatches(monkeypatch):
    """arch_sha mismatch → RuntimeWarning."""
    monkeypatch.setattr(
        "frasian.learned.training._checkpoint._torch_version",
        lambda: "2.0.0",
    )
    state = _matching_state(torch_version="2.0.0")
    state["arch_sha"] = "deadbeef" * 3  # 24 chars; not the real sha
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warn_on_metadata_mismatch(state, artifact_path="/dummy/foo.pt")
    msgs = [str(w.message) for w in caught if issubclass(w.category, RuntimeWarning)]
    assert any("arch_sha mismatch" in m for m in msgs), (
        f"expected arch_sha mismatch warning; got {msgs!r}"
    )


def test_warns_when_torch_version_field_missing(monkeypatch):
    """Pre-1.4-S3 checkpoints have no `torch_version` key → friendly warning."""
    monkeypatch.setattr(
        "frasian.learned.training._checkpoint._torch_version",
        lambda: "2.0.0",
    )
    state = _matching_state()
    state.pop("torch_version")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warn_on_metadata_mismatch(state, artifact_path="/dummy/foo.pt")
    msgs = [str(w.message) for w in caught if issubclass(w.category, RuntimeWarning)]
    assert any("no `torch_version` field" in m for m in msgs), (
        f"expected pre-1.4-S3 torch_version warning; got {msgs!r}"
    )


def test_warns_when_arch_sha_field_missing(monkeypatch):
    """Pre-1.4-S3 checkpoints have no `arch_sha` key → friendly warning."""
    monkeypatch.setattr(
        "frasian.learned.training._checkpoint._torch_version",
        lambda: "2.0.0",
    )
    state = _matching_state()
    state.pop("arch_sha")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warn_on_metadata_mismatch(state, artifact_path="/dummy/foo.pt")
    msgs = [str(w.message) for w in caught if issubclass(w.category, RuntimeWarning)]
    assert any("no `arch_sha` field" in m for m in msgs), (
        f"expected pre-1.4-S3 arch_sha warning; got {msgs!r}"
    )


def test_arch_spec_sha_changes_when_hidden_sizes_change():
    """Different hidden_sizes → different sha (the whole point)."""
    a = arch_spec_sha(
        {"theta_dim": 1, "hidden_sizes": (64, 64)},
        {"theta_dim": 1, "hidden_sizes": (64, 64)},
    )
    b = arch_spec_sha(
        {"theta_dim": 1, "hidden_sizes": (128, 64)},
        {"theta_dim": 1, "hidden_sizes": (64, 64)},
    )
    assert a != b
    # Same kwargs, different list/tuple form of hidden_sizes → same sha.
    c = arch_spec_sha(
        {"theta_dim": 1, "hidden_sizes": [64, 64]},
        {"theta_dim": 1, "hidden_sizes": (64, 64)},
    )
    assert a == c, "tuple/list of hidden_sizes must hash identically"


def test_arch_spec_sha_is_24_hex_chars():
    sha = arch_spec_sha(
        {"theta_dim": 1, "hidden_sizes": (64, 64)},
        {"theta_dim": 1, "hidden_sizes": (64, 64)},
    )
    assert len(sha) == 24
    assert all(c in "0123456789abcdef" for c in sha)


def test_arch_spec_sha_changes_when_architecture_version_changes(monkeypatch):
    """Bumping ``architecture.__version__`` flips the sha so the load-time
    mismatch warning fires even when shapes are unchanged.

    Phase 4 skeptic §4: the sha previously SHA'd only over kwargs
    (``theta_dim`` + ``hidden_sizes``). A future PR swapping GELU for
    LeakyReLU in ``architecture.py`` would not change the kwargs, so
    the old sha stayed equal and ``warn_on_metadata_mismatch`` did
    not fire. Including the bumpable ``__version__`` covers that
    failure mode.
    """
    eta_kwargs = {"theta_dim": 1, "hidden_sizes": (64, 64)}
    validity_kwargs = {"theta_dim": 1, "hidden_sizes": (64, 64)}
    sha_at_v1 = arch_spec_sha(eta_kwargs, validity_kwargs)
    monkeypatch.setattr(
        "frasian.learned.training._checkpoint._architecture_version",
        lambda: "9.9-test",
    )
    sha_at_v2 = arch_spec_sha(eta_kwargs, validity_kwargs)
    assert sha_at_v1 != sha_at_v2, (
        "arch_spec_sha must include architecture.__version__ so a bump "
        "of the version string flips the sha."
    )


@pytest.mark.L2
def test_unknown_torch_returns_none(monkeypatch):
    """When torch isn't installed, ``_torch_version`` returns None and
    ``warn_on_metadata_mismatch`` does not crash.

    This covers the audit env where the test runs at all (the calling
    code only invokes this on load-time when torch IS installed; but
    the helper must still degrade gracefully).
    """
    monkeypatch.setattr(
        "frasian.learned.training._checkpoint._torch_version",
        lambda: None,
    )
    state = _matching_state()
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        warn_on_metadata_mismatch(state, artifact_path="/dummy/foo.pt")
