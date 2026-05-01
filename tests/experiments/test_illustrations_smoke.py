"""Smoke tests for every illustration script under
`src/frasian/experiments/illustrations/`.

These import + run each demo's `main(smoke=True)` to:
  - verify the demos still execute end-to-end after refactors,
  - bring those modules into pytest's coverage report (otherwise they
    show 0% because pytest never imports them),
  - mirror what `.github/workflows/method-completeness.yaml` does.

Each demo writes a PNG into a tmp directory; the test only asserts the
file was produced, not its pixel content.
"""

from __future__ import annotations

from pathlib import Path

import pytest

DEMOS = [
    "frasian.experiments.illustrations.identity_demo",
    "frasian.experiments.illustrations.wald_demo",
    "frasian.experiments.illustrations.waldo_demo",
    "frasian.experiments.illustrations.power_law_demo",
    "frasian.experiments.illustrations.smoothness_demo",
    "frasian.experiments.illustrations.confidence_distribution_demo",
]


@pytest.mark.L4
@pytest.mark.parametrize("module", DEMOS)
def test_illustration_smoke(module, tmp_path: Path):
    import importlib

    out = tmp_path / f"{module.split('.')[-1]}.png"
    mod = importlib.import_module(module)
    written = mod.main(smoke=True, out=out)
    assert written.exists(), f"{module} produced no file at {written}"
