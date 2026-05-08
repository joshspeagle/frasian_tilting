"""Verify every registered method has a brief, property tests, and an illustration.

Exit code 0 if every registered class passes; 1 otherwise. Run by
`.github/workflows/method-completeness.yaml` (added in Step 7) and as a
local pre-commit gate.

For each `RegistryEntry`:
  - The brief at `<entry.brief>` exists.
  - Every required section header (per `REQUIRED_SECTIONS`) is present
    in the brief.
  - At least one property test file exists matching
    `tests/properties/test_<name>_*.py` OR
    `tests/properties/test_*<name>*.py` (loose match for normal_distribution
    serving normal_normal, etc.).
  - For tilting schemes / statistics / experiments / models, an
    illustration script exists under
    `src/frasian/experiments/illustrations/<name>_demo.py` (if relevant).
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

REQUIRED_SECTIONS = [
    "Summary",
    "Motivation",
    "Definition",
    "Derivation",
    "Predicted behavior",
    "Failure modes",
    "Invariants",
    "Literature",
    "Links",
]

# Categories that must ship with an illustration.
ILLUSTRATED_KINDS = {"tilting", "statistic"}


def _check_brief(entry, errors: list[str]) -> None:
    brief_path = REPO_ROOT / entry.brief
    if not brief_path.exists():
        errors.append(f"  - {entry.kind} '{entry.name}': brief missing at {entry.brief!r}")
        return
    text = brief_path.read_text()
    for section in REQUIRED_SECTIONS:
        # Match Markdown header `## <section>` (case-insensitive, exact).
        pattern = re.compile(rf"^##\s+{re.escape(section)}\s*$", re.IGNORECASE | re.MULTILINE)
        m = pattern.search(text)
        if not m:
            errors.append(
                f"  - {entry.kind} '{entry.name}': brief at {entry.brief!r} "
                f"missing section '## {section}'"
            )
            continue
        # Audit P1 M.7: section bodies must be non-empty. Pre-fix a
        # bare `## Derivation` header with no body would pass; this
        # let `/propose-method` ship empty briefs that the reviewer
        # had to manually inspect. Now we slice from the end of the
        # matched header to the next `##` (or EOF) and require some
        # non-whitespace, non-comment content.
        body_start = m.end()
        next_hdr = re.search(r"^##\s+\S", text[body_start:], re.MULTILINE)
        body = text[body_start:body_start + next_hdr.start()] if next_hdr else text[body_start:]
        # Strip whitespace and HTML comments; require at least one
        # alphabetic character in the body.
        stripped = re.sub(r"<!--.*?-->", "", body, flags=re.DOTALL).strip()
        # Allow `TODO` / `(intentionally empty)` only with explicit
        # opt-out so a placeholder isn't silently accepted.
        if not re.search(r"[A-Za-z]", stripped):
            errors.append(
                f"  - {entry.kind} '{entry.name}': brief at {entry.brief!r} "
                f"section '## {section}' has empty body (must contain "
                f"non-whitespace content)."
            )


def _check_property_tests(entry, errors: list[str]) -> None:
    if entry.status == "stub":
        return  # stubs may have only skipped property tests; that is OK

    # Different test directories for different kinds.
    if entry.kind == "experiment":
        # Experiments are structural; their tests live in tests/experiments/.
        candidates = list((REPO_ROOT / "tests" / "experiments").glob(f"test_*{entry.name}*.py"))
        test_loc = "tests/experiments"
    else:
        candidates = list((REPO_ROOT / "tests" / "properties").glob(f"test_*{entry.name}*.py"))
        test_loc = "tests/properties"
    # Allow the conjugate-Normal model's invariants to live under
    # test_normal_distribution.py (the protocol tests for its constituent
    # NormalDistribution cover the load-bearing properties).
    if (
        not candidates
        and entry.name == "normal_normal"
        and (REPO_ROOT / "tests" / "properties" / "test_normal_distribution.py").exists()
    ):
        return
    if not candidates:
        errors.append(
            f"  - {entry.kind} '{entry.name}': no test file matching "
            f"'{test_loc}/test_*{entry.name}*.py'"
        )


def _check_illustration(entry, errors: list[str]) -> None:
    if entry.kind not in ILLUSTRATED_KINDS:
        return
    if entry.status == "stub":
        return
    illust = (
        REPO_ROOT / "src" / "frasian" / "experiments" / "illustrations" / f"{entry.name}_demo.py"
    )
    if not illust.exists():
        errors.append(
            f"  - {entry.kind} '{entry.name}': missing illustration at "
            f"src/frasian/experiments/illustrations/{entry.name}_demo.py"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action="store_true")
    parser.parse_args(argv)

    # Make sure concrete implementations are loaded into the registry.
    from frasian._registry_bootstrap import bootstrap

    bootstrap()
    from frasian._registry import registry

    errors: list[str] = []
    n_checked = 0
    for entry in registry.all_entries():
        n_checked += 1
        _check_brief(entry, errors)
        _check_property_tests(entry, errors)
        _check_illustration(entry, errors)

    if errors:
        offending = set()
        for e in errors:
            parts = e.split("'")
            if len(parts) > 1:
                offending.add(parts[1])
        print(f"method-completeness check FAILED on " f"{len(offending)} method(s):")
        for e in errors:
            print(e)
        return 1

    print(f"method-completeness check OK ({n_checked} entries verified)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
