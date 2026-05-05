"""Decorator-based plugin registry.

Concrete `TiltingScheme` / `TestStatistic` / `Experiment` implementations
register themselves at import time via decorators. A single
`_registry_bootstrap` module imports every registered class so mypy and the
method-completeness CI check can statically enumerate methods.

Registration also records the relative path to the method's brief markdown so
`tools/check_method_completeness.py` can verify documentation completeness.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal, TypeVar

from ._errors import RegistryConflictError

Status = Literal["stub", "implemented", "deprecated"]
Kind = Literal["model", "tilting", "statistic", "experiment", "diagnostic"]

T = TypeVar("T")


@dataclass(frozen=True)
class RegistryEntry:
    """One row in the registry."""

    name: str
    kind: Kind
    cls: type
    brief: str  # path relative to repo root, e.g. "docs/methods/power_law.md"
    status: Status = "implemented"
    source_file: str = ""


@dataclass
class _Slice:
    """A view onto entries of a single kind."""

    _entries: dict[str, RegistryEntry] = field(default_factory=dict)

    def __contains__(self, name: str) -> bool:
        return name in self._entries

    def __getitem__(self, name: str) -> Any:
        return self._entries[name].cls

    def __iter__(self) -> Iterator[str]:
        return iter(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def all(self) -> list[Any]:
        """Every registered class in this slice, regardless of status."""
        return [e.cls for e in self._entries.values()]

    def implemented(self) -> list[Any]:
        """Shortcut for `where(status='implemented')`. Used by the runner
        to skip stubs by default — running stub cells produces uninformative
        all-NaN results and wastes compute on the cross-product."""
        return self.where(status="implemented")

    def entries(self) -> list[RegistryEntry]:
        """Every `RegistryEntry` (with metadata) for tooling like the
        completeness checker that needs the brief path / status / source file."""
        return list(self._entries.values())

    def where(
        self, *, status: Status | None = None, name__in: Iterable[str] | None = None
    ) -> list[Any]:
        """Filter the slice by `status` and/or `name__in`. Both are optional;
        when neither is supplied, behaves as `all()`."""
        names = set(name__in) if name__in is not None else None
        out: list[Any] = []
        for entry in self._entries.values():
            if status is not None and entry.status != status:
                continue
            if names is not None and entry.name not in names:
                continue
            out.append(entry.cls)
        return out


class Registry:
    """Global plugin registry. Singleton at module level."""

    def __init__(self) -> None:
        self.models = _Slice()
        self.tiltings = _Slice()
        self.statistics = _Slice()
        self.experiments = _Slice()
        self.diagnostics = _Slice()

    def _slice(self, kind: Kind) -> _Slice:
        return getattr(self, kind + "s") if kind != "model" else self.models

    def register(self, entry: RegistryEntry) -> None:
        """Insert `entry` into the appropriate slice; raise `RegistryConflictError`
        if the (kind, name) is already taken."""
        slice_ = self._slice(entry.kind)
        if entry.name in slice_._entries:
            existing = slice_._entries[entry.name]
            raise RegistryConflictError(
                f"{entry.kind} '{entry.name}' already registered "
                f"by {existing.cls.__module__}.{existing.cls.__qualname__}"
            )
        slice_._entries[entry.name] = entry

    def all_entries(self) -> list[RegistryEntry]:
        """Concat of every slice's entries — used by the completeness checker."""
        out: list[RegistryEntry] = []
        for s in (self.models, self.tiltings, self.statistics, self.experiments, self.diagnostics):
            out.extend(s.entries())
        return out

    def clear(self) -> None:
        """Reset the registry. Tests use this; production never calls it."""
        self.models = _Slice()
        self.tiltings = _Slice()
        self.statistics = _Slice()
        self.experiments = _Slice()
        self.diagnostics = _Slice()


registry = Registry()


def _make_decorator(kind: Kind) -> Callable[..., Callable[[type[T]], type[T]]]:
    def decorator(
        *, name: str, brief: str, status: Status = "implemented"
    ) -> Callable[[type[T]], type[T]]:
        def wrap(cls: type[T]) -> type[T]:
            source_file = ""
            try:
                import inspect

                source_file = str(Path(inspect.getfile(cls)).resolve())
            except (TypeError, OSError):
                pass
            registry.register(
                RegistryEntry(
                    name=name,
                    kind=kind,
                    cls=cls,
                    brief=brief,
                    status=status,
                    source_file=source_file,
                )
            )
            cls.__frasian_registry_name__ = name  # type: ignore[attr-defined]
            cls.__frasian_registry_kind__ = kind  # type: ignore[attr-defined]
            return cls

        return wrap

    return decorator


register_model = _make_decorator("model")
register_tilting = _make_decorator("tilting")
register_statistic = _make_decorator("statistic")
register_experiment = _make_decorator("experiment")
register_diagnostic = _make_decorator("diagnostic")
