"""Lazy imports of optional dependencies with a structured install hint."""

from __future__ import annotations

import importlib
from types import ModuleType

from .errors import OptionalDependencyError


def require_optional(module: str, *, extra: str, feature: str) -> ModuleType:
    """Import ``module`` or raise :class:`OptionalDependencyError`.

    ``extra`` names the lunarscout extra that installs it, for example
    ``"plot"`` for matplotlib.
    """
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        install = f"pip install 'lunarscout[{extra}]'"
        raise OptionalDependencyError(
            f"{feature} requires the optional dependency '{module.split('.')[0]}'. "
            f"Install it with: {install}",
            code="optional_dependency_missing",
            details={"module": module, "extra": extra, "install": install},
        ) from exc
