"""Shared progress contracts for long-running Lunarscout operations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeAlias


Backend: TypeAlias = Literal["auto", "cpu", "cuda"]
SelectedBackend: TypeAlias = Literal["cpu", "cuda"]


@dataclass(frozen=True, slots=True)
class ProgressEvent:
    """One immutable progress observation from a long-running operation."""

    operation: str
    stage: str
    completed: int
    total: int
    fraction: float
    backend: SelectedBackend | None
    message: str
    tile_y: int | None = None
    tile_x: int | None = None
    path: Path | None = None
