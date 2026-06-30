from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import ScenarioError, ScenarioPathError, ScenarioStateError
from .temporal import TimeRange


_PRIMARY_DEM_RELATIVE_PATH = Path("dem.tif")
_HORIZONS_RELATIVE_PATH = Path("lighting/horizons")


def _resolved_scenario_root(path: str | Path) -> Path:
    try:
        root = Path(path).expanduser().resolve(strict=True)
    except (OSError, RuntimeError, ValueError) as exc:
        raise ScenarioError(
            "Scenario root does not exist or cannot be resolved.",
            code="scenario_root_not_found",
            details={"path": str(path), "error": str(exc)},
        ) from exc
    if not root.is_dir():
        raise ScenarioError(
            "Scenario root must be a directory.",
            code="scenario_root_not_directory",
            details={"path": str(root)},
        )
    return root


@dataclass(frozen=True, slots=True)
class Scenario:
    """Filesystem-only access to standard paths inside one scenario root."""

    root: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", _resolved_scenario_root(self.root))

    def _resolve_relative(self, relative_path: str | Path, *, allow_root: bool) -> Path:
        candidate_input = Path(relative_path)
        if candidate_input.is_absolute():
            raise ScenarioPathError(
                "Scenario-relative methods do not accept absolute paths.",
                code="scenario_absolute_path_rejected",
                details={"path": str(relative_path), "scenario_root": str(self.root)},
            )
        try:
            candidate = (self.root / candidate_input).resolve(strict=False)
        except (OSError, RuntimeError, ValueError) as exc:
            raise ScenarioPathError(
                "Scenario-relative path cannot be resolved.",
                code="scenario_path_invalid",
                details={"path": str(relative_path), "error": str(exc)},
            ) from exc
        if not candidate.is_relative_to(self.root):
            raise ScenarioPathError(
                "Resolved path escapes the scenario root.",
                code="scenario_path_escape",
                details={
                    "path": str(relative_path),
                    "resolved_path": str(candidate),
                    "scenario_root": str(self.root),
                },
            )
        if not allow_root and candidate == self.root:
            raise ScenarioPathError(
                "An output path must identify a location below the scenario root.",
                code="scenario_output_path_empty",
                details={"path": str(relative_path), "scenario_root": str(self.root)},
            )
        return candidate

    def path(self, relative_path: str | Path) -> Path:
        """Resolve a scenario-relative path without creating it."""

        return self._resolve_relative(relative_path, allow_root=True)

    def dem_path(self) -> Path:
        """Return the canonical primary DEM path (``dem.tif``)."""

        return self.path(_PRIMARY_DEM_RELATIVE_PATH)

    def horizons_path(self) -> Path:
        """Return the canonical horizon-tile directory path."""

        return self.path(_HORIZONS_RELATIVE_PATH)

    def output_path(self, relative_path: str | Path) -> Path:
        """Resolve a non-empty scenario-relative output path without creating it."""

        return self._resolve_relative(relative_path, allow_root=False)

    def _native_temporal(
        self,
        signal: str,
        *,
        times: TimeRange,
        storage: str,
        output: str | Path | None,
        observer_elevation_meters: float,
        overwrite: bool,
        max_in_memory_bytes: int,
        scratch_directory: str | Path | None,
        progress_callback: Any | None,
        cancellation_requested: Any | None,
        _client: Any | None = None,
        _components: Any | None = None,
    ):
        from .native_temporal import generate_temporal_signal
        from .temporal_store import _layer_metadata

        dem_path = self.dem_path()
        _dtype, georef = _layer_metadata(dem_path)
        output_path = self.output_path(output) if output is not None else None
        return generate_temporal_signal(
            signal=signal,
            scenario_root=self.root,
            dem_path=dem_path,
            horizons_path=self.horizons_path(),
            times=times,
            georef=georef,
            storage=storage,  # type: ignore[arg-type]
            output_path=output_path,
            observer_elevation_meters=observer_elevation_meters,
            overwrite=overwrite,
            max_in_memory_bytes=max_in_memory_bytes,
            scratch_directory=scratch_directory,
            progress_callback=progress_callback,
            cancellation_requested=cancellation_requested,
            _client=_client,
            _components=_components,
        )

    def sun_fraction(
        self,
        *,
        times: TimeRange,
        storage: str,
        output: str | Path | None = None,
        observer_elevation_meters: float = 0.0,
        overwrite: bool = False,
        max_in_memory_bytes: int = 2 * 1024 * 1024 * 1024,
        scratch_directory: str | Path | None = None,
        progress_callback: Any | None = None,
        cancellation_requested: Any | None = None,
        _client: Any | None = None,
        _components: Any | None = None,
    ):
        """Generate fractional solar visibility using explicit result storage."""

        return self._native_temporal(
            "sun_fraction",
            times=times,
            storage=storage,
            output=output,
            observer_elevation_meters=observer_elevation_meters,
            overwrite=overwrite,
            max_in_memory_bytes=max_in_memory_bytes,
            scratch_directory=scratch_directory,
            progress_callback=progress_callback,
            cancellation_requested=cancellation_requested,
            _client=_client,
            _components=_components,
        )

    def sun_over_horizon_deg(
        self,
        *,
        times: TimeRange,
        storage: str,
        output: str | Path | None = None,
        observer_elevation_meters: float = 0.0,
        overwrite: bool = False,
        max_in_memory_bytes: int = 2 * 1024 * 1024 * 1024,
        scratch_directory: str | Path | None = None,
        progress_callback: Any | None = None,
        cancellation_requested: Any | None = None,
        _client: Any | None = None,
        _components: Any | None = None,
    ):
        """Generate solar-center elevation above the local horizon in degrees."""

        return self._native_temporal(
            "sun_over_horizon_deg",
            times=times,
            storage=storage,
            output=output,
            observer_elevation_meters=observer_elevation_meters,
            overwrite=overwrite,
            max_in_memory_bytes=max_in_memory_bytes,
            scratch_directory=scratch_directory,
            progress_callback=progress_callback,
            cancellation_requested=cancellation_requested,
            _client=_client,
            _components=_components,
        )

    def earth_over_horizon_deg(
        self,
        *,
        times: TimeRange,
        storage: str,
        output: str | Path | None = None,
        observer_elevation_meters: float = 0.0,
        overwrite: bool = False,
        max_in_memory_bytes: int = 2 * 1024 * 1024 * 1024,
        scratch_directory: str | Path | None = None,
        progress_callback: Any | None = None,
        cancellation_requested: Any | None = None,
        _client: Any | None = None,
        _components: Any | None = None,
    ):
        """Generate Earth-center elevation above the local horizon in degrees."""

        return self._native_temporal(
            "earth_over_horizon_deg",
            times=times,
            storage=storage,
            output=output,
            observer_elevation_meters=observer_elevation_meters,
            overwrite=overwrite,
            max_in_memory_bytes=max_in_memory_bytes,
            scratch_directory=scratch_directory,
            progress_callback=progress_callback,
            cancellation_requested=cancellation_requested,
            _client=_client,
            _components=_components,
        )

    def psr(
        self,
        output: str | Path,
        *,
        horizons: str | Path | None = None,
        overwrite: bool = False,
        progress_callback: Any | None = None,
        cancellation_requested: Any | None = None,
        _bridge: Any | None = None,
    ) -> Path:
        """Generate a native permanent-shadow byte mask without registration.

        Value 255 means the Sun center never clears the local horizon across
        the native operation's six-hour 1970-2044 samples; value 0 means it
        clears the horizon at least once. GDAL validity mask 0 marks pixels
        whose required horizon tile was unavailable; mask 255 marks calculated
        pixels. Observer elevation is fixed at zero.
        """

        from .native_product import generate_psr_product

        horizons_path = self.horizons_path() if horizons is None else self.path(horizons)
        return generate_psr_product(
            scenario_root=self.root,
            dem_path=self.dem_path(),
            horizons_path=horizons_path,
            output_path=self.output_path(output),
            overwrite=overwrite,
            progress_callback=progress_callback,
            cancellation_requested=cancellation_requested,
            _bridge=_bridge,
        )


def open_scenario(
    path: str | Path,
    *,
    state: Any | None = None,
) -> Scenario:
    """Open an existing scenario directory in filesystem-only mode."""

    if state is not None:
        raise ScenarioStateError(
            "Attached scenario state is not implemented in this Lunarscout slice.",
            code="scenario_state_unavailable",
        )
    return Scenario(root=Path(path))
