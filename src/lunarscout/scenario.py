from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
import struct
from collections.abc import Sequence
from typing import Any, BinaryIO

import numpy as np

from .errors import (
    GeoTiffMetadataError,
    NativeInputError,
    NativeProductError,
    ScenarioError,
    ScenarioPathError,
    ScenarioStateError,
)
from .geotiff import read_geotiff
from .spice_geometry import (
    BodyName,
    LonLat,
    body_azimuth_elevation,
    body_azimuth_elevation_over_horizon,
    plot_body_elevations,
)
from .temporal import TimeRange


_PRIMARY_DEM_RELATIVE_PATH = Path("dem.tif")
_HORIZONS_RELATIVE_PATH = Path("horizons")
_HILLSHADE_RELATIVE_PATH = Path("hillshade.tif")
_SLOPE_RELATIVE_PATH = Path("slope.tif")
_ASPECT_RELATIVE_PATH = Path("aspect.tif")
_ROUGHNESS_RELATIVE_PATH = Path("roughness.tif")
_HORIZON_PATCH_SIZE = 128
_HORIZON_SAMPLES = 1440
_HORIZON_PATCH_PIXELS = _HORIZON_PATCH_SIZE * _HORIZON_PATCH_SIZE
_HORIZON_TOTAL_SAMPLES = _HORIZON_PATCH_PIXELS * _HORIZON_SAMPLES
_HORIZON_MAX_COMPRESSED_BYTES = 2 * _HORIZON_SAMPLES
_HORIZON_MIN_ELEVATION_DEG = -50.0
_HORIZON_MAX_ELEVATION_DEG = 50.0
_HORIZON_SHORT_TO_ELEVATION_SCALE = _HORIZON_MAX_ELEVATION_DEG / 32767.0
_BODY_ANGULAR_DIAMETER_DEG = {
    "sun": 0.536,
    "earth": 2.0,
}
_BODY_PLOT_COLORS = {
    "sun": "gold",
    "earth": "blue",
}


def _validate_center_azimuth(center_azimuth: float) -> float:
    try:
        center = float(center_azimuth)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ScenarioPathError(
            "Center azimuth must be a finite number of degrees.",
            code="scenario_invalid_center_azimuth",
            details={"center_azimuth": center_azimuth},
        ) from exc
    if not np.isfinite(center):
        raise ScenarioPathError(
            "Center azimuth must be a finite number of degrees.",
            code="scenario_invalid_center_azimuth",
            details={"center_azimuth": center_azimuth},
        )
    return center


def _azimuth_label(value: float) -> str:
    wrapped = value % 360.0
    cardinals = ((0.0, "N"), (90.0, "E"), (180.0, "S"), (270.0, "W"))
    for cardinal_value, label in cardinals:
        if abs(wrapped - cardinal_value) < 1e-9:
            return label
    return f"{wrapped:g} deg"


def _azimuth_window(
    ax: Any,
    center_azimuth: float | None,
) -> tuple[float, float, float]:
    if center_azimuth is None:
        left, right = ax.get_xlim()
        center = (float(left) + float(right)) / 2.0
        return float(left), float(right), center
    center = _validate_center_azimuth(center_azimuth)
    return center - 180.0, center + 180.0, center


def _wrap_azimuths_to_window(
    azimuths: np.ndarray,
    *,
    center: float,
) -> np.ndarray:
    return ((azimuths.astype(np.float64) - center + 180.0) % 360.0) + center - 180.0


def _with_wrap_breaks(
    x_values: np.ndarray,
    y_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if x_values.size < 2:
        return x_values, y_values
    break_indices = np.flatnonzero(np.abs(np.diff(x_values)) > 180.0) + 1
    if break_indices.size == 0:
        return x_values, y_values

    x_parts = []
    y_parts = []
    start = 0
    for stop in break_indices:
        x_parts.append(x_values[start:stop])
        y_parts.append(y_values[start:stop])
        x_parts.append(np.asarray([np.nan], dtype=np.float64))
        y_parts.append(np.asarray([np.nan], dtype=np.float64))
        start = int(stop)
    x_parts.append(x_values[start:])
    y_parts.append(y_values[start:])
    return np.concatenate(x_parts), np.concatenate(y_parts)


def _body_plot_color(body: BodyName | str) -> str | None:
    return _BODY_PLOT_COLORS.get(str(body).strip().lower())


def _body_sequence(bodies: Sequence[BodyName] | BodyName) -> tuple[BodyName, ...]:
    if isinstance(bodies, str):
        normalized = (bodies,)
    else:
        normalized = tuple(bodies)
    if not normalized:
        raise ScenarioPathError(
            "At least one body must be provided.",
            code="scenario_body_list_empty",
        )
    return normalized


def _minimal_wrapped_azimuths(azimuths: np.ndarray) -> tuple[float, float, float]:
    values = np.asarray(azimuths, dtype=np.float64).ravel() % 360.0
    if values.size == 0:
        raise ScenarioPathError(
            "At least one body position is required for zoomed body path plotting.",
            code="scenario_body_path_empty",
        )
    if values.size == 1:
        center = float(values[0])
        wrapped = _wrap_azimuths_to_window(values, center=center)
        return center, float(wrapped[0]), float(wrapped[0])

    sorted_values = np.sort(values)
    gaps = np.diff(np.concatenate([sorted_values, sorted_values[:1] + 360.0]))
    gap_index = int(np.argmax(gaps))
    start = float(sorted_values[(gap_index + 1) % values.size])
    span = float(360.0 - gaps[gap_index])
    center = start + span / 2.0
    wrapped = _wrap_azimuths_to_window(values, center=center)
    return center, float(np.min(wrapped)), float(np.max(wrapped))


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


@dataclass(slots=True)
class Scenario:
    """Filesystem-only access to standard paths inside one scenario root."""

    root: Path
    _horizon_file_handle: BinaryIO | None = field(default=None, init=False, repr=False)
    _horizon_file_path: Path | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.root = _resolved_scenario_root(self.root)

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

    def root_path(self) -> Path:
        """Return the resolved scenario root directory."""

        return self.root

    def dem_path(self) -> Path:
        """Return the canonical primary DEM path (``dem.tif``)."""

        return self.path(_PRIMARY_DEM_RELATIVE_PATH)

    def horizons_path(self) -> Path:
        """Return the canonical horizon-tile directory path."""

        return self.path(_HORIZONS_RELATIVE_PATH)

    def hillshade_path(self) -> Path:
        """Return the canonical native hillshade product path."""

        return self.path(_HILLSHADE_RELATIVE_PATH)

    def slope_path(self) -> Path:
        """Return the canonical native slope product path."""

        return self.path(_SLOPE_RELATIVE_PATH)

    def aspect_path(self) -> Path:
        """Return the canonical native aspect product path."""

        return self.path(_ASPECT_RELATIVE_PATH)

    def roughness_path(self) -> Path:
        """Return the canonical native roughness product path."""

        return self.path(_ROUGHNESS_RELATIVE_PATH)

    def output_path(self, relative_path: str | Path) -> Path:
        """Resolve a non-empty scenario-relative output path without creating it."""

        return self._resolve_relative(relative_path, allow_root=False)

    def lightmap(
        self,
        output: str | Path,
        **kwargs: Any,
    ) -> Path:
        """Generate a public Python lightmap at a scenario-relative path."""

        from .products import generate_lightmap

        return generate_lightmap(
            self.dem_path(),
            self.horizons_path(),
            self.output_path(output),
            **kwargs,
        )

    def sun_elevation(self, output: str | Path, **kwargs: Any) -> Path:
        """Generate public Sun terrain-relative elevation bands."""

        from .products import generate_sun_elevation

        return generate_sun_elevation(
            self.dem_path(),
            self.horizons_path(),
            self.output_path(output),
            **kwargs,
        )

    def earth_elevation(self, output: str | Path, **kwargs: Any) -> Path:
        """Generate public Earth terrain-relative elevation bands."""

        from .products import generate_earth_elevation

        return generate_earth_elevation(
            self.dem_path(),
            self.horizons_path(),
            self.output_path(output),
            **kwargs,
        )

    def safe_havens(self, output: str | Path, **kwargs: Any) -> Path:
        """Generate public safe-haven duration bands."""

        from .products import generate_safe_havens

        return generate_safe_havens(
            self.dem_path(),
            self.horizons_path(),
            self.output_path(output),
            **kwargs,
        )

    def mission_duration_from_sunlight(
        self, output: str | Path, **kwargs: Any
    ) -> Path:
        from .products import mission_duration_from_sunlight

        return mission_duration_from_sunlight(
            self.dem_path(), self.horizons_path(), self.output_path(output), **kwargs
        )

    def mission_duration_from_sun_elevation(
        self, output: str | Path, **kwargs: Any
    ) -> Path:
        from .products import mission_duration_from_sun_elevation

        return mission_duration_from_sun_elevation(
            self.dem_path(), self.horizons_path(), self.output_path(output), **kwargs
        )

    def mission_duration_from_sunlight_and_earth(
        self, output: str | Path, **kwargs: Any
    ) -> Path:
        from .products import mission_duration_from_sunlight_and_earth

        return mission_duration_from_sunlight_and_earth(
            self.dem_path(), self.horizons_path(), self.output_path(output), **kwargs
        )

    def mission_duration_from_sun_and_earth_elevation(
        self, output: str | Path, **kwargs: Any
    ) -> Path:
        from .products import mission_duration_from_sun_and_earth_elevation

        return mission_duration_from_sun_and_earth_elevation(
            self.dem_path(), self.horizons_path(), self.output_path(output), **kwargs
        )

    def _resolve_dem_input_path(self, path: str | Path) -> Path:
        candidate = Path(path).expanduser()
        if candidate.is_absolute():
            return candidate.resolve()
        return self.path(candidate)

    def generate_horizons(
        self,
        *,
        dem_paths: Sequence[str | Path] | None = None,
        surrounding_dems: Sequence[str | Path] | None = None,
        observer_elevation: float = 0.0,
        skip_existing: bool = True,
        compress_horizons: bool = True,
        disable_hierarchy: bool = False,
        progress_callback: Any | None = None,
        cancellation_requested: Any | None = None,
        _generator: Any | None = None,
    ) -> Path:
        """Generate horizon files into the scenario's canonical horizons directory."""

        if dem_paths is None:
            surrounding = surrounding_dems or ()
            resolved_dem_paths = [
                self.dem_path(),
                *(self._resolve_dem_input_path(path) for path in surrounding),
            ]
        else:
            if surrounding_dems is not None:
                raise NativeInputError(
                    "Pass either dem_paths or surrounding_dems, not both.",
                    code="native_horizons_ambiguous_dem_paths",
                )
            resolved_dem_paths = [
                self._resolve_dem_input_path(path) for path in dem_paths
            ]

        if _generator is None:
            from .native_horizon import GenerateHorizons

            generator = GenerateHorizons
        else:
            generator = _generator

        return generator(
            self.horizons_path(),
            resolved_dem_paths,
            observer_elevation=observer_elevation,
            skip_existing=skip_existing,
            compress_horizons=compress_horizons,
            disable_hierarchy=disable_hierarchy,
            progress_callback=progress_callback,
            cancellation_requested=cancellation_requested,
        )

    def _native_terrain_product(
        self,
        kind: str,
        output: Path,
        *,
        overwrite: bool,
        _terrain_products: Any | None = None,
    ) -> Path:
        from .native_terrain import generate_terrain_product

        return generate_terrain_product(
            kind,  # type: ignore[arg-type]
            dem_path=self.dem_path(),
            output_path=output,
            overwrite=overwrite,
            _terrain_products=_terrain_products,
        )

    def create_hillshade(
        self,
        *,
        overwrite: bool = False,
        _terrain_products: Any | None = None,
    ) -> Path:
        """Create the canonical native GDAL hillshade product."""

        return self._native_terrain_product(
            "hillshade",
            self.hillshade_path(),
            overwrite=overwrite,
            _terrain_products=_terrain_products,
        )

    def create_slope(
        self,
        *,
        overwrite: bool = False,
        _terrain_products: Any | None = None,
    ) -> Path:
        """Create the canonical native GDAL slope product."""

        return self._native_terrain_product(
            "slope",
            self.slope_path(),
            overwrite=overwrite,
            _terrain_products=_terrain_products,
        )

    def create_aspect(
        self,
        *,
        overwrite: bool = False,
        _terrain_products: Any | None = None,
    ) -> Path:
        """Create the canonical native GDAL aspect product."""

        return self._native_terrain_product(
            "aspect",
            self.aspect_path(),
            overwrite=overwrite,
            _terrain_products=_terrain_products,
        )

    def create_roughness(
        self,
        *,
        overwrite: bool = False,
        _terrain_products: Any | None = None,
    ) -> Path:
        """Create the canonical native GDAL roughness product."""

        return self._native_terrain_product(
            "roughness",
            self.roughness_path(),
            overwrite=overwrite,
            _terrain_products=_terrain_products,
        )

    @staticmethod
    def _validate_dem_pixel_coordinate(value: int, name: str) -> int:
        try:
            coordinate = int(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ScenarioPathError(
                "DEM pixel coordinates must be non-negative integers.",
                code="scenario_invalid_dem_pixel_coordinate",
                details={name: value},
            ) from exc
        if coordinate != value or coordinate < 0:
            raise ScenarioPathError(
                "DEM pixel coordinates must be non-negative integers.",
                code="scenario_invalid_dem_pixel_coordinate",
                details={name: value},
            )
        return coordinate

    @staticmethod
    def _validate_observer_height_decimeters(value: int) -> int:
        try:
            height = int(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ScenarioPathError(
                "Observer height must be an integer number of decimeters.",
                code="scenario_invalid_observer_height",
                details={"observer_height_decimeters": value},
            ) from exc
        if height != value or height < 0 or height > 999:
            raise ScenarioPathError(
                "Observer height must fit the D3 horizon filename field.",
                code="scenario_invalid_observer_height",
                details={"observer_height_decimeters": value},
            )
        return height

    def horizon_patch_pixel(self, x: int, y: int) -> tuple[int, int]:
        """Return ``(x, y)`` local coordinates within the 128x128 horizon patch."""

        pixel_x = self._validate_dem_pixel_coordinate(x, "x")
        pixel_y = self._validate_dem_pixel_coordinate(y, "y")
        return pixel_x % _HORIZON_PATCH_SIZE, pixel_y % _HORIZON_PATCH_SIZE

    def horizon_patch_row_col(self, x: int, y: int) -> tuple[int, int]:
        """Return ``(row, col)`` of the horizon patch containing a DEM pixel."""

        pixel_x = self._validate_dem_pixel_coordinate(x, "x")
        pixel_y = self._validate_dem_pixel_coordinate(y, "y")
        return pixel_y // _HORIZON_PATCH_SIZE, pixel_x // _HORIZON_PATCH_SIZE

    def horizon_file_path(
        self,
        x: int,
        y: int,
        observer_height_decimeters: int,
    ) -> Path | None:
        """Return the existing horizon file path for a DEM pixel, preferring ``.cbin``."""

        patch_row, patch_col = self.horizon_patch_row_col(x, y)
        height_dm = self._validate_observer_height_decimeters(observer_height_decimeters)
        tile_y = patch_row * _HORIZON_PATCH_SIZE
        tile_x = patch_col * _HORIZON_PATCH_SIZE
        file_stem = f"horizon_{tile_y:05d}_{tile_x:05d}_{height_dm:03d}"
        y_directory = self.horizons_path() / f"{tile_y:05d}"
        for path in (
            y_directory / f"{file_stem}.cbin",
            y_directory / f"{file_stem}.bin",
            self.horizons_path() / f"{file_stem}.cbin",
            self.horizons_path() / f"{file_stem}.bin",
        ):
            if path.is_file():
                return path
        return None

    @staticmethod
    def _decode_compressed_horizon_block(encoded: bytes) -> np.ndarray:
        if len(encoded) < 2:
            raise NativeProductError(
                "Compressed horizon block is too short.",
                code="horizon_block_decode_failed",
            )
        values = np.empty(_HORIZON_SAMPLES, dtype=np.float32)
        acc = struct.unpack(">h", encoded[:2])[0]
        values[0] = acc * _HORIZON_SHORT_TO_ELEVATION_SCALE
        read = 2
        written = 1
        while read < len(encoded) and written < _HORIZON_SAMPLES:
            first = encoded[read]
            read += 1
            if (first & 0x80) == 0:
                delta = first & 0x7F
                if delta & 0x40:
                    delta -= 0x80
            else:
                if read >= len(encoded):
                    break
                low = encoded[read]
                read += 1
                high = ((first << 1) & 0x80) | (first & 0x7F)
                delta = struct.unpack(">h", bytes((high, low)))[0]
            acc = ((acc + delta + 32768) % 65536) - 32768
            values[written] = acc * _HORIZON_SHORT_TO_ELEVATION_SCALE
            written += 1
        if written != _HORIZON_SAMPLES:
            raise NativeProductError(
                "Compressed horizon block did not decode to 1440 samples.",
                code="horizon_block_decode_failed",
                details={"decoded_samples": written},
            )
        return values

    @staticmethod
    def horizon_from_open_file(
        file_handle: BinaryIO,
        patch_x: int,
        patch_y: int,
    ) -> np.ndarray:
        """Read one pixel horizon from an open ``.bin`` or ``.cbin`` horizon file."""

        local_x = Scenario._validate_dem_pixel_coordinate(patch_x, "patch_x")
        local_y = Scenario._validate_dem_pixel_coordinate(patch_y, "patch_y")
        if local_x >= _HORIZON_PATCH_SIZE or local_y >= _HORIZON_PATCH_SIZE:
            raise ScenarioPathError(
                "Horizon patch coordinates must be within a 128x128 patch.",
                code="scenario_horizon_patch_coordinate_out_of_range",
                details={"patch_x": patch_x, "patch_y": patch_y},
            )

        pixel_index = local_y * _HORIZON_PATCH_SIZE + local_x
        file_name = str(getattr(file_handle, "name", "")).lower()
        if file_name.endswith(".bin"):
            byte_offset = pixel_index * _HORIZON_SAMPLES * 4
            file_handle.seek(byte_offset, os.SEEK_SET)
            data = file_handle.read(_HORIZON_SAMPLES * 4)
            if len(data) != _HORIZON_SAMPLES * 4:
                raise NativeProductError(
                    "Uncompressed horizon file ended before the requested horizon.",
                    code="horizon_file_read_failed",
                )
            return np.frombuffer(data, dtype="<f4", count=_HORIZON_SAMPLES).astype(
                np.float32,
                copy=True,
            )

        if file_name.endswith(".cbin"):
            file_handle.seek(0, os.SEEK_SET)
            for horizon_index in range(pixel_index + 1):
                length_data = file_handle.read(2)
                if len(length_data) != 2:
                    raise NativeProductError(
                        "Compressed horizon file ended while reading a block length.",
                        code="horizon_file_read_failed",
                        details={"horizon_index": horizon_index},
                    )
                encoded_len = int.from_bytes(length_data, "little", signed=False)
                if encoded_len <= 0 or encoded_len > _HORIZON_MAX_COMPRESSED_BYTES:
                    raise NativeProductError(
                        "Compressed horizon block length is invalid.",
                        code="horizon_file_read_failed",
                        details={"horizon_index": horizon_index, "encoded_length": encoded_len},
                    )
                if horizon_index == pixel_index:
                    encoded = file_handle.read(encoded_len)
                    if len(encoded) != encoded_len:
                        raise NativeProductError(
                            "Compressed horizon file ended while reading encoded data.",
                            code="horizon_file_read_failed",
                            details={"horizon_index": horizon_index},
                        )
                    return Scenario._decode_compressed_horizon_block(encoded)
                file_handle.seek(encoded_len, os.SEEK_CUR)

        raise NativeInputError(
            "Horizon file must have a .bin or .cbin extension.",
            code="horizon_file_unsupported_extension",
            details={"path": str(getattr(file_handle, "name", ""))},
        )

    def horizon_for_pixel(
        self,
        x: int,
        y: int,
        observer_height_decimeters: int,
    ) -> np.ndarray | None:
        """Fetch one DEM pixel's horizon, caching one open horizon tile file."""

        path = self.horizon_file_path(x, y, observer_height_decimeters)
        if path is None:
            return None
        if self._horizon_file_path != path or self._horizon_file_handle is None:
            self.close_horizon_file()
            self._horizon_file_handle = path.open("rb")
            self._horizon_file_path = path
        patch_x, patch_y = self.horizon_patch_pixel(x, y)
        return self.horizon_from_open_file(self._horizon_file_handle, patch_x, patch_y)

    def close_horizon_file(self) -> None:
        """Close the cached horizon tile file handle, if one is open."""

        handle = self._horizon_file_handle
        self._horizon_file_handle = None
        self._horizon_file_path = None
        if handle is not None:
            handle.close()

    def lonlat_to_dem_pixel(self, point: LonLat) -> tuple[float, float]:
        """Return the floating-point ``(x, y)`` DEM pixel position for a lon/lat."""

        _values, georef = read_geotiff(self.dem_path())
        if georef is None:
            raise GeoTiffMetadataError(
                "The scenario DEM is not georeferenced.",
                code="scenario_dem_not_georeferenced",
                details={"dem_path": str(self.dem_path())},
            )
        x, y = georef.lonlat_to_pixel(point.longitude, point.latitude)
        return float(x), float(y)

    @staticmethod
    def _configure_azimuth_elevation_axis(
        ax: Any,
        *,
        center_azimuth: float,
        elevation_limits: tuple[float, float] | None,
        grid: bool,
    ) -> None:
        center = _validate_center_azimuth(center_azimuth)
        ticks = [center + offset for offset in (-180.0, -90.0, 0.0, 90.0, 180.0)]
        ax.set_xlim(center - 180.0, center + 180.0)
        if elevation_limits is not None:
            ax.set_ylim(float(elevation_limits[0]), float(elevation_limits[1]))
        ax.set_xticks(ticks)
        ax.set_xticklabels([_azimuth_label(tick) for tick in ticks])
        ax.set_xlabel("Azimuth")
        ax.set_ylabel("Elevation (deg)")
        ax.grid(bool(grid))

    def plot_azimuth_elevation_axes(
        self,
        *,
        center_azimuth: float = 0.0,
        elevation_limits: tuple[float, float] | None = (-90.0, 90.0),
        grid: bool = True,
    ):
        """Create an empty azimuth/elevation plot using the horizon convention."""

        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        self._configure_azimuth_elevation_axis(
            ax,
            center_azimuth=center_azimuth,
            elevation_limits=elevation_limits,
            grid=grid,
        )
        return fig, ax

    def plot_horizon(
        self,
        point: LonLat,
        *,
        observer_height_decimeters: int = 0,
        center_azimuth: float = 0.0,
        grid: bool = True,
    ):
        """Plot the stored horizon for the DEM pixel nearest a lon/lat point."""

        x, y = self.lonlat_to_dem_pixel(point)
        pixel_x = int(round(x))
        pixel_y = int(round(y))
        horizon = self.horizon_for_pixel(
            pixel_x,
            pixel_y,
            observer_height_decimeters,
        )
        if horizon is None:
            raise NativeProductError(
                "No horizon file exists for the requested lon/lat point.",
                code="scenario_horizon_file_missing",
                details={
                    "longitude": point.longitude,
                    "latitude": point.latitude,
                    "x": x,
                    "y": y,
                    "pixel_x": pixel_x,
                    "pixel_y": pixel_y,
                    "observer_height_decimeters": observer_height_decimeters,
                },
            )

        sample_azimuth_degrees = np.arange(_HORIZON_SAMPLES, dtype=np.float32) * (
            360.0 / _HORIZON_SAMPLES
        )
        center = _validate_center_azimuth(center_azimuth)
        plot_azimuth_degrees = (
            (sample_azimuth_degrees.astype(np.float64) - center + 180.0) % 360.0
        ) + center - 180.0
        order = np.argsort(plot_azimuth_degrees)

        fig, ax = self.plot_azimuth_elevation_axes(
            center_azimuth=center,
            elevation_limits=None,
            grid=grid,
        )
        ax.plot(plot_azimuth_degrees[order], horizon[order])
        ax.set_ylabel("Horizon elevation (deg)")
        return fig, ax

    def body_azimuth_elevation_over_horizon(
        self,
        point: LonLat,
        body: BodyName,
        times: Any,
        *,
        observer_height_decimeters: int = 0,
        ensure_kernels: bool = True,
    ) -> np.ndarray:
        """Return body azimuth and elevation above the horizon at ``point``."""

        dem_x, dem_y = self.lonlat_to_dem_pixel(point)
        pixel_x = int(round(dem_x))
        pixel_y = int(round(dem_y))
        horizon = self.horizon_for_pixel(
            pixel_x,
            pixel_y,
            observer_height_decimeters,
        )
        if horizon is None:
            raise NativeProductError(
                "No horizon file exists for the requested lon/lat point.",
                code="scenario_horizon_file_missing",
                details={
                    "longitude": point.longitude,
                    "latitude": point.latitude,
                    "x": dem_x,
                    "y": dem_y,
                    "pixel_x": pixel_x,
                    "pixel_y": pixel_y,
                    "observer_height_decimeters": observer_height_decimeters,
                },
            )
        return body_azimuth_elevation_over_horizon(
            point,
            body,
            times,
            horizon,
            ensure_kernels=ensure_kernels,
        )

    def plot_body_elevations(
        self,
        point: LonLat,
        bodies: Sequence[BodyName],
        times: Any,
        *,
        horizon: np.ndarray | None = None,
        over_horizon: bool = False,
        observer_height_decimeters: int = 0,
        grid: bool = True,
        ensure_kernels: bool = True,
    ):
        """Plot body elevations, optionally relative to the scenario horizon."""

        plot_horizon = horizon
        if plot_horizon is None and over_horizon:
            dem_x, dem_y = self.lonlat_to_dem_pixel(point)
            pixel_x = int(round(dem_x))
            pixel_y = int(round(dem_y))
            plot_horizon = self.horizon_for_pixel(
                pixel_x,
                pixel_y,
                observer_height_decimeters,
            )
            if plot_horizon is None:
                raise NativeProductError(
                    "No horizon file exists for the requested lon/lat point.",
                    code="scenario_horizon_file_missing",
                    details={
                        "longitude": point.longitude,
                        "latitude": point.latitude,
                        "x": dem_x,
                        "y": dem_y,
                        "pixel_x": pixel_x,
                        "pixel_y": pixel_y,
                        "observer_height_decimeters": observer_height_decimeters,
                    },
                )
        return plot_body_elevations(
            point,
            bodies,
            times,
            horizon=plot_horizon,
            grid=grid,
            ensure_kernels=ensure_kernels,
        )

    @staticmethod
    def _body_angular_diameter(body: BodyName | str) -> float:
        key = str(body).strip().lower()
        if key not in _BODY_ANGULAR_DIAMETER_DEG:
            # Let the SPICE helper raise its structured unsupported-body error.
            return 0.0
        return _BODY_ANGULAR_DIAMETER_DEG[key]

    def plot_body_position(
        self,
        ax: Any,
        point: LonLat,
        body: BodyName,
        time: Any,
        *,
        style: str = "center",
        center_azimuth: float | None = None,
        label: str | None = None,
        ensure_kernels: bool = True,
        **plot_kwargs: Any,
    ):
        """Overlay the center point or apparent limb of the Sun or Earth."""

        _left, _right, center = _azimuth_window(ax, center_azimuth)
        if center_azimuth is not None:
            self._configure_azimuth_elevation_axis(
                ax,
                center_azimuth=center,
                elevation_limits=None,
                grid=any(line.get_visible() for line in ax.get_xgridlines()),
            )
        angles = body_azimuth_elevation(
            point,
            body,
            [time],
            ensure_kernels=ensure_kernels,
        )
        azimuth = float(_wrap_azimuths_to_window(angles[:, 0], center=center)[0])
        elevation = float(angles[0, 1])
        style_key = str(style).strip().lower()
        default_color = _body_plot_color(body)
        if style_key == "center":
            kwargs = {"marker": "o", "linestyle": "None"}
            if default_color is not None:
                kwargs["color"] = default_color
            kwargs.update(plot_kwargs)
            (artist,) = ax.plot([azimuth], [elevation], label=label, **kwargs)
            return artist
        if style_key == "limb":
            from matplotlib.patches import Ellipse

            diameter = self._body_angular_diameter(body)
            patch_kwargs = {"fill": False}
            if default_color is not None:
                patch_kwargs["edgecolor"] = default_color
            patch_kwargs.update(plot_kwargs)
            if label is not None:
                patch_kwargs["label"] = label
            ellipse = Ellipse(
                (azimuth, elevation),
                width=diameter,
                height=diameter,
                **patch_kwargs,
            )
            ax.add_patch(ellipse)
            return ellipse
        raise ScenarioPathError(
            "Body position style must be 'center' or 'limb'.",
            code="scenario_invalid_body_plot_style",
            details={"style": style},
        )

    def plot_body_path(
        self,
        ax: Any,
        point: LonLat,
        body: BodyName,
        times: Any,
        *,
        style: str = "center",
        center_azimuth: float | None = None,
        label: str | None = None,
        ensure_kernels: bool = True,
        **plot_kwargs: Any,
    ):
        """Overlay center and/or limb paths of the Sun or Earth."""

        _left, _right, center = _azimuth_window(ax, center_azimuth)
        if center_azimuth is not None:
            self._configure_azimuth_elevation_axis(
                ax,
                center_azimuth=center,
                elevation_limits=None,
                grid=any(line.get_visible() for line in ax.get_xgridlines()),
            )
        angles = body_azimuth_elevation(
            point,
            body,
            times,
            ensure_kernels=ensure_kernels,
        )
        x_values = _wrap_azimuths_to_window(angles[:, 0], center=center)
        style_key = str(style).strip().lower()
        default_color = _body_plot_color(body)
        artists = []
        line_kwargs = dict(plot_kwargs)
        if default_color is not None and "color" not in line_kwargs and "c" not in line_kwargs:
            line_kwargs["color"] = default_color

        def plot_line(y_values: np.ndarray, line_label: str | None, **kwargs: Any) -> None:
            x_plot, y_plot = _with_wrap_breaks(x_values, y_values.astype(np.float64))
            (artist,) = ax.plot(x_plot, y_plot, label=line_label, **kwargs)
            artists.append(artist)

        def fill_limb_band(
            lower_values: np.ndarray,
            upper_values: np.ndarray,
            band_label: str | None,
            **kwargs: Any,
        ) -> None:
            x_plot, lower_plot = _with_wrap_breaks(
                x_values,
                lower_values.astype(np.float64),
            )
            _x_plot, upper_plot = _with_wrap_breaks(
                x_values,
                upper_values.astype(np.float64),
            )
            fill_kwargs = dict(kwargs)
            fill_kwargs.setdefault("alpha", 0.5)
            fill_kwargs.setdefault("linewidth", 0)
            artist = ax.fill_between(
                x_plot,
                lower_plot,
                upper_plot,
                label=band_label,
                **fill_kwargs,
            )
            artists.append(artist)

        if style_key in {"center", "center_and_limbs"}:
            plot_line(angles[:, 1], label, **line_kwargs)
        if style_key in {"limbs", "center_and_limbs"}:
            radius = self._body_angular_diameter(body) / 2.0
            band_label = label if style_key == "limbs" else None
            fill_limb_band(
                angles[:, 1] - radius,
                angles[:, 1] + radius,
                band_label,
                **line_kwargs,
            )
        if not artists:
            raise ScenarioPathError(
                "Body path style must be 'center', 'limbs', or 'center_and_limbs'.",
                code="scenario_invalid_body_plot_style",
                details={"style": style},
            )
        return artists[0] if len(artists) == 1 else tuple(artists)

    def plot_zoomed_body_path(
        self,
        point: LonLat,
        bodies: Sequence[BodyName] | BodyName,
        times: Any,
        *,
        observer_height_decimeters: int = 0,
        grid: bool = True,
        ensure_kernels: bool = True,
        margin_degrees: float = 1.0,
    ):
        """Plot body limb paths against the local horizon in a zoomed equal-scale view."""

        body_values = _body_sequence(bodies)
        try:
            margin = float(margin_degrees)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ScenarioPathError(
                "Margin must be a finite non-negative number of degrees.",
                code="scenario_invalid_zoom_margin",
                details={"margin_degrees": margin_degrees},
            ) from exc
        if not np.isfinite(margin) or margin < 0.0:
            raise ScenarioPathError(
                "Margin must be a finite non-negative number of degrees.",
                code="scenario_invalid_zoom_margin",
                details={"margin_degrees": margin_degrees},
            )

        dem_x, dem_y = self.lonlat_to_dem_pixel(point)
        pixel_x = int(round(dem_x))
        pixel_y = int(round(dem_y))
        horizon = self.horizon_for_pixel(
            pixel_x,
            pixel_y,
            observer_height_decimeters,
        )
        if horizon is None:
            raise NativeProductError(
                "No horizon file exists for the requested lon/lat point.",
                code="scenario_horizon_file_missing",
                details={
                    "longitude": point.longitude,
                    "latitude": point.latitude,
                    "x": dem_x,
                    "y": dem_y,
                    "pixel_x": pixel_x,
                    "pixel_y": pixel_y,
                    "observer_height_decimeters": observer_height_decimeters,
                },
            )

        body_angles: list[tuple[BodyName, np.ndarray]] = []
        all_azimuths = []
        y_mins = []
        y_maxs = []
        for body in body_values:
            angles = body_azimuth_elevation(
                point,
                body,
                times,
                ensure_kernels=ensure_kernels,
            )
            if angles.shape[0] < 1:
                raise ScenarioPathError(
                    "Body path plotting requires at least one time sample.",
                    code="scenario_body_path_empty",
                )
            radius = self._body_angular_diameter(body) / 2.0
            body_angles.append((body, angles))
            all_azimuths.append(angles[:, 0])
            y_mins.append(float(np.min(angles[:, 1] - radius)))
            y_maxs.append(float(np.max(angles[:, 1] + radius)))

        center, x_min, x_max = _minimal_wrapped_azimuths(np.concatenate(all_azimuths))
        sample_azimuth_degrees = np.arange(_HORIZON_SAMPLES, dtype=np.float64) * (
            360.0 / _HORIZON_SAMPLES
        )
        horizon_x = _wrap_azimuths_to_window(sample_azimuth_degrees, center=center)
        x_left = x_min - margin
        x_right = x_max + margin
        if x_left == x_right:
            x_left -= max(margin, 1.0)
            x_right += max(margin, 1.0)

        horizon_mask = (horizon_x >= x_left) & (horizon_x <= x_right)
        if not np.any(horizon_mask):
            nearest_index = int(
                np.argmin(np.abs(horizon_x - (x_left + x_right) / 2.0))
            )
            horizon_values_in_frame = np.asarray(
                [horizon[nearest_index]],
                dtype=np.float64,
            )
        else:
            horizon_values_in_frame = horizon[horizon_mask].astype(np.float64)
        y_bottom = float(np.min(horizon_values_in_frame)) - margin
        y_top = max(
            float(np.max(y_maxs)) + margin,
            float(np.max(horizon_values_in_frame)) + margin,
        )
        if y_bottom == y_top:
            y_bottom -= max(margin, 1.0)
            y_top += max(margin, 1.0)
        y_bottom = min(y_bottom, float(np.min(y_mins)) - margin)

        fig, ax = self.plot_azimuth_elevation_axes(
            center_azimuth=center,
            elevation_limits=None,
            grid=grid,
        )
        order = np.argsort(horizon_x)
        ax.plot(horizon_x[order], horizon[order], color="black", label="Horizon")
        ax.set_xlim(x_left, x_right)
        ax.set_ylim(y_bottom, y_top)
        ax.set_aspect("equal", adjustable="box")
        ax.set_ylabel("Elevation (deg)")

        from matplotlib.patches import Ellipse

        for body, angles in body_angles:
            color = _body_plot_color(body)
            x_values = _wrap_azimuths_to_window(angles[:, 0], center=center)
            radius = self._body_angular_diameter(body) / 2.0
            lower = angles[:, 1] - radius
            upper = angles[:, 1] + radius
            x_plot, lower_plot = _with_wrap_breaks(x_values, lower.astype(np.float64))
            _x_plot, upper_plot = _with_wrap_breaks(x_values, upper.astype(np.float64))
            fill_kwargs: dict[str, Any] = {"alpha": 0.5, "linewidth": 0}
            if color is not None:
                fill_kwargs["color"] = color
            ax.fill_between(
                x_plot,
                lower_plot,
                upper_plot,
                label=str(body),
                **fill_kwargs,
            )

            limb_kwargs: dict[str, Any] = {"fill": False, "alpha": 1.0}
            if color is not None:
                limb_kwargs["edgecolor"] = color
            initial_limb = Ellipse(
                (float(x_values[0]), float(angles[0, 1])),
                width=radius * 2.0,
                height=radius * 2.0,
                **limb_kwargs,
            )
            ax.add_patch(initial_limb)

        return fig, ax

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
        **kwargs: Any,
    ) -> Path:
        """Generate a public Python permanent-shadow classification product."""

        from .products import generate_psr

        horizons_path = self.horizons_path() if horizons is None else self.path(horizons)
        return generate_psr(
            self.dem_path(),
            horizons_path,
            self.output_path(output),
            **kwargs,
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
