"""NumPy contracts at the future Numba horizon host/device boundary.

This module contains no CUDA or Python.NET imports. It freezes storage, units,
precision boundaries, indexing, and validation before any algorithm is ported.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import numpy.typing as npt


HOST_FLOAT_DTYPE = np.dtype("<f8")
DEVICE_FLOAT_DTYPE = np.dtype("<f4")
DEVICE_INT_DTYPE = np.dtype("<i4")

MAP_PARAMETER_FIELDS = (
    "radius_m", "scale", "false_easting_m", "false_northing_m",
    "inverse_transform_determinant", "transform_0", "transform_1",
    "transform_2", "transform_3", "transform_4", "transform_5",
)
PROJECTION_PARAMETER_FIELDS = (
    "radius_m", "latitude_origin_rad", "longitude_origin_rad", "scale",
    "false_easting_m", "false_northing_m",
)
LEVEL_METADATA_FIELDS = ("level", "offset", "width", "height")
SEGMENT_FIELDS = (
    "start_pixel_x", "start_pixel_y", "x0", "y0", "a1", "a2", "a3", "a4",
    "b1", "b2", "b3", "b4", "s_start_km", "s_end_km",
    "s_start_chord_km", "planar_to_chord_c1", "planar_to_chord_c2",
    "planar_to_chord_c3",
)
KERNEL_FLOAT_FIELDS = (
    "observer_elevation_m", "minimum_traverse_distance_km", "gamma_center_rad",
    "d_gamma_dx_rad_per_pixel", "d_gamma_dy_rad_per_pixel",
)
KERNEL_INT_FIELDS = (
    "debug_azimuth_index", "debug_flags", "primary_width", "primary_height",
)
SUPPORTED_SUBPATCH_SIZES = frozenset((2, 4, 8, 16, 32, 64, 128))
SLOPE_SENTINEL = np.float32(-np.inf)


class ContractValidationError(ValueError):
    """Raised before CUDA initialization when an array contract is invalid."""


def _require_array(
    value: npt.NDArray[Any],
    *,
    name: str,
    dtype: np.dtype[Any],
    ndim: int,
    shape: tuple[int | None, ...] | None = None,
    finite: bool = False,
) -> None:
    if not isinstance(value, np.ndarray):
        raise ContractValidationError(f"{name} must be a NumPy array")
    if value.dtype != dtype:
        raise ContractValidationError(
            f"{name} must have dtype {dtype.str}, got {value.dtype.str}"
        )
    if value.ndim != ndim:
        raise ContractValidationError(f"{name} must have {ndim} dimensions")
    if shape is not None and any(
        expected is not None and actual != expected
        for actual, expected in zip(value.shape, shape, strict=True)
    ):
        raise ContractValidationError(f"{name} must have shape {shape}, got {value.shape}")
    if not value.flags.c_contiguous:
        raise ContractValidationError(f"{name} must be C-contiguous")
    if finite and not np.all(np.isfinite(value)):
        raise ContractValidationError(f"{name} must contain only finite values")


def host_float64(values: Any, *, name: str) -> npt.NDArray[np.float64]:
    """Make an explicit, finite, C-contiguous host-geometry conversion."""
    result = np.ascontiguousarray(values, dtype=HOST_FLOAT_DTYPE)
    _require_array(result, name=name, dtype=HOST_FLOAT_DTYPE, ndim=result.ndim, finite=True)
    return result


def device_float32(values: Any, *, name: str) -> npt.NDArray[np.float32]:
    """Make the sole explicit conversion from host values to device floats."""
    result = np.ascontiguousarray(values, dtype=DEVICE_FLOAT_DTYPE)
    _require_array(result, name=name, dtype=DEVICE_FLOAT_DTYPE, ndim=result.ndim, finite=True)
    return result


def device_int32(values: Any, *, name: str) -> npt.NDArray[np.int32]:
    """Convert integral values to the device integer dtype without truncation."""
    source = np.asarray(values)
    if not np.all(np.isfinite(source)) or not np.all(source == np.floor(source)):
        raise ContractValidationError(f"{name} must contain finite integers")
    bounds = np.iinfo(np.int32)
    if source.size and (np.min(source) < bounds.min or np.max(source) > bounds.max):
        raise ContractValidationError(f"{name} values exceed int32 bounds")
    result = np.ascontiguousarray(source, dtype=DEVICE_INT_DTYPE)
    _require_array(result, name=name, dtype=DEVICE_INT_DTYPE, ndim=result.ndim)
    return result


@dataclass(frozen=True, slots=True)
class ContractConfiguration:
    tile_width: int
    tile_height: int
    azimuth_count: int
    subpatch_size: int
    dem_count: int
    primary_width: int
    primary_height: int

    def __post_init__(self) -> None:
        positive = {
            "tile_width": self.tile_width, "tile_height": self.tile_height,
            "azimuth_count": self.azimuth_count, "dem_count": self.dem_count,
            "primary_width": self.primary_width, "primary_height": self.primary_height,
        }
        for name, value in positive.items():
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ContractValidationError(f"{name} must be a positive integer")
        if self.subpatch_size not in SUPPORTED_SUBPATCH_SIZES:
            raise ContractValidationError(
                f"subpatch_size must be one of {sorted(SUPPORTED_SUBPATCH_SIZES)}"
            )
        if 128 % self.subpatch_size:
            raise ContractValidationError("subpatch_size must evenly divide 128")
        if self.tile_width > 128 or self.tile_height > 128:
            raise ContractValidationError("tile dimensions cannot exceed 128 pixels")

    @property
    def pixel_count(self) -> int:
        return self.tile_width * self.tile_height

    @property
    def subpatches_per_dimension(self) -> int:
        # Deliberately mirrors C#, which derives both subpatch axes from tile width.
        return self.tile_width // self.subpatch_size + 2

    @property
    def subpatch_count(self) -> int:
        return self.subpatches_per_dimension**2

    @property
    def output_shape(self) -> tuple[int, int]:
        return self.pixel_count, self.azimuth_count


@dataclass(frozen=True, slots=True)
class KernelParameters:
    floats: npt.NDArray[np.float32]
    integers: npt.NDArray[np.int32]

    def __post_init__(self) -> None:
        _require_array(
            self.floats, name="kernel float parameters", dtype=DEVICE_FLOAT_DTYPE,
            ndim=1, shape=(len(KERNEL_FLOAT_FIELDS),), finite=True,
        )
        _require_array(
            self.integers, name="kernel integer parameters", dtype=DEVICE_INT_DTYPE,
            ndim=1, shape=(len(KERNEL_INT_FIELDS),),
        )
        if self.integers[2] <= 0 or self.integers[3] <= 0:
            raise ContractValidationError("primary dimensions must be positive")

    @classmethod
    def create(
        cls, *, observer_elevation_m: float, minimum_traverse_distance_km: float,
        gamma_center_rad: float, d_gamma_dx_rad_per_pixel: float,
        d_gamma_dy_rad_per_pixel: float, debug_azimuth_index: int,
        debug_flags: int, primary_width: int, primary_height: int,
    ) -> KernelParameters:
        return cls(
            device_float32(
                (observer_elevation_m, minimum_traverse_distance_km, gamma_center_rad,
                 d_gamma_dx_rad_per_pixel, d_gamma_dy_rad_per_pixel),
                name="kernel float parameters",
            ),
            device_int32(
                (debug_azimuth_index, debug_flags, primary_width, primary_height),
                name="kernel integer parameters",
            ),
        )


@dataclass(frozen=True, slots=True)
class SegmentTensor:
    values: npt.NDArray[np.float32]
    dem_ids: npt.NDArray[np.int32]
    configuration: ContractConfiguration

    def __post_init__(self) -> None:
        expected = (
            self.configuration.azimuth_count, self.configuration.subpatch_count,
            self.configuration.dem_count, len(SEGMENT_FIELDS),
        )
        _require_array(
            self.values, name="segment values", dtype=DEVICE_FLOAT_DTYPE,
            ndim=4, shape=expected, finite=True,
        )
        _require_array(
            self.dem_ids, name="segment DEM IDs", dtype=DEVICE_INT_DTYPE,
            ndim=3, shape=expected[:-1],
        )
        expected_ids = np.broadcast_to(
            np.arange(self.configuration.dem_count, dtype=DEVICE_INT_DTYPE),
            self.dem_ids.shape,
        )
        if not np.array_equal(self.dem_ids, expected_ids):
            raise ContractValidationError("segment DEM IDs do not match the DEM axis")

    def flat_index(self, azimuth: int, subpatch: int, dem: int) -> int:
        self._validate_index(azimuth, subpatch, dem)
        return ((azimuth * self.configuration.subpatch_count + subpatch)
                * self.configuration.dem_count + dem)

    def segment(self, azimuth: int, subpatch: int, dem: int) -> npt.NDArray[np.float32]:
        flat = self.flat_index(azimuth, subpatch, dem)
        return self.values.reshape(-1, len(SEGMENT_FIELDS))[flat]

    def interpolation_selection(
        self, pixel_column: int, pixel_row: int
    ) -> tuple[tuple[int, int, int, int], tuple[np.float32, np.float32]]:
        if not 0 <= pixel_column < self.configuration.tile_width:
            raise ContractValidationError("pixel_column is outside the tile")
        if not 0 <= pixel_row < self.configuration.tile_height:
            raise ContractValidationError("pixel_row is outside the tile")
        size = self.configuration.subpatch_size
        count = self.configuration.subpatches_per_dimension
        gx = (pixel_column - size / 2.0) / size + 1.0
        gy = (pixel_row - size / 2.0) / size + 1.0
        left, top = int(gx), int(gy)
        tx, ty = gx - left, gy - top
        if left < 0:
            left, tx = 0, 0.0
        if top < 0:
            top, ty = 0, 0.0
        if left > count - 2:
            left, tx = count - 2, 1.0
        if top > count - 2:
            top, ty = count - 2, 1.0
        right, bottom = left + 1, top + 1
        return (
            (top * count + left, top * count + right,
             bottom * count + left, bottom * count + right),
            (np.float32(tx), np.float32(ty)),
        )

    def interpolate(
        self, azimuth: int, pixel_column: int, pixel_row: int, dem: int
    ) -> npt.NDArray[np.float32]:
        indices, (tx, ty) = self.interpolation_selection(pixel_column, pixel_row)
        segments = [self.segment(azimuth, index, dem) for index in indices]
        top = segments[0] + (segments[1] - segments[0]) * tx
        bottom = segments[2] + (segments[3] - segments[2]) * tx
        return np.ascontiguousarray(top + (bottom - top) * ty, dtype=DEVICE_FLOAT_DTYPE)

    def _validate_index(self, azimuth: int, subpatch: int, dem: int) -> None:
        for name, value, stop in (
            ("azimuth", azimuth, self.configuration.azimuth_count),
            ("subpatch", subpatch, self.configuration.subpatch_count),
            ("dem", dem, self.configuration.dem_count),
        ):
            if not isinstance(value, int) or not 0 <= value < stop:
                raise ContractValidationError(f"{name} index is out of bounds")


@dataclass(frozen=True, slots=True)
class PyramidArrays:
    level0: npt.NDArray[np.float32]
    mips: npt.NDArray[np.float32]
    levels: npt.NDArray[np.int32]
    map_parameters: npt.NDArray[np.float32]
    projection_parameters: npt.NDArray[np.float32]

    def __post_init__(self) -> None:
        _require_array(self.level0, name="pyramid level 0", dtype=DEVICE_FLOAT_DTYPE, ndim=2)
        _require_array(self.mips, name="pyramid mips", dtype=DEVICE_FLOAT_DTYPE, ndim=1)
        _require_array(
            self.levels, name="pyramid level metadata", dtype=DEVICE_INT_DTYPE,
            ndim=2, shape=(None, len(LEVEL_METADATA_FIELDS)),
        )
        _require_array(
            self.map_parameters, name="map parameters", dtype=DEVICE_FLOAT_DTYPE,
            ndim=1, shape=(len(MAP_PARAMETER_FIELDS),), finite=True,
        )
        _require_array(
            self.projection_parameters, name="projection parameters",
            dtype=DEVICE_FLOAT_DTYPE, ndim=1,
            shape=(len(PROJECTION_PARAMETER_FIELDS),), finite=True,
        )
        if len(self.levels) == 0:
            raise ContractValidationError("pyramid must contain level 0")
        expected_offset = 0
        for index, (level, offset, width, height) in enumerate(self.levels):
            if level != index or width <= 0 or height <= 0:
                raise ContractValidationError("invalid pyramid level metadata")
            if index == 0:
                if offset != 0 or self.level0.shape != (height, width):
                    raise ContractValidationError("level-0 metadata does not match its array")
            else:
                if offset != expected_offset:
                    raise ContractValidationError("pyramid mip offsets are not contiguous")
                expected_offset += int(width * height)
        if expected_offset != self.mips.size:
            raise ContractValidationError("pyramid mip buffer length does not match metadata")

    @classmethod
    def from_artifact(
        cls, arrays: Mapping[str, npt.NDArray[Any]], prefix: str,
    ) -> PyramidArrays:
        levels = device_int32(
            arrays[f"{prefix}__level_metadata"],
            name="pyramid level metadata",
        )
        mip_parts = [
            np.ravel(arrays[f"{prefix}__level_{level}"], order="C")
            for level in range(1, len(levels))
        ]
        mips = (
            np.ascontiguousarray(np.concatenate(mip_parts), dtype=DEVICE_FLOAT_DTYPE)
            if mip_parts else np.empty(0, dtype=DEVICE_FLOAT_DTYPE)
        )
        return cls(
            np.ascontiguousarray(arrays[f"{prefix}__level_0"]), mips, levels,
            np.ascontiguousarray(arrays[f"{prefix}__map_parameters"]),
            np.ascontiguousarray(arrays[f"{prefix}__projection_parameters"]),
        )

    def cell(self, level: int, x: int, y: int) -> np.float32:
        if not isinstance(level, int) or not 0 <= level < len(self.levels):
            raise ContractValidationError("pyramid level is out of bounds")
        _, offset, width, height = (int(value) for value in self.levels[level])
        if not 0 <= x < width or not 0 <= y < height:
            raise ContractValidationError("pyramid cell is out of bounds")
        if level == 0:
            return self.level0[y, x]
        return self.mips[offset + y * width + x]


@dataclass(frozen=True, slots=True)
class HorizonBuffers:
    slopes: npt.NDArray[np.float32]

    def __post_init__(self) -> None:
        _require_array(self.slopes, name="slope buffer", dtype=DEVICE_FLOAT_DTYPE, ndim=2)
        if np.any(np.isnan(self.slopes)) or np.any(np.isposinf(self.slopes)):
            raise ContractValidationError(
                "slope buffer may only use negative infinity as a sentinel"
            )

    @classmethod
    def empty(cls, configuration: ContractConfiguration) -> HorizonBuffers:
        return cls(np.full(configuration.output_shape, SLOPE_SENTINEL, dtype=DEVICE_FLOAT_DTYPE))

    def degrees(self) -> npt.NDArray[np.float32]:
        """Convert slopes once, after every DEM pass has been merged."""
        return np.ascontiguousarray(
            np.degrees(np.arctan(self.slopes.astype(np.float64))),
            dtype=DEVICE_FLOAT_DTYPE,
        )

    def merge_pass(self, pass_slopes: npt.NDArray[np.float32]) -> HorizonBuffers:
        """Merge one DEM pass in slope space without mutating either input."""
        _require_array(
            pass_slopes,
            name="DEM-pass slope buffer",
            dtype=DEVICE_FLOAT_DTYPE,
            ndim=2,
            shape=self.slopes.shape,
        )
        if np.any(np.isnan(pass_slopes)) or np.any(np.isposinf(pass_slopes)):
            raise ContractValidationError(
                "DEM-pass slope buffer may only use negative infinity as a sentinel"
            )
        return HorizonBuffers(
            np.ascontiguousarray(np.maximum(self.slopes, pass_slopes))
        )


@dataclass(frozen=True, slots=True)
class ReferenceArtifact:
    metadata: Mapping[str, Any]
    arrays: Mapping[str, npt.NDArray[Any]]


def _array_sha256(array: npt.NDArray[Any]) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes(order="C")).hexdigest()


def load_reference_artifact(metadata_path: Path, npz_path: Path) -> ReferenceArtifact:
    """Load and validate every Phase 1 array without CUDA or Python.NET."""
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    manifest = metadata["arrays"]
    loaded: dict[str, npt.NDArray[Any]] = {}
    with np.load(npz_path, allow_pickle=False) as archive:
        if set(archive.files) != set(manifest):
            raise ContractValidationError("artifact array names do not match the manifest")
        for name, entry in manifest.items():
            array = np.ascontiguousarray(archive[name])
            if array.dtype.str != entry["dtype"] or list(array.shape) != entry["shape"]:
                raise ContractValidationError(f"artifact contract mismatch for {name}")
            if _array_sha256(array) != entry["sha256_c_order_data"]:
                raise ContractValidationError(f"artifact hash mismatch for {name}")
            loaded[name] = array
    return ReferenceArtifact(metadata=metadata, arrays=loaded)
