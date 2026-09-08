"""Public dynamic-environment provider contracts and in-memory adapters."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterable, Sequence
from datetime import datetime, timezone
from numbers import Integral
from pathlib import Path
from typing import Any, Protocol, TypeVar, runtime_checkable

import numpy as np
import numpy.typing as npt
from numpy.typing import NDArray

from ..alignment import same_grid
from ..errors import ConfigurationSpaceError, PlanningError, TrajectoryInputError
from ..georeference import GeoReference
from ..temporal import TimeRange, _parse_time
from ._time_contract import IntervalTimeAxis, as_utc


@runtime_checkable
class SunVectorProvider(Protocol):
    """Supply geometric Moon-to-Sun vectors in Moon-ME, in metres."""

    def vectors(
        self, times: Iterable[datetime] | TimeRange
    ) -> NDArray[np.float64]: ...


@runtime_checkable
class SunlightProvider(Protocol):
    """Supply solar-fraction bytes for UTC times and raster windows."""

    @property
    def georef(self) -> GeoReference: ...

    def read(
        self, x0: int, y0: int, width: int, height: int, time: datetime
    ) -> NDArray[np.uint8]: ...


@runtime_checkable
class EarthElevationProvider(Protocol):
    """Supply Earth elevation above the local terrain horizon in degrees."""

    @property
    def georef(self) -> GeoReference: ...

    def read(
        self, x0: int, y0: int, width: int, height: int, time: datetime
    ) -> NDArray[np.float32]: ...


@runtime_checkable
class ConfigurationSpaceProvider(Protocol):
    """Supply problem-specific hard occupancy constraints."""

    @property
    def georef(self) -> GeoReference: ...

    def read(
        self, x0: int, y0: int, width: int, height: int, time: datetime
    ) -> NDArray[np.bool_]: ...


_T = TypeVar("_T", bound=np.generic)
_HORIZON_TILE_SIZE = 128
_HORIZON_SHAPE = (128, 128, 1440)


def _time_values(times: Iterable[datetime] | TimeRange) -> tuple[datetime, ...]:
    if isinstance(times, TimeRange):
        return tuple(
            _parse_time(value, source_timezone=None).astimezone(timezone.utc)
            for value in times.values
        )
    try:
        values = tuple(times)
    except TypeError as exc:
        raise TrajectoryInputError(
            "times must be an iterable of timezone-aware datetimes.",
            code="trajectory_invalid_vector_times",
        ) from exc
    return tuple(
        as_utc(value, name=f"times[{index}]")
        for index, value in enumerate(values)
    )


def _window(
    georef: GeoReference,
    x0: int,
    y0: int,
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    values = (x0, y0, width, height)
    if any(
        isinstance(value, bool) or not isinstance(value, Integral)
        for value in values
    ):
        raise TrajectoryInputError(
            "Provider window coordinates and dimensions must be integers.",
            code="trajectory_invalid_provider_window",
            details={"window": list(values)},
        )
    resolved = tuple(int(value) for value in values)
    rx, ry, rw, rh = resolved
    if (
        rw <= 0
        or rh <= 0
        or rx < 0
        or ry < 0
        or rx + rw > georef.width
        or ry + rh > georef.height
    ):
        raise TrajectoryInputError(
            "Provider window must be non-empty and inside the provider grid.",
            code="trajectory_invalid_provider_window",
            details={
                "window": list(resolved),
                "grid_width": georef.width,
                "grid_height": georef.height,
            },
        )
    return rx, ry, rw, rh


def _readonly_copy(values: NDArray[_T]) -> NDArray[_T]:
    output = np.array(values, copy=True, order="C")
    output.flags.writeable = False
    return output


def _cache_size(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < 0:
        raise TrajectoryInputError(
            "cache_entries must be a non-negative integer.",
            code="trajectory_invalid_provider_cache",
            details={"cache_entries": value},
        )
    return int(value)


class _WindowCache:
    def __init__(self, maximum: int) -> None:
        self.maximum = _cache_size(maximum)
        self.values: OrderedDict[tuple[Any, ...], NDArray[Any]] = OrderedDict()

    def get(self, key: tuple[Any, ...]) -> NDArray[Any] | None:
        value = self.values.pop(key, None)
        if value is not None:
            self.values[key] = value
        return value

    def put(self, key: tuple[Any, ...], value: NDArray[Any]) -> None:
        if self.maximum == 0:
            return
        self.values.pop(key, None)
        self.values[key] = value
        while len(self.values) > self.maximum:
            self.values.popitem(last=False)

    def clear(self) -> None:
        self.values.clear()


class ExplicitSunVectorProvider:
    """Serve caller-supplied vectors by exact UTC timestamp."""

    def __init__(
        self,
        times: Iterable[datetime] | TimeRange,
        vectors_m: npt.ArrayLike,
    ) -> None:
        time_values = _time_values(times)
        vectors = np.array(vectors_m, dtype=np.float64, copy=True, order="C")
        if not time_values or vectors.shape != (len(time_values), 3):
            raise TrajectoryInputError(
                "Explicit Sun vectors must have shape (time, 3) and be non-empty.",
                code="trajectory_invalid_sun_vectors",
                details={"shape": list(vectors.shape), "time_count": len(time_values)},
            )
        if np.any(~np.isfinite(vectors)) or np.any(
            np.linalg.norm(vectors, axis=1) == 0.0
        ):
            raise TrajectoryInputError(
                "Explicit Sun vectors must be finite and non-zero.",
                code="trajectory_invalid_sun_vectors",
            )
        if len(set(time_values)) != len(time_values):
            raise TrajectoryInputError(
                "Explicit Sun-vector timestamps must be unique.",
                code="trajectory_invalid_vector_times",
            )
        vectors.flags.writeable = False
        self._vectors = {
            value: vectors[index] for index, value in enumerate(time_values)
        }

    def vectors(
        self, times: Iterable[datetime] | TimeRange
    ) -> NDArray[np.float64]:
        requested = _time_values(times)
        if not requested:
            raise TrajectoryInputError(
                "At least one Sun-vector timestamp is required.",
                code="trajectory_invalid_vector_times",
            )
        missing = [value.isoformat() for value in requested if value not in self._vectors]
        if missing:
            raise TrajectoryInputError(
                "No explicit Sun vector exists at a requested timestamp.",
                code="trajectory_sun_vector_time_not_found",
                details={"times": missing},
            )
        output = np.ascontiguousarray(
            [self._vectors[value] for value in requested], dtype=np.float64
        )
        output.flags.writeable = False
        return output


class SpiceSunVectorProvider:
    """Lazily generate and cache Moon-ME Sun vectors using Lunarscout SPICE."""

    def __init__(self, *, ensure_kernels: bool = True, cache_entries: int = 256) -> None:
        if not isinstance(ensure_kernels, (bool, np.bool_)):
            raise TrajectoryInputError(
                "ensure_kernels must be Boolean.",
                code="trajectory_invalid_vector_provider",
            )
        self.ensure_kernels = bool(ensure_kernels)
        self._cache = _WindowCache(cache_entries)
        self._closed = False

    def vectors(
        self, times: Iterable[datetime] | TimeRange
    ) -> NDArray[np.float64]:
        if self._closed:
            raise PlanningError(
                "The Sun-vector provider is closed.",
                code="trajectory_provider_closed",
            )
        requested = _time_values(times)
        if not requested:
            raise TrajectoryInputError(
                "At least one Sun-vector timestamp is required.",
                code="trajectory_invalid_vector_times",
            )
        resolved: dict[datetime, NDArray[np.float64]] = {}
        missing_values: list[datetime] = []
        for value in dict.fromkeys(requested):
            cached = self._cache.get((value,))
            if cached is None:
                missing_values.append(value)
            else:
                resolved[value] = cached
        missing = tuple(missing_values)
        if missing:
            try:
                from ..spice_geometry import body_vectors_moon_me

                generated = body_vectors_moon_me(
                    "sun", missing, ensure_kernels=self.ensure_kernels
                )
            except Exception as exc:
                raise PlanningError(
                    "Unable to obtain SPICE-backed Sun vectors.",
                    code="trajectory_sun_vector_generation_failed",
                    details={"error": str(exc)},
                ) from exc
            generated = np.asarray(generated)
            if (
                generated.dtype != np.dtype(np.float64)
                or generated.shape != (len(missing), 3)
                or np.any(~np.isfinite(generated))
                or np.any(np.linalg.norm(generated, axis=1) == 0.0)
            ):
                raise PlanningError(
                    "The SPICE Sun-vector provider returned invalid data.",
                    code="trajectory_invalid_provider_result",
                    details={
                        "shape": list(generated.shape),
                        "dtype": str(generated.dtype),
                    },
                )
            for index, value in enumerate(missing):
                vector = _readonly_copy(generated[index])
                resolved[value] = vector
                self._cache.put((value,), vector)
        output = np.ascontiguousarray(
            [
                resolved[value]
                if value in resolved
                else self._cache.get((value,))
                for value in requested
            ],
            dtype=np.float64,
        )
        output.flags.writeable = False
        return output

    def close(self) -> None:
        self._cache.clear()
        self._closed = True

    def __enter__(self) -> SpiceSunVectorProvider:
        if self._closed:
            raise PlanningError(
                "The Sun-vector provider is closed.",
                code="trajectory_provider_closed",
            )
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        self.close()


class _IntervalArrayProvider:
    _dtype: np.dtype[Any]
    _signal_name: str

    def __init__(
        self,
        boundaries: Iterable[datetime],
        values: npt.ArrayLike,
        georef: GeoReference,
        *,
        cache_entries: int = 32,
    ) -> None:
        if not isinstance(georef, GeoReference):
            raise TrajectoryInputError(
                "Provider georef must be a GeoReference.",
                code="trajectory_invalid_provider_grid",
            )
        try:
            boundary_values = tuple(boundaries)
        except TypeError as exc:
            raise TrajectoryInputError(
                "Provider boundaries must be an iterable of datetimes.",
                code="trajectory_invalid_dynamic_time",
            ) from exc
        axis = IntervalTimeAxis(boundary_values)
        array = np.asarray(values)
        expected = (axis.interval_count, georef.height, georef.width)
        if array.dtype != self._dtype or array.shape != expected:
            raise TrajectoryInputError(
                f"{self._signal_name} values have an invalid shape or dtype.",
                code="trajectory_invalid_provider_data",
                details={
                    "signal": self._signal_name,
                    "shape": list(array.shape),
                    "expected_shape": list(expected),
                    "dtype": str(array.dtype),
                    "expected_dtype": str(self._dtype),
                },
            )
        if np.issubdtype(self._dtype, np.floating) and np.any(~np.isfinite(array)):
            raise TrajectoryInputError(
                f"{self._signal_name} values must be finite.",
                code="trajectory_invalid_provider_data",
                details={"signal": self._signal_name},
            )
        self._values = _readonly_copy(array)
        self._axis = axis
        self._georef = georef
        self._cache = _WindowCache(cache_entries)
        self._closed = False

    @property
    def georef(self) -> GeoReference:
        return self._georef

    @property
    def boundaries(self) -> tuple[datetime, ...]:
        return self._axis.boundaries

    def _read(
        self, x0: int, y0: int, width: int, height: int, time: datetime
    ) -> NDArray[Any]:
        if self._closed:
            raise ConfigurationSpaceError(
                "The environment provider is closed.",
                code="trajectory_provider_closed",
                details={"signal": self._signal_name},
            )
        rx, ry, rw, rh = _window(self.georef, x0, y0, width, height)
        interval = self._axis.interval_index(time)
        key = (interval, rx, ry, rw, rh)
        cached = self._cache.get(key)
        if cached is None:
            cached = _readonly_copy(
                self._values[interval, ry : ry + rh, rx : rx + rw]
            )
            self._cache.put(key, cached)
        return _readonly_copy(cached)

    def read_many(
        self,
        x0: int,
        y0: int,
        width: int,
        height: int,
        times: Iterable[datetime],
    ) -> NDArray[Any]:
        try:
            requested = tuple(times)
        except TypeError as exc:
            raise TrajectoryInputError(
                "times must be an iterable of timezone-aware datetimes.",
                code="trajectory_invalid_dynamic_time",
            ) from exc
        if not requested:
            raise TrajectoryInputError(
                "At least one provider timestamp is required.",
                code="trajectory_invalid_dynamic_time",
            )
        output = np.stack(
            [self._read(x0, y0, width, height, value) for value in requested]
        )
        output.flags.writeable = False
        return output

    def close(self) -> None:
        self._cache.clear()
        self._closed = True

    def __enter__(self) -> _IntervalArrayProvider:
        if self._closed:
            raise ConfigurationSpaceError(
                "The environment provider is closed.",
                code="trajectory_provider_closed",
            )
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        self.close()


class ArraySunlightProvider(_IntervalArrayProvider):
    """Serve a copied ``uint8[interval, y, x]`` sunlight cube."""

    _dtype = np.dtype(np.uint8)
    _signal_name = "sunlight"

    def read(
        self, x0: int, y0: int, width: int, height: int, time: datetime
    ) -> NDArray[np.uint8]:
        return self._read(x0, y0, width, height, time)


class HorizonSunlightProvider:
    """Evaluate sunlight from an existing DEM and precomputed horizon store."""

    def __init__(
        self,
        dem_path: str | Path,
        horizons_path: str | Path,
        sun_vectors: SunVectorProvider,
        *,
        observer_elevation_m: float = 0.0,
        horizon_cache_entries: int = 1,
        window_cache_entries: int = 32,
        time_batch_size: int = 32,
    ) -> None:
        if not isinstance(sun_vectors, SunVectorProvider):
            raise TrajectoryInputError(
                "sun_vectors must conform to SunVectorProvider.",
                code="trajectory_invalid_vector_provider",
            )
        try:
            observer = float(observer_elevation_m)
        except (TypeError, ValueError, OverflowError) as exc:
            raise TrajectoryInputError(
                "observer_elevation_m must be finite, non-negative, and below 100.",
                code="trajectory_invalid_observer_elevation",
            ) from exc
        if not np.isfinite(observer) or not 0.0 <= observer < 100.0:
            raise TrajectoryInputError(
                "observer_elevation_m must be finite, non-negative, and below 100.",
                code="trajectory_invalid_observer_elevation",
                details={"observer_elevation_m": observer_elevation_m},
            )
        if (
            isinstance(time_batch_size, bool)
            or not isinstance(time_batch_size, Integral)
            or int(time_batch_size) < 1
        ):
            raise TrajectoryInputError(
                "time_batch_size must be a positive integer.",
                code="trajectory_invalid_provider_batch_size",
            )
        horizon_root = Path(horizons_path).expanduser().resolve()
        if not horizon_root.is_dir():
            raise TrajectoryInputError(
                "horizons_path must identify a precomputed horizon directory.",
                code="trajectory_horizons_not_found",
                details={"path": str(horizon_root)},
            )
        try:
            from ..products import _load_dem

            dem, georef = _load_dem(dem_path)
        except Exception as exc:
            raise TrajectoryInputError(
                "Unable to open the horizon provider DEM.",
                code="trajectory_provider_dem_invalid",
                details={"path": str(dem_path), "error": str(exc)},
            ) from exc
        self.sun_vectors = sun_vectors
        self.observer_elevation_m = observer
        self.time_batch_size = int(time_batch_size)
        self._dem = dem
        self._georef = georef
        self._horizon_root = horizon_root
        self._horizon_cache = _WindowCache(horizon_cache_entries)
        self._window_cache = _WindowCache(window_cache_entries)
        self._store: Any | None = None
        self._calculator: Any | None = None
        self._closed = False

    @property
    def georef(self) -> GeoReference:
        return self._georef

    def _ensure_store(self) -> Any:
        if self._closed:
            raise ConfigurationSpaceError(
                "The horizon sunlight provider is closed.",
                code="trajectory_provider_closed",
            )
        if self._store is None:
            from .._numba_horizon.file_format import HorizonTileStore

            self._store = HorizonTileStore(self._horizon_root)
        return self._store

    def _ensure_calculator(self) -> Any:
        if self._closed:
            raise ConfigurationSpaceError(
                "The horizon sunlight provider is closed.",
                code="trajectory_provider_closed",
            )
        if self._calculator is None:
            from .._numba_horizon.lightmap_cpu import LightmapCpuSession

            self._calculator = LightmapCpuSession(
                time_batch_size=self.time_batch_size
            )
        return self._calculator

    def _horizons(self, tile_y: int, tile_x: int) -> NDArray[np.float32]:
        key = (tile_y, tile_x, self.observer_elevation_m)
        cached = self._horizon_cache.get(key)
        if cached is not None:
            return cached
        store = self._ensure_store()
        try:
            values = store.read(tile_y, tile_x, self.observer_elevation_m)
        except Exception as exc:
            raise ConfigurationSpaceError(
                "Unable to read a required horizon tile.",
                code="trajectory_horizon_read_failed",
                details={"tile_x": tile_x, "tile_y": tile_y, "error": str(exc)},
            ) from exc
        if values is None:
            raise ConfigurationSpaceError(
                "A required precomputed horizon tile is missing.",
                code="trajectory_horizon_missing",
                details={"tile_x": tile_x, "tile_y": tile_y},
            )
        array = np.asarray(values)
        if (
            array.shape != _HORIZON_SHAPE
            or array.dtype != np.dtype(np.float32)
            or np.any(~np.isfinite(array))
        ):
            raise ConfigurationSpaceError(
                "A required horizon tile contains invalid data.",
                code="trajectory_invalid_horizon_tile",
                details={
                    "tile_x": tile_x,
                    "tile_y": tile_y,
                    "shape": list(array.shape),
                    "dtype": str(array.dtype),
                },
            )
        cached = _readonly_copy(array)
        self._horizon_cache.put(key, cached)
        return cached

    def read_many(
        self,
        x0: int,
        y0: int,
        width: int,
        height: int,
        times: Iterable[datetime],
    ) -> NDArray[np.uint8]:
        if self._closed:
            raise ConfigurationSpaceError(
                "The horizon sunlight provider is closed.",
                code="trajectory_provider_closed",
            )
        rx, ry, rw, rh = _window(self.georef, x0, y0, width, height)
        try:
            requested = tuple(
                as_utc(value, name=f"times[{index}]")
                for index, value in enumerate(times)
            )
        except TypeError as exc:
            raise TrajectoryInputError(
                "times must be an iterable of timezone-aware datetimes.",
                code="trajectory_invalid_dynamic_time",
            ) from exc
        if not requested:
            raise TrajectoryInputError(
                "At least one provider timestamp is required.",
                code="trajectory_invalid_dynamic_time",
            )
        keys = [(rx, ry, rw, rh, value) for value in requested]
        cached_layers = [self._window_cache.get(key) for key in keys]
        if all(value is not None for value in cached_layers):
            result = np.stack(cached_layers)
            result.flags.writeable = False
            return result
        try:
            vectors = self.sun_vectors.vectors(requested)
        except Exception as exc:
            raise ConfigurationSpaceError(
                "Unable to obtain Sun vectors for a sunlight request.",
                code="trajectory_provider_read_failed",
                details={"provider": type(self.sun_vectors).__name__, "error": str(exc)},
            ) from exc
        vectors_array = np.asarray(vectors)
        if (
            vectors_array.dtype != np.dtype(np.float64)
            or vectors_array.shape != (len(requested), 3)
            or np.any(~np.isfinite(vectors_array))
            or np.any(np.linalg.norm(vectors_array, axis=1) == 0.0)
        ):
            raise ConfigurationSpaceError(
                "The Sun-vector provider returned invalid data.",
                code="trajectory_invalid_provider_result",
                details={
                    "provider": type(self.sun_vectors).__name__,
                    "shape": list(vectors_array.shape),
                    "dtype": str(vectors_array.dtype),
                },
            )
        output = np.empty((len(requested), rh, rw), dtype=np.uint8)
        calculator = self._ensure_calculator()
        tile_y_start = (ry // _HORIZON_TILE_SIZE) * _HORIZON_TILE_SIZE
        tile_x_start = (rx // _HORIZON_TILE_SIZE) * _HORIZON_TILE_SIZE
        for tile_y in range(tile_y_start, ry + rh, _HORIZON_TILE_SIZE):
            valid_height = min(_HORIZON_TILE_SIZE, self.georef.height - tile_y)
            overlap_y0 = max(ry, tile_y)
            overlap_y1 = min(ry + rh, tile_y + valid_height)
            for tile_x in range(tile_x_start, rx + rw, _HORIZON_TILE_SIZE):
                valid_width = min(_HORIZON_TILE_SIZE, self.georef.width - tile_x)
                overlap_x0 = max(rx, tile_x)
                overlap_x1 = min(rx + rw, tile_x + valid_width)
                horizons = self._horizons(tile_y, tile_x)
                try:
                    tiles = tuple(
                        calculator.iter_patch_tiles(
                            self._dem,
                            horizons,
                            vectors_array,
                            tile_y=tile_y,
                            tile_x=tile_x,
                            valid_height=valid_height,
                            valid_width=valid_width,
                        )
                    )
                    calculated = np.stack(tiles)
                except Exception as exc:
                    raise ConfigurationSpaceError(
                        "Unable to calculate a sunlight window.",
                        code="trajectory_provider_calculation_failed",
                        details={
                            "tile_x": tile_x,
                            "tile_y": tile_y,
                            "error": str(exc),
                        },
                    ) from exc
                expected = (len(requested), valid_height, valid_width)
                if calculated.dtype != np.dtype(np.uint8) or calculated.shape != expected:
                    raise ConfigurationSpaceError(
                        "The sunlight calculator returned invalid data.",
                        code="trajectory_invalid_provider_result",
                        details={
                            "shape": list(calculated.shape),
                            "expected_shape": list(expected),
                            "dtype": str(calculated.dtype),
                        },
                    )
                output[
                    :,
                    overlap_y0 - ry : overlap_y1 - ry,
                    overlap_x0 - rx : overlap_x1 - rx,
                ] = calculated[
                    :,
                    overlap_y0 - tile_y : overlap_y1 - tile_y,
                    overlap_x0 - tile_x : overlap_x1 - tile_x,
                ]
        for index, key in enumerate(keys):
            self._window_cache.put(key, _readonly_copy(output[index]))
        output.flags.writeable = False
        return output

    def read(
        self, x0: int, y0: int, width: int, height: int, time: datetime
    ) -> NDArray[np.uint8]:
        return _readonly_copy(self.read_many(x0, y0, width, height, (time,))[0])

    def close(self) -> None:
        self._horizon_cache.clear()
        self._window_cache.clear()
        self._store = None
        self._calculator = None
        self._closed = True

    def __enter__(self) -> HorizonSunlightProvider:
        if self._closed:
            raise ConfigurationSpaceError(
                "The horizon sunlight provider is closed.",
                code="trajectory_provider_closed",
            )
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        self.close()


class ArrayEarthElevationProvider(_IntervalArrayProvider):
    """Serve a copied finite ``float32[interval, y, x]`` Earth cube."""

    _dtype = np.dtype(np.float32)
    _signal_name = "earth_elevation"

    def read(
        self, x0: int, y0: int, width: int, height: int, time: datetime
    ) -> NDArray[np.float32]:
        return self._read(x0, y0, width, height, time)


def _provider_result(
    provider: Any,
    expected_dtype: np.dtype[Any],
    window: tuple[int, int, int, int],
    time: datetime,
    *,
    signal: str,
) -> NDArray[Any]:
    x0, y0, width, height = window
    try:
        values = provider.read(x0, y0, width, height, time)
    except (TrajectoryInputError, ConfigurationSpaceError):
        raise
    except Exception as exc:
        raise ConfigurationSpaceError(
            "An environment provider failed to supply requested data.",
            code="trajectory_provider_read_failed",
            details={
                "signal": signal,
                "provider": type(provider).__name__,
                "error": str(exc),
            },
        ) from exc
    array = np.asarray(values)
    if array.shape != (height, width) or array.dtype != expected_dtype or (
        np.issubdtype(expected_dtype, np.floating) and np.any(~np.isfinite(array))
    ):
        raise ConfigurationSpaceError(
            "An environment provider returned invalid data.",
            code="trajectory_invalid_provider_result",
            details={
                "signal": signal,
                "provider": type(provider).__name__,
                "shape": list(array.shape),
                "expected_shape": [height, width],
                "dtype": str(array.dtype),
                "expected_dtype": str(expected_dtype),
            },
        )
    return array


class StaticConfigurationSpaceProvider:
    """Serve a copied static Boolean occupancy raster for every valid UTC time."""

    def __init__(
        self, allowed: npt.ArrayLike, georef: GeoReference, *, cache_entries: int = 16
    ) -> None:
        array = np.asarray(allowed)
        if (
            not isinstance(georef, GeoReference)
            or array.shape != (georef.height, georef.width)
            or array.dtype != np.dtype(np.bool_)
        ):
            raise TrajectoryInputError(
                "Static configuration space must be Boolean and match georef.",
                code="trajectory_invalid_provider_data",
                details={"shape": list(array.shape), "dtype": str(array.dtype)},
            )
        self._georef = georef
        self._allowed = _readonly_copy(array)
        self._cache = _WindowCache(cache_entries)
        self._closed = False

    @property
    def georef(self) -> GeoReference:
        return self._georef

    def read(
        self, x0: int, y0: int, width: int, height: int, time: datetime
    ) -> NDArray[np.bool_]:
        if self._closed:
            raise ConfigurationSpaceError(
                "The configuration provider is closed.",
                code="trajectory_provider_closed",
            )
        as_utc(time, name="time")
        rx, ry, rw, rh = _window(self.georef, x0, y0, width, height)
        key = (rx, ry, rw, rh)
        cached = self._cache.get(key)
        if cached is None:
            cached = _readonly_copy(self._allowed[ry : ry + rh, rx : rx + rw])
            self._cache.put(key, cached)
        return _readonly_copy(cached)

    def close(self) -> None:
        self._cache.clear()
        self._closed = True

    def __enter__(self) -> StaticConfigurationSpaceProvider:
        if self._closed:
            raise ConfigurationSpaceError(
                "The configuration provider is closed.",
                code="trajectory_provider_closed",
            )
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        self.close()


class SunlightThresholdProvider:
    """Convert sunlight bytes to an inclusive hard occupancy constraint."""

    def __init__(
        self,
        source: SunlightProvider,
        minimum_fraction: float,
        *,
        cache_entries: int = 32,
    ) -> None:
        try:
            threshold = float(minimum_fraction)
        except (TypeError, ValueError, OverflowError) as exc:
            raise TrajectoryInputError(
                "minimum_fraction must be between zero and one.",
                code="trajectory_invalid_sunlight_threshold",
            ) from exc
        if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            raise TrajectoryInputError(
                "minimum_fraction must be between zero and one.",
                code="trajectory_invalid_sunlight_threshold",
                details={"minimum_fraction": minimum_fraction},
            )
        if not isinstance(getattr(source, "georef", None), GeoReference):
            raise TrajectoryInputError(
                "Sunlight provider must expose a GeoReference.",
                code="trajectory_invalid_provider_grid",
            )
        self.source = source
        self.minimum_fraction = threshold
        self._georef = source.georef
        self._cache = _WindowCache(cache_entries)
        self._closed = False

    @property
    def georef(self) -> GeoReference:
        return self._georef

    def read(
        self, x0: int, y0: int, width: int, height: int, time: datetime
    ) -> NDArray[np.bool_]:
        if self._closed:
            raise ConfigurationSpaceError(
                "The configuration provider is closed.",
                code="trajectory_provider_closed",
            )
        window = _window(self.georef, x0, y0, width, height)
        utc = as_utc(time, name="time")
        key = (*window, utc)
        cached = self._cache.get(key)
        if cached is None:
            values = _provider_result(
                self.source, np.dtype(np.uint8), window, utc, signal="sunlight"
            )
            cached = _readonly_copy(
                values.astype(np.float64) / 255.0 >= self.minimum_fraction
            )
            self._cache.put(key, cached)
        return _readonly_copy(cached)

    def close(self) -> None:
        self._cache.clear()
        self._closed = True

    def __enter__(self) -> SunlightThresholdProvider:
        if self._closed:
            raise ConfigurationSpaceError(
                "The configuration provider is closed.",
                code="trajectory_provider_closed",
            )
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        self.close()


class EarthElevationThresholdProvider:
    """Convert Earth elevation to an inclusive hard occupancy constraint."""

    def __init__(
        self,
        source: EarthElevationProvider,
        minimum_degrees: float,
        *,
        cache_entries: int = 32,
    ) -> None:
        try:
            threshold = float(minimum_degrees)
        except (TypeError, ValueError, OverflowError) as exc:
            raise TrajectoryInputError(
                "minimum_degrees must be finite.",
                code="trajectory_invalid_earth_threshold",
            ) from exc
        if not np.isfinite(threshold):
            raise TrajectoryInputError(
                "minimum_degrees must be finite.",
                code="trajectory_invalid_earth_threshold",
            )
        if not isinstance(getattr(source, "georef", None), GeoReference):
            raise TrajectoryInputError(
                "Earth-elevation provider must expose a GeoReference.",
                code="trajectory_invalid_provider_grid",
            )
        self.source = source
        self.minimum_degrees = threshold
        self._georef = source.georef
        self._cache = _WindowCache(cache_entries)
        self._closed = False
    @property
    def georef(self) -> GeoReference:
        return self._georef

    def read(
        self, x0: int, y0: int, width: int, height: int, time: datetime
    ) -> NDArray[np.bool_]:
        if self._closed:
            raise ConfigurationSpaceError(
                "The configuration provider is closed.",
                code="trajectory_provider_closed",
            )
        window = _window(self.georef, x0, y0, width, height)
        utc = as_utc(time, name="time")
        key = (*window, utc)
        cached = self._cache.get(key)
        if cached is None:
            values = _provider_result(
                self.source,
                np.dtype(np.float32),
                window,
                utc,
                signal="earth_elevation",
            )
            cached = _readonly_copy(values >= self.minimum_degrees)
            self._cache.put(key, cached)
        return _readonly_copy(cached)

    def close(self) -> None:
        self._cache.clear()
        self._closed = True

    def __enter__(self) -> EarthElevationThresholdProvider:
        if self._closed:
            raise ConfigurationSpaceError(
                "The configuration provider is closed.",
                code="trajectory_provider_closed",
            )
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        self.close()


class AllOfConfigurationSpaceProvider:
    """Combine configuration providers with logical AND on one shared grid."""

    def __init__(
        self,
        providers: Sequence[ConfigurationSpaceProvider],
        *,
        cache_entries: int = 32,
    ) -> None:
        values = tuple(providers)
        if not values:
            raise TrajectoryInputError(
                "At least one configuration provider is required.",
                code="trajectory_empty_configuration_providers",
            )
        georef = getattr(values[0], "georef", None)
        if not isinstance(georef, GeoReference):
            raise TrajectoryInputError(
                "Configuration providers must expose GeoReference grids.",
                code="trajectory_invalid_provider_grid",
            )
        for index, provider in enumerate(values[1:], start=1):
            other = getattr(provider, "georef", None)
            try:
                matches = isinstance(other, GeoReference) and same_grid(georef, other)
            except Exception as exc:
                raise TrajectoryInputError(
                    "Unable to compare configuration-provider grids.",
                    code="trajectory_invalid_provider_grid",
                    details={"provider_index": index, "error": str(exc)},
                ) from exc
            if not matches:
                raise TrajectoryInputError(
                    "Configuration-provider grids must match.",
                    code="trajectory_provider_grid_mismatch",
                    details={"provider_index": index},
                )
        self.providers = values
        self._georef = georef
        self._cache = _WindowCache(cache_entries)
        self._closed = False

    @property
    def georef(self) -> GeoReference:
        return self._georef

    def read(
        self, x0: int, y0: int, width: int, height: int, time: datetime
    ) -> NDArray[np.bool_]:
        if self._closed:
            raise ConfigurationSpaceError(
                "The configuration provider is closed.",
                code="trajectory_provider_closed",
            )
        window = _window(self.georef, x0, y0, width, height)
        utc = as_utc(time, name="time")
        key = (*window, utc)
        cached = self._cache.get(key)
        if cached is None:
            combined = np.ones((window[3], window[2]), dtype=np.bool_)
            for provider in self.providers:
                combined &= _provider_result(
                    provider,
                    np.dtype(np.bool_),
                    window,
                    utc,
                    signal="configuration_space",
                )
            cached = _readonly_copy(combined)
            self._cache.put(key, cached)
        return _readonly_copy(cached)

    def close(self) -> None:
        self._cache.clear()
        self._closed = True

    def __enter__(self) -> AllOfConfigurationSpaceProvider:
        if self._closed:
            raise ConfigurationSpaceError(
                "The configuration provider is closed.",
                code="trajectory_provider_closed",
            )
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        self.close()


__all__ = [
    "AllOfConfigurationSpaceProvider",
    "ArrayEarthElevationProvider",
    "ArraySunlightProvider",
    "ConfigurationSpaceProvider",
    "EarthElevationProvider",
    "EarthElevationThresholdProvider",
    "ExplicitSunVectorProvider",
    "HorizonSunlightProvider",
    "SpiceSunVectorProvider",
    "StaticConfigurationSpaceProvider",
    "SunVectorProvider",
    "SunlightProvider",
    "SunlightThresholdProvider",
]
