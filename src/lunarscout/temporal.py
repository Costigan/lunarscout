from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Any, Literal, TypeAlias
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import numpy as np
from numpy.typing import NDArray

from .errors import TemporalCubeError, TemporalOperationError, TimeRangeError
from .georeference import GeoReference


TimeInput: TypeAlias = str | date | datetime | np.datetime64
Nodata: TypeAlias = int | float | None | Literal["auto"]
_MAX_TIME_COUNT = 1_000_000
_TIME_DTYPE = np.dtype("datetime64[us]")


def utc_datetime(
    year: int,
    month: int,
    day: int,
    hour: int = 0,
    minute: int = 0,
    second: int = 0,
    microsecond: int = 0,
    *,
    fold: int = 0,
) -> datetime:
    """Construct a standard timezone-aware UTC datetime."""

    return datetime(
        year,
        month,
        day,
        hour,
        minute,
        second,
        microsecond,
        tzinfo=timezone.utc,
        fold=fold,
    )


def _source_timezone(name: str | None):
    if name is None:
        return timezone.utc
    try:
        return ZoneInfo(name)
    except ZoneInfoNotFoundError as exc:
        raise TimeRangeError(
            "Unknown source timezone.",
            code="time_range_unknown_timezone",
            details={"source_timezone": name},
        ) from exc


def _datetime64_to_datetime(value: np.datetime64) -> datetime:
    if np.isnat(value):
        raise TimeRangeError(
            "Time values cannot be NaT.",
            code="time_range_nat",
        )
    microseconds = int(value.astype(_TIME_DTYPE).astype(np.int64))
    try:
        return datetime(1970, 1, 1, tzinfo=timezone.utc) + timedelta(
            microseconds=microseconds
        )
    except (OverflowError, ValueError) as exc:
        raise TimeRangeError(
            "Time value is outside the supported Python datetime range.",
            code="time_range_out_of_range",
            details={"value": str(value)},
        ) from exc


def _parse_time(value: TimeInput, *, source_timezone: str | None) -> datetime:
    if isinstance(value, np.datetime64):
        return _datetime64_to_datetime(value)
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, date):
        parsed = datetime(value.year, value.month, value.day)
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            raise TimeRangeError(
                "Time strings cannot be empty.",
                code="time_range_invalid_time",
            )
        if text.endswith(("Z", "z")):
            text = f"{text[:-1]}+00:00"
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError as exc:
            raise TimeRangeError(
                "Time must be an ISO 8601 string, date, datetime, or numpy.datetime64.",
                code="time_range_invalid_time",
                details={"value": value},
            ) from exc
    else:
        raise TimeRangeError(
            "Unsupported time value.",
            code="time_range_invalid_time",
            details={"type": type(value).__name__},
        )
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=_source_timezone(source_timezone))
    return parsed.astimezone(timezone.utc)


def _datetime64_utc(value: datetime) -> np.datetime64:
    utc_naive = value.astimezone(timezone.utc).replace(tzinfo=None)
    return np.datetime64(utc_naive, "us")


@dataclass(frozen=True, slots=True)
class TimeRange:
    """Inclusive UTC sampling domain for temporal calculations."""

    start: datetime
    stop: datetime
    step_hours: float
    _values: NDArray[np.datetime64] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        start = _parse_time(self.start, source_timezone=None)
        stop = _parse_time(self.stop, source_timezone=None)
        try:
            step_hours = float(self.step_hours)
        except (TypeError, ValueError, OverflowError) as exc:
            raise TimeRangeError(
                "step_hours must be a finite positive number.",
                code="time_range_invalid_step",
                details={"step_hours": self.step_hours},
            ) from exc
        if not np.isfinite(step_hours) or step_hours <= 0:
            raise TimeRangeError(
                "step_hours must be a finite positive number.",
                code="time_range_invalid_step",
                details={"step_hours": self.step_hours},
            )
        if stop < start:
            raise TimeRangeError(
                "stop must be greater than or equal to start.",
                code="time_range_invalid_order",
            )
        step_microseconds = int(round(step_hours * 3_600_000_000.0))
        if step_microseconds < 1:
            raise TimeRangeError(
                "step_hours is smaller than the supported one-microsecond resolution.",
                code="time_range_step_too_small",
                details={"step_hours": step_hours},
            )
        span_microseconds = int((stop - start) / timedelta(microseconds=1))
        count = span_microseconds // step_microseconds + 1
        if count > _MAX_TIME_COUNT:
            raise TimeRangeError(
                "Time range contains too many samples.",
                code="time_range_too_large",
                details={"time_count": count, "maximum": _MAX_TIME_COUNT},
            )
        start_value = _datetime64_utc(start).astype(np.int64)
        offsets = np.arange(count, dtype=np.int64) * step_microseconds
        values = (start_value + offsets).astype(_TIME_DTYPE)
        values.flags.writeable = False
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "stop", stop)
        object.__setattr__(self, "step_hours", step_microseconds / 3_600_000_000.0)
        object.__setattr__(self, "_values", values)

    @property
    def values(self) -> NDArray[np.datetime64]:
        return self._values

    @property
    def time_count(self) -> int:
        return int(self._values.size)

    def __len__(self) -> int:
        return self.time_count


def times(
    start: TimeInput,
    stop: TimeInput,
    *,
    step_hours: float,
    source_timezone: str | None = None,
) -> TimeRange:
    """Construct an inclusive UTC ``TimeRange``."""

    return TimeRange(
        start=_parse_time(start, source_timezone=source_timezone),
        stop=_parse_time(stop, source_timezone=source_timezone),
        step_hours=step_hours,
    )


@dataclass(frozen=True, slots=True, eq=False)
class TemporalCube:
    """Named in-memory temporal raster with shape ``(time, y, x)``."""

    values: NDArray[Any]
    times: NDArray[np.datetime64] | TimeRange
    georef: GeoReference

    def __post_init__(self) -> None:
        values = np.asarray(self.values)
        if values.ndim != 3:
            raise TemporalCubeError(
                "TemporalCube values must have shape (time, y, x).",
                code="temporal_cube_invalid_shape",
                details={"shape": list(values.shape)},
            )
        if values.shape[0] < 1:
            raise TemporalCubeError(
                "TemporalCube must contain at least one time sample.",
                code="temporal_cube_empty",
            )
        if np.issubdtype(values.dtype, np.complexfloating):
            raise TemporalCubeError(
                "Complex TemporalCube values are not supported in v0.1.",
                code="temporal_cube_unsupported_dtype",
                details={"dtype": str(values.dtype)},
            )
        if not (
            np.issubdtype(values.dtype, np.number)
            or np.issubdtype(values.dtype, np.bool_)
        ):
            raise TemporalCubeError(
                "TemporalCube values must use a numeric or Boolean NumPy dtype.",
                code="temporal_cube_unsupported_dtype",
                details={"dtype": str(values.dtype)},
            )
        if not isinstance(self.georef, GeoReference):
            raise TemporalCubeError(
                "TemporalCube georef must be a GeoReference.",
                code="temporal_cube_invalid_georef",
            )
        expected_spatial_shape = (self.georef.height, self.georef.width)
        if values.shape[1:] != expected_spatial_shape:
            raise TemporalCubeError(
                "TemporalCube spatial shape does not match its GeoReference.",
                code="temporal_cube_spatial_shape_mismatch",
                details={
                    "shape": list(values.shape[1:]),
                    "expected_shape": list(expected_spatial_shape),
                },
            )
        raw_times = self.times.values if isinstance(self.times, TimeRange) else self.times
        time_values = np.asarray(raw_times)
        if time_values.ndim != 1 or not np.issubdtype(time_values.dtype, np.datetime64):
            raise TemporalCubeError(
                "TemporalCube times must be a one-dimensional datetime64 array or TimeRange.",
                code="temporal_cube_invalid_times",
                details={"shape": list(time_values.shape), "dtype": str(time_values.dtype)},
            )
        time_values = time_values.astype(_TIME_DTYPE, copy=True)
        if time_values.size != values.shape[0]:
            raise TemporalCubeError(
                "TemporalCube time count does not match its values.",
                code="temporal_cube_time_count_mismatch",
                details={"times": int(time_values.size), "values": int(values.shape[0])},
            )
        if np.any(np.isnat(time_values)):
            raise TemporalCubeError(
                "TemporalCube times cannot contain NaT.",
                code="temporal_cube_nat",
            )
        if time_values.size > 1 and np.any(np.diff(time_values).astype(np.int64) <= 0):
            raise TemporalCubeError(
                "TemporalCube times must be strictly increasing.",
                code="temporal_cube_times_not_increasing",
            )
        time_values.flags.writeable = False
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "times", time_values)

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.values.shape

    @property
    def dtype(self) -> np.dtype[Any]:
        return self.values.dtype

    @property
    def time_count(self) -> int:
        return int(self.values.shape[0])

    @property
    def height(self) -> int:
        return int(self.values.shape[1])

    @property
    def width(self) -> int:
        return int(self.values.shape[2])

    @property
    def nbytes(self) -> int:
        return int(self.values.nbytes + self.times.nbytes)

    @property
    def dimensions(self) -> tuple[str, str, str]:
        return ("time", "y", "x")


def _resolved_nodata(cube: TemporalCube, nodata: Nodata) -> int | float | None:
    if nodata == "auto":
        return cube.georef.nodata
    if nodata is None or isinstance(nodata, (int, float, np.integer, np.floating)):
        return nodata.item() if isinstance(nodata, (np.integer, np.floating)) else nodata
    raise TemporalOperationError(
        "nodata must be 'auto', None, or a numeric value.",
        code="temporal_invalid_nodata",
        details={"nodata": nodata},
    )


def _valid_values(cube: TemporalCube, nodata: int | float | None):
    if nodata is None:
        return cube.values
    if isinstance(nodata, (float, np.floating)) and np.isnan(nodata):
        invalid = np.isnan(cube.values)
    else:
        invalid = cube.values == nodata
    return np.ma.array(cube.values, mask=invalid, copy=False)


def _finish_reduction(
    reduced: Any,
    cube: TemporalCube,
    nodata: int | float | None,
) -> tuple[NDArray[Any], GeoReference]:
    if np.ma.isMaskedArray(reduced):
        if nodata is None:
            result = np.asarray(reduced)
        else:
            try:
                result = np.asarray(np.ma.filled(reduced, nodata))
            except (TypeError, ValueError, OverflowError) as exc:
                raise TemporalOperationError(
                    "Output nodata cannot be represented by the reduced dtype.",
                    code="temporal_unrepresentable_nodata",
                    details={"dtype": str(reduced.dtype), "nodata": nodata},
                ) from exc
    else:
        result = np.asarray(reduced)
    return result, cube.georef.with_nodata(nodata)


def _file_backed_reduction(
    cube: Any,
    operation: str,
    *,
    nodata: Nodata,
    dtype: np.dtype[Any] | type[Any] | str | None = None,
    ddof: float = 0,
) -> tuple[NDArray[Any], GeoReference] | None:
    if isinstance(cube, TemporalCube):
        return None
    reducer = getattr(cube, "_temporal_reduce", None)
    if not callable(reducer):
        raise TemporalOperationError(
            "Temporal reductions require a TemporalCube or TemporalGeoTiffSeries.",
            code="temporal_invalid_input",
            details={"type": type(cube).__name__},
        )
    return reducer(operation, nodata=nodata, dtype=dtype, ddof=ddof)


def temporal_mean(
    cube: Any,
    *,
    nodata: Nodata = "auto",
    dtype: np.dtype[Any] | type[Any] | str | None = None,
) -> tuple[NDArray[Any], GeoReference]:
    file_backed = _file_backed_reduction(
        cube, "mean", nodata=nodata, dtype=dtype
    )
    if file_backed is not None:
        return file_backed
    resolved = _resolved_nodata(cube, nodata)
    values = _valid_values(cube, resolved)
    return _finish_reduction(values.mean(axis=0, dtype=dtype), cube, resolved)


def temporal_min(
    cube: Any,
    *,
    nodata: Nodata = "auto",
) -> tuple[NDArray[Any], GeoReference]:
    file_backed = _file_backed_reduction(cube, "min", nodata=nodata)
    if file_backed is not None:
        return file_backed
    resolved = _resolved_nodata(cube, nodata)
    return _finish_reduction(_valid_values(cube, resolved).min(axis=0), cube, resolved)


def temporal_max(
    cube: Any,
    *,
    nodata: Nodata = "auto",
) -> tuple[NDArray[Any], GeoReference]:
    file_backed = _file_backed_reduction(cube, "max", nodata=nodata)
    if file_backed is not None:
        return file_backed
    resolved = _resolved_nodata(cube, nodata)
    return _finish_reduction(_valid_values(cube, resolved).max(axis=0), cube, resolved)


def temporal_std(
    cube: Any,
    *,
    nodata: Nodata = "auto",
    dtype: np.dtype[Any] | type[Any] | str | None = None,
    ddof: float = 0,
) -> tuple[NDArray[Any], GeoReference]:
    file_backed = _file_backed_reduction(
        cube, "std", nodata=nodata, dtype=dtype, ddof=ddof
    )
    if file_backed is not None:
        return file_backed
    resolved = _resolved_nodata(cube, nodata)
    values = _valid_values(cube, resolved)
    return _finish_reduction(
        values.std(axis=0, dtype=dtype, ddof=ddof),
        cube,
        resolved,
    )
