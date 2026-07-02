from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from math import cos, radians, sin
from typing import Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray

from . import spice as _kernel_state
from .errors import SpiceGeometryError, TimeRangeError
from .temporal import TimeInput, _parse_time


BodyName: TypeAlias = Literal["sun", "earth"]
_BODY_TARGETS = {
    "sun": "SUN",
    "earth": "EARTH",
}


@dataclass(frozen=True, slots=True)
class LonLat:
    longitude: float
    latitude: float

    def __post_init__(self) -> None:
        try:
            longitude = float(self.longitude)
            latitude = float(self.latitude)
        except (TypeError, ValueError, OverflowError) as exc:
            raise SpiceGeometryError(
                "Longitude and latitude must be finite numbers.",
                code="spice_invalid_lonlat",
                details={"longitude": self.longitude, "latitude": self.latitude},
            ) from exc
        if not np.isfinite(longitude) or not np.isfinite(latitude):
            raise SpiceGeometryError(
                "Longitude and latitude must be finite numbers.",
                code="spice_invalid_lonlat",
                details={"longitude": self.longitude, "latitude": self.latitude},
            )
        if not -90.0 <= latitude <= 90.0:
            raise SpiceGeometryError(
                "Latitude must be between -90 and 90 degrees.",
                code="spice_invalid_latitude",
                details={"latitude": latitude},
            )
        object.__setattr__(self, "longitude", longitude)
        object.__setattr__(self, "latitude", latitude)


def iter_times(
    start: TimeInput,
    stop: TimeInput,
    step: timedelta,
) -> Iterator[datetime]:
    start_time = _parse_time(start, source_timezone=None)
    stop_time = _parse_time(stop, source_timezone=None)
    if not isinstance(step, timedelta):
        raise TimeRangeError(
            "step must be a datetime.timedelta.",
            code="time_iterator_invalid_step",
            details={"type": type(step).__name__},
        )
    if step <= timedelta(0):
        raise TimeRangeError(
            "step must be positive.",
            code="time_iterator_invalid_step",
        )
    if stop_time < start_time:
        raise TimeRangeError(
            "stop must be greater than or equal to start.",
            code="time_iterator_invalid_order",
        )

    current = start_time
    while current <= stop_time:
        yield current
        current = current + step


def _spiceypy():
    try:
        import spiceypy
    except ImportError as exc:
        raise SpiceGeometryError(
            "SpiceyPy is required for SPICE geometry operations.",
            code="spiceypy_unavailable",
        ) from exc
    return spiceypy


def _body_target(body: BodyName | str) -> str:
    key = str(body).strip().lower()
    try:
        return _BODY_TARGETS[key]
    except KeyError as exc:
        raise SpiceGeometryError(
            "Unsupported SPICE body name.",
            code="spice_unsupported_body",
            details={"body": body, "supported": sorted(_BODY_TARGETS)},
        ) from exc


def _time_list(times: Iterable[datetime]) -> list[datetime]:
    values = [_parse_time(time, source_timezone=None) for time in times]
    return values


def _spice_utc(time: datetime) -> str:
    utc = time.astimezone(timezone.utc).replace(tzinfo=None)
    return utc.isoformat(timespec="microseconds")


def _moon_surface_and_basis(
    point: LonLat,
    radii: Sequence[float],
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    axes = np.asarray(radii, dtype=np.float64)
    if axes.shape != (3,) or np.any(~np.isfinite(axes)) or np.any(axes <= 0.0):
        raise SpiceGeometryError(
            "SPICE did not provide valid Moon radii.",
            code="spice_invalid_moon_radii",
            details={"radii": [float(value) for value in axes.ravel()]},
        )

    longitude = radians(point.longitude)
    latitude = radians(point.latitude)
    cos_lat = cos(latitude)
    unit_up = np.asarray(
        [
            cos_lat * cos(longitude),
            cos_lat * sin(longitude),
            sin(latitude),
        ],
        dtype=np.float64,
    )
    radius = 1.0 / np.sqrt(np.sum((unit_up / axes) ** 2))
    surface = radius * unit_up
    north = np.asarray(
        [
            -sin(latitude) * cos(longitude),
            -sin(latitude) * sin(longitude),
            cos(latitude),
        ],
        dtype=np.float64,
    )
    east = np.asarray([-sin(longitude), cos(longitude), 0.0], dtype=np.float64)
    down = -unit_up
    return surface, north, east, down


def _vectors_to_azimuth_elevation(
    vectors: NDArray[np.float64],
) -> NDArray[np.float64]:
    output = np.empty((vectors.shape[0], 2), dtype=np.float64)
    output[:, 0] = np.degrees(np.arctan2(vectors[:, 1], vectors[:, 0])) % 360.0
    output[:, 1] = np.degrees(
        np.arctan2(-vectors[:, 2], np.hypot(vectors[:, 0], vectors[:, 1]))
    )
    return output


def body_vectors_ned(
    point: LonLat,
    body: BodyName,
    times: Iterable[datetime],
    *,
    ensure_kernels: bool = True,
) -> NDArray[np.float64]:
    if ensure_kernels:
        _kernel_state.ensure_default_kernels()

    target = _body_target(body)
    time_values = _time_list(times)
    spiceypy = _spiceypy()
    try:
        _dim, radii = spiceypy.bodvrd("MOON", "RADII", 3)
    except Exception as exc:
        raise SpiceGeometryError(
            "Unable to read Moon radii from loaded SPICE kernels.",
            code="spice_moon_radii_failed",
            details={"error": str(exc)},
        ) from exc

    surface, north, east, down = _moon_surface_and_basis(point, radii)
    output = np.empty((len(time_values), 3), dtype=np.float64)
    for index, time_value in enumerate(time_values):
        try:
            et = spiceypy.utc2et(_spice_utc(time_value))
            position, _light_time = spiceypy.spkpos(
                target,
                et,
                "MOON_ME",
                "LT+S",
                "MOON",
            )
        except Exception as exc:
            raise SpiceGeometryError(
                "Unable to compute SPICE body position.",
                code="spice_body_position_failed",
                details={
                    "body": body,
                    "time": time_value.isoformat(),
                    "error": str(exc),
                },
            ) from exc
        topocentric = np.asarray(position, dtype=np.float64) - surface
        output[index, 0] = float(np.dot(topocentric, north))
        output[index, 1] = float(np.dot(topocentric, east))
        output[index, 2] = float(np.dot(topocentric, down))
    return output


def body_vectors_ned_dataframe(
    point: LonLat,
    body: BodyName,
    times: Iterable[datetime],
    *,
    ensure_kernels: bool = True,
):
    import pandas as pd

    time_values = _time_list(times)
    vectors = body_vectors_ned(
        point,
        body,
        time_values,
        ensure_kernels=ensure_kernels,
    )
    return pd.DataFrame(
        {
            "time": time_values,
            "x": vectors[:, 0],
            "y": vectors[:, 1],
            "z": vectors[:, 2],
        }
    )


def body_azimuth_elevation(
    point: LonLat,
    body: BodyName,
    times: Iterable[datetime],
    *,
    ensure_kernels: bool = True,
) -> NDArray[np.float64]:
    vectors = body_vectors_ned(
        point,
        body,
        times,
        ensure_kernels=ensure_kernels,
    )
    return _vectors_to_azimuth_elevation(vectors)


def body_azimuth_elevation_dataframe(
    point: LonLat,
    body: BodyName,
    times: Iterable[datetime],
    *,
    ensure_kernels: bool = True,
):
    import pandas as pd

    time_values = _time_list(times)
    angles = body_azimuth_elevation(
        point,
        body,
        time_values,
        ensure_kernels=ensure_kernels,
    )
    return pd.DataFrame(
        {
            "time": time_values,
            "azimuth": angles[:, 0],
            "elevation": angles[:, 1],
        }
    )


def plot_body_elevation(
    point: LonLat,
    body: BodyName,
    times: Iterable[datetime],
    *,
    grid: bool = True,
    ensure_kernels: bool = True,
):
    import matplotlib.pyplot as plt

    time_values = _time_list(times)
    angles = body_azimuth_elevation(
        point,
        body,
        time_values,
        ensure_kernels=ensure_kernels,
    )
    fig, ax = plt.subplots()
    ax.plot(time_values, angles[:, 1], label=str(body).lower())
    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel("Elevation (deg)")
    ax.grid(bool(grid))
    return fig, ax


def plot_body_elevations(
    point: LonLat,
    bodies: Sequence[BodyName],
    times: Iterable[datetime],
    *,
    grid: bool = True,
    ensure_kernels: bool = True,
):
    import matplotlib.pyplot as plt

    time_values = _time_list(times)
    fig, ax = plt.subplots()
    for body in bodies:
        angles = body_azimuth_elevation(
            point,
            body,
            time_values,
            ensure_kernels=ensure_kernels,
        )
        ax.plot(time_values, angles[:, 1], label=str(body).lower())
    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel("Elevation (deg)")
    ax.grid(bool(grid))
    if bodies:
        ax.legend()
    return fig, ax
