"""Private explicit/generated Moon-ME vector boundary for tiled products."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Literal

import numpy as np
import numpy.typing as npt

from lunarscout.spice_geometry import iter_times
from lunarscout.temporal import TimeInput, TimeRange, _parse_time


BodyName = Literal["sun", "earth"]
TimeConversion = Literal["utc2et", "linear_from_anchor"]
_TARGETS = {"sun": "SUN", "earth": "EARTH"}


@dataclass(frozen=True, slots=True)
class MoonMeVectorSeries:
    times_utc: tuple[datetime, ...]
    vectors_m: npt.NDArray[np.float64]

    def __post_init__(self) -> None:
        values = self.vectors_m
        if (
            not isinstance(values, np.ndarray)
            or values.dtype != np.dtype(np.float64)
            or values.shape != (len(self.times_utc), 3)
            or len(self.times_utc) == 0
            or not values.flags.c_contiguous
            or not np.all(np.isfinite(values))
        ):
            raise ValueError("Moon-ME vectors must be finite C-contiguous float64[time, 3]")
        if any(time.tzinfo is None or time.utcoffset() is None for time in self.times_utc):
            raise ValueError("vector timestamps must be timezone-aware")


def _times_tuple(values: Iterable[TimeInput] | TimeRange) -> tuple[datetime, ...]:
    if isinstance(values, TimeRange):
        values = values.values
    return tuple(
        _parse_time(value, source_timezone=None).astimezone(timezone.utc)
        for value in values
    )


def _requested_times(
    *,
    times: Iterable[TimeInput] | TimeRange | None,
    start: TimeInput | None,
    stop: TimeInput | None,
    step: timedelta | None,
) -> tuple[datetime, ...]:
    if times is not None:
        return _times_tuple(times)
    if start is None or stop is None or step is None:
        raise ValueError("provide times or start, stop, and step")
    return tuple(iter_times(start, stop, step))


def _spice_utc(value: datetime) -> str:
    return value.astimezone(timezone.utc).replace(tzinfo=None).isoformat(
        timespec="microseconds"
    )


def generate_moon_me_vectors(
    body: BodyName | str,
    times: Iterable[TimeInput] | TimeRange,
    *,
    ensure_kernels: bool = True,
    time_conversion: TimeConversion = "utc2et",
    linear_anchor: TimeInput = "2023-12-01T00:00:00Z",
) -> MoonMeVectorSeries:
    """Generate geometric Moon-centered positions in Moon-ME, in meters.

    ``utc2et`` converts every timestamp independently and is the default for
    mission-time pointing accuracy. ``linear_from_anchor`` converts one UTC
    anchor and adds ordinary elapsed seconds. The latter reproduces the
    current C# convention and may be selected only where equivalence to
    per-timestamp ``utc2et`` has been demonstrated for the intended product;
    it does not model leap seconds crossed relative to the anchor.
    """
    key = str(body).strip().lower()
    if key not in _TARGETS:
        raise ValueError("body must be 'sun' or 'earth'")
    time_values = _times_tuple(times)
    if ensure_kernels:
        from lunarscout import spice

        spice.ensure_default_kernels()
    try:
        import spiceypy
    except ImportError as exc:
        raise RuntimeError("SpiceyPy is required to generate Moon-ME vectors") from exc
    if time_conversion == "utc2et":
        ephemeris_times = np.fromiter(
            (spiceypy.utc2et(_spice_utc(value)) for value in time_values),
            dtype=np.float64,
            count=len(time_values),
        )
    elif time_conversion == "linear_from_anchor":
        anchor = _parse_time(linear_anchor, source_timezone=None).astimezone(timezone.utc)
        anchor_et = float(spiceypy.utc2et(_spice_utc(anchor)))
        ephemeris_times = np.fromiter(
            (anchor_et + (value - anchor).total_seconds() for value in time_values),
            dtype=np.float64,
            count=len(time_values),
        )
    else:
        raise ValueError("time_conversion must be 'utc2et' or 'linear_from_anchor'")
    positions_km, _light_times = spiceypy.spkpos(
        _TARGETS[key], ephemeris_times, "MOON_ME", "NONE", "MOON"
    )
    output = np.ascontiguousarray(positions_km, dtype=np.float64) * 1000.0
    return MoonMeVectorSeries(time_values, np.ascontiguousarray(output))


def resolve_moon_me_vectors(
    body: BodyName | str,
    *,
    explicit_vectors_m: npt.ArrayLike | None = None,
    explicit_times: Iterable[TimeInput] | TimeRange | None = None,
    times: Iterable[TimeInput] | TimeRange | None = None,
    start: TimeInput | None = None,
    stop: TimeInput | None = None,
    step: timedelta | None = None,
    ensure_kernels: bool = True,
    time_conversion: TimeConversion = "utc2et",
    linear_anchor: TimeInput = "2023-12-01T00:00:00Z",
) -> MoonMeVectorSeries:
    """Resolve vectors, with explicit arrays overriding generation arguments."""
    if explicit_vectors_m is not None:
        timestamp_source = explicit_times if explicit_times is not None else times
        if timestamp_source is None:
            raise ValueError("explicit vectors require explicit_times or times")
        time_values = _times_tuple(timestamp_source)
        vectors = np.ascontiguousarray(explicit_vectors_m, dtype=np.float64)
        return MoonMeVectorSeries(time_values, vectors)
    time_values = _requested_times(times=times, start=start, stop=stop, step=step)
    return generate_moon_me_vectors(
        body,
        time_values,
        ensure_kernels=ensure_kernels,
        time_conversion=time_conversion,
        linear_anchor=linear_anchor,
    )
