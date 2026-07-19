"""Private patch-major landed mission-duration product functions."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime
import hashlib
from pathlib import Path
from typing import Any, Literal, TextIO

import numpy as np
import numpy.typing as npt

from lunarscout.georeference import GeoReference

from .file_format import HorizonTileStore
from .geometry import DemGrid
from .mission_duration import (
    CandidateStartInterval,
    DurationUnit,
    _utc,
    candidate_start_intervals,
    reduce_longest_candidate_duration_stream,
    validate_evaluation_samples,
)
from .pipeline import PatchDescriptor, enumerate_patches
from .product_store import ProductJob, ResumableTiledProduct
from .psr import _validate_vectors
from .psr_pipeline import _inventory_identity


Backend = Literal["auto", "cpu", "cuda"]
SignalCalculator = Callable[..., Iterable[npt.ArrayLike]]


class MissionDurationPipelineCancelled(RuntimeError):
    """Cancellation observed between bounded mission-duration work units."""


@dataclass(frozen=True, slots=True)
class MissionDurationProgress:
    completed_patches: int
    total_patches: int
    tile_y: int | None
    tile_x: int | None
    state: str
    backend: Literal["cpu", "cuda"] | None = None


def _vector_hash(vectors: npt.NDArray[np.float64]) -> str:
    return hashlib.sha256(vectors.astype("<f8", copy=False).tobytes()).hexdigest()


def _sessions(
    backend: Backend, count: int, time_batch_size: int
) -> tuple[list[Any], Literal["cpu", "cuda"]]:
    if backend not in ("auto", "cpu", "cuda"):
        raise ValueError("backend must be 'auto', 'cpu', or 'cuda'")
    if time_batch_size < 1:
        raise ValueError("time_batch_size must be positive")
    if backend == "cpu":
        from .lightmap_cpu import LightmapCpuSession

        return (
            [LightmapCpuSession(time_batch_size=time_batch_size) for _ in range(count)],
            "cpu",
        )
    from .cuda_backend import CudaBackendError
    from .lightmap_cuda import LightmapCudaSession

    try:
        return (
            [LightmapCudaSession(time_batch_size=time_batch_size) for _ in range(count)],
            "cuda",
        )
    except CudaBackendError:
        if backend == "cuda":
            raise
        from .lightmap_cpu import LightmapCpuSession

        return (
            [LightmapCpuSession(time_batch_size=time_batch_size) for _ in range(count)],
            "cpu",
        )


def _run_duration_product(
    *,
    algorithm: str,
    dem: DemGrid,
    georef: GeoReference,
    horizon_store: HorizonTileStore,
    output_path: str | Path,
    times_utc: Sequence[datetime | str],
    evaluation_start_utc: datetime | str,
    evaluation_stop_utc: datetime | str,
    start_intervals: Sequence[
        CandidateStartInterval | tuple[datetime | str, datetime | str]
    ],
    sun_vectors_m: npt.ArrayLike,
    sun_signal: Literal["fraction", "margin"],
    sun_threshold: float,
    earth_vectors_m: npt.ArrayLike | None = None,
    earth_threshold_deg: float | None = None,
    output_unit: DurationUnit = "hours",
    observer_elevation_m: float = 0.0,
    invalid_value: float = 0.0,
    overwrite: bool = False,
    start_fresh: bool = False,
    cancellation_requested: Callable[[], bool] | None = None,
    progress_callback: Callable[[MissionDurationProgress], None] | None = None,
    progress_stream: TextIO | None = None,
    backend: Backend = "auto",
    time_batch_size: int = 32,
    _sun_calculator: SignalCalculator | None = None,
    _earth_calculator: SignalCalculator | None = None,
) -> Path:
    if (georef.width, georef.height) != (dem.width, dem.height):
        raise ValueError("GeoReference and DEM dimensions do not match")
    if output_unit not in ("hours", "days"):
        raise ValueError("output_unit must be 'hours' or 'days'")
    timestamps, _sample_hours = validate_evaluation_samples(
        times_utc,
        evaluation_start_utc=evaluation_start_utc,
        evaluation_stop_utc=evaluation_stop_utc,
    )
    intervals = candidate_start_intervals(
        start_intervals,
        evaluation_start_utc=evaluation_start_utc,
        evaluation_stop_utc=evaluation_stop_utc,
    )
    sun_vectors = _validate_vectors(sun_vectors_m)
    evaluation_start = _utc(evaluation_start_utc)
    evaluation_stop = _utc(evaluation_stop_utc)
    if len(timestamps) != len(sun_vectors):
        raise ValueError("timestamps and Sun vectors must align")
    if not np.isfinite(sun_threshold):
        raise ValueError("Sun threshold must be finite")
    if sun_signal == "fraction" and not 0.0 <= sun_threshold <= 1.0:
        raise ValueError("sunlight-fraction threshold must be between zero and one")
    earth_vectors = None
    if earth_vectors_m is not None:
        earth_vectors = _validate_vectors(earth_vectors_m)
        if len(earth_vectors) != len(timestamps):
            raise ValueError("timestamps and Earth vectors must align")
        if earth_threshold_deg is None or not np.isfinite(earth_threshold_deg):
            raise ValueError("Earth elevation threshold must be finite")
    elif earth_threshold_deg is not None:
        raise ValueError("Earth vectors are required with an Earth threshold")

    needed_sessions = int(_sun_calculator is None) + int(
        earth_vectors is not None and _earth_calculator is None
    )
    sessions, session_backend = _sessions(
        backend, needed_sessions, time_batch_size
    )
    selected_backend: Literal["cpu", "cuda"] | None = (
        session_backend if needed_sessions else None
    )
    session_index = 0
    if _sun_calculator is None:
        sun_session = sessions[session_index]
        session_index += 1
        sun_calculator = (
            sun_session.iter_patch_fraction_tiles
            if sun_signal == "fraction"
            else sun_session.iter_patch_margin_tiles
        )
    else:
        sun_calculator = _sun_calculator
    if earth_vectors is not None:
        if _earth_calculator is None:
            earth_calculator = sessions[session_index].iter_patch_margin_tiles
        else:
            earth_calculator = _earth_calculator
    else:
        earth_calculator = None

    patches = enumerate_patches(dem.width, dem.height)
    inventory = _inventory_identity(horizon_store, patches, observer_elevation_m)
    configuration: dict[str, Any] = {
        "evaluation_start_utc": evaluation_start.isoformat(),
        "evaluation_stop_utc": evaluation_stop.isoformat(),
        "candidate_start_intervals": [
            [item.start_utc.isoformat(), item.stop_utc.isoformat()]
            for item in intervals
        ],
        "output_unit": output_unit,
        "sun_signal": sun_signal,
        "sun_threshold": float(sun_threshold),
        "sun_vectors_sha256": _vector_hash(sun_vectors),
        "observer_elevation_m": float(observer_elevation_m),
    }
    if earth_vectors is not None:
        configuration.update(
            {
                "earth_threshold_deg": float(earth_threshold_deg),
                "earth_vectors_sha256": _vector_hash(earth_vectors),
            }
        )
    product = ResumableTiledProduct(
        output_path,
        ProductJob(
            georef=georef,
            dtype=np.float32,
            band_count=len(intervals),
            timestamps_utc=tuple(item.start_utc for item in intervals),
            band_metadata=tuple(
                {
                    "CANDIDATE_START_UTC": item.start_utc.isoformat(),
                    "CANDIDATE_STOP_UTC": item.stop_utc.isoformat(),
                    "DURATION_UNIT": output_unit,
                }
                for item in intervals
            ),
            invalid_value=invalid_value,
            algorithm=algorithm,
            configuration=configuration,
            horizon_inventory_identity=inventory,
        ),
        overwrite=overwrite,
        start_fresh=start_fresh,
        backend=selected_backend,
    )

    def cancelled() -> bool:
        return bool(cancellation_requested and cancellation_requested())

    completed = len(product.completed_patches)

    def report(patch: PatchDescriptor | None, state: str) -> None:
        event = MissionDurationProgress(
            completed,
            len(patches),
            None if patch is None else patch.tile_y,
            None if patch is None else patch.tile_x,
            state,
            selected_backend,
        )
        if progress_callback is not None:
            progress_callback(event)
        if progress_stream is not None:
            location = "" if patch is None else f" row={patch.tile_y} col={patch.tile_x}"
            print(
                f"Mission duration {state}:{location} {completed}/{len(patches)} patches",
                file=progress_stream,
                flush=True,
            )

    report(None, "start")
    for patch in patches:
        if product.is_complete(patch.tile_y, patch.tile_x):
            continue
        if cancelled():
            raise MissionDurationPipelineCancelled(
                "mission-duration generation was cancelled"
            )
        report(patch, "read")
        try:
            horizons = horizon_store.read(
                patch.tile_y, patch.tile_x, observer_elevation_m
            )
        except (OSError, ValueError):
            horizons = None
        if horizons is None:
            product.write_invalid_patch(patch.tile_y, patch.tile_x)
            state = "invalid"
        else:
            report(patch, "calculate")
            sun_tiles = sun_calculator(
                dem,
                horizons,
                sun_vectors,
                tile_y=patch.tile_y,
                tile_x=patch.tile_x,
                valid_height=patch.height,
                valid_width=patch.width,
            )
            earth_tiles = (
                earth_calculator(
                    dem,
                    horizons,
                    earth_vectors,
                    tile_y=patch.tile_y,
                    tile_x=patch.tile_x,
                    valid_height=patch.height,
                    valid_width=patch.width,
                )
                if earth_calculator is not None and earth_vectors is not None
                else None
            )

            def conditions() -> Iterable[npt.NDArray[np.bool_]]:
                if earth_tiles is None:
                    for sun_tile in sun_tiles:
                        if cancelled():
                            raise MissionDurationPipelineCancelled(
                                "mission-duration generation was cancelled"
                            )
                        yield np.asarray(sun_tile) >= sun_threshold
                else:
                    assert earth_threshold_deg is not None
                    for sun_tile, earth_tile in zip(
                        sun_tiles, earth_tiles, strict=True
                    ):
                        if cancelled():
                            raise MissionDurationPipelineCancelled(
                                "mission-duration generation was cancelled"
                            )
                        yield (np.asarray(sun_tile) >= sun_threshold) & (
                            np.asarray(earth_tile) >= earth_threshold_deg
                        )

            duration_tiles = reduce_longest_candidate_duration_stream(
                conditions(),
                times_utc=timestamps,
                evaluation_start_utc=evaluation_start,
                evaluation_stop_utc=evaluation_stop,
                start_intervals=intervals,
                output_unit=output_unit,
            )
            if cancelled():
                raise MissionDurationPipelineCancelled(
                    "mission-duration generation was cancelled"
                )
            report(patch, "write")
            product.write_patch(patch.tile_y, patch.tile_x, duration_tiles)
            state = "valid"
        completed += 1
        report(patch, state)
    if cancelled():
        raise MissionDurationPipelineCancelled(
            "mission-duration generation was cancelled"
        )
    result = product.finalize()
    report(None, "complete")
    return result


def run_sunlight_duration_product(*, sunlight_fraction_threshold: float, **kwargs) -> Path:
    """Longest continuous sunlight-fraction duration by candidate-start interval."""
    return _run_duration_product(
        algorithm="landed-mission-sunlight-duration",
        sun_signal="fraction",
        sun_threshold=sunlight_fraction_threshold,
        **kwargs,
    )


def run_sun_elevation_duration_product(*, sun_elevation_threshold_deg: float, **kwargs) -> Path:
    """Longest continuous Sun-center local-horizon margin duration."""
    return _run_duration_product(
        algorithm="landed-mission-sun-elevation-duration",
        sun_signal="margin",
        sun_threshold=sun_elevation_threshold_deg,
        **kwargs,
    )


def run_sunlight_earth_elevation_duration_product(
    *,
    sunlight_fraction_threshold: float,
    earth_elevation_threshold_deg: float,
    earth_vectors_m: npt.ArrayLike,
    **kwargs,
) -> Path:
    """Longest duration satisfying sunlight fraction and Earth margin."""
    return _run_duration_product(
        algorithm="landed-mission-sunlight-earth-elevation-duration",
        sun_signal="fraction",
        sun_threshold=sunlight_fraction_threshold,
        earth_vectors_m=earth_vectors_m,
        earth_threshold_deg=earth_elevation_threshold_deg,
        **kwargs,
    )


def run_sun_elevation_earth_elevation_duration_product(
    *,
    sun_elevation_threshold_deg: float,
    earth_elevation_threshold_deg: float,
    earth_vectors_m: npt.ArrayLike,
    **kwargs,
) -> Path:
    """Longest duration satisfying Sun-center and Earth-center margins."""
    return _run_duration_product(
        algorithm="landed-mission-sun-earth-elevation-duration",
        sun_signal="margin",
        sun_threshold=sun_elevation_threshold_deg,
        earth_vectors_m=earth_vectors_m,
        earth_threshold_deg=earth_elevation_threshold_deg,
        **kwargs,
    )
