"""Partitioned horizon worker for scheduler-free batch runs.

Each worker process runs :func:`run_horizon_partition` over one strided share of
the patch list; the existing skip-if-exists logic coordinates resume across
pods through the shared output directory.  This is the entry point behind the
``lunarscout-horizon-worker`` console script used by the Kubernetes Job in
``deploy/horizon-job.yaml``.
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
import threading
import traceback
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

from ..errors import OperationCancelledError
from ..horizon import generate_horizons

# Exit status for a pod stopped by SIGTERM (128 + 15), as a shell reports it.
EXIT_TERMINATED = 143
# Staging files older than this are orphans of killed writers; a live
# writer holds its staging file for seconds.
STALE_STAGING_SECONDS = 3600.0


class _Tee:
    """Write to several streams so output reaches both the pod log and a durable
    shared-filesystem file."""

    def __init__(self, *streams) -> None:
        self._streams = streams

    def write(self, data: str) -> int:
        for stream in self._streams:
            stream.write(data)
        return len(data)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()


def run_horizon_partition(
    primary_dem: str | Path,
    output: str | Path,
    *,
    surrounding_dems: list[str | Path] | None = None,
    observer_height_m: float = 0.0,
    compress: bool = False,
    pod_count: int,
    pod_nth: int,
    cancellation_requested: Callable[[], bool] | None = None,
) -> Path:
    """Generate the horizon patches assigned to one partition of a batch.

    Pod ``pod_nth`` of ``pod_count`` processes every patch whose enumeration
    index is congruent to ``pod_nth`` modulo ``pod_count``, so ``pod_count``
    pods cover every patch exactly once.  The first DEM in ``primary_dem`` plus
    ``surrounding_dems`` defines the output grid and extended terrain, exactly
    as in :func:`lunarscout.generate_horizons`.

    A patch whose own calculation fails is reported after the rest of the
    partition has been generated, so a retry repeats only that patch.
    """
    dem_paths = [str(primary_dem)] + [str(path) for path in (surrounding_dems or [])]
    return generate_horizons(
        output,
        dem_paths,
        observer_height_m=observer_height_m,
        compress=compress,
        patch_offset=pod_nth,
        patch_stride=pod_count,
        verbose=True,
        cancellation_requested=cancellation_requested,
        skip_failed_patches=True,
    )


def _pod_nth_from_env() -> int:
    return int(
        os.environ.get("JOB_COMPLETION_INDEX", os.environ.get("POD_NTH", "0"))
    )


def _env_bool(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def _env_int(name: str) -> int | None:
    value = os.environ.get(name)
    return int(value) if value else None


def _node_label() -> str:
    pod = os.environ.get("HOSTNAME") or os.environ.get("POD_NAME")
    node = os.environ.get("NODE_NAME")
    parts: list[str] = []
    if pod:
        parts.append(pod)
    if node:
        parts.append(f"node={node}")
    return " ".join(parts) if parts else "unknown node"


def _describe_gpu() -> str:
    """One-line GPU identity for correlating failures with specific hardware."""
    try:
        from ..cuda import status
    except Exception as exc:  # pragma: no cover - import should always work
        return f"gpu: identity unavailable ({exc})"
    result = status()
    if not result.available:
        return f"gpu: unavailable ({result.reason})"
    capability = (
        f"sm_{result.compute_capability[0]}{result.compute_capability[1]}"
        if result.compute_capability
        else "unknown"
    )
    memory = (
        f"{result.total_memory_bytes / 2**30:.1f} GiB"
        if result.total_memory_bytes
        else "unknown"
    )
    driver = result.cuda_driver_version or "unknown"
    return (
        f"gpu: {result.device_name} ({capability}) "
        f"driver {driver} memory {memory}"
    )


def _install_sigterm_handler() -> threading.Event:
    """Turn SIGTERM into a cancellation request checked between patches.

    As PID 1 in a container, Python would otherwise ignore SIGTERM and be
    killed only when the grace period ends, possibly mid-write.
    """
    terminated = threading.Event()

    def handle(_signum, _frame) -> None:
        terminated.set()

    signal.signal(signal.SIGTERM, handle)
    return terminated


def _coverage(primary_dem: str | Path, output: str | Path, observer_height_m: float) -> str:
    """Report how many of the primary DEM's tiles exist, across all partitions."""
    import rasterio

    from .file_format import HorizonTileStore
    from .pipeline import enumerate_patches

    try:
        with rasterio.open(primary_dem) as dataset:
            width, height = dataset.width, dataset.height
        expected = {(patch.tile_y, patch.tile_x) for patch in enumerate_patches(width, height)}
        present = HorizonTileStore(output).list_existing_tiles(observer_height_m)
    except Exception as exc:  # the report must never mask the run's own result
        return f"coverage: unavailable ({exc})"
    found = len(expected & present)
    return (
        f"coverage: {found} of {len(expected)} tiles present across all partitions"
        + ("" if found == len(expected) else
           "; if every pod has finished, check POD_COUNT against the Job's completions")
    )


def main(argv: list[str] | None = None) -> int:
    # Every argument falls back to an environment variable so the Kubernetes
    # Job can provide the scenario through a ConfigMap (envFrom) while the same
    # command stays runnable locally with explicit flags.
    parser = argparse.ArgumentParser(
        description="Process one partition of the horizon patch list.",
    )
    parser.add_argument(
        "--primary-dem", default=os.environ.get("PRIMARY_DEM"), help="Primary DEM path."
    )
    parser.add_argument(
        "--surrounding-dems",
        default=os.environ.get("SURROUNDING_DEMS", ""),
        help="Comma-separated surrounding DEM paths.",
    )
    parser.add_argument(
        "--output", default=os.environ.get("OUTPUT_DIR"), help="Shared output directory."
    )
    parser.add_argument(
        "--observer-height-m",
        type=float,
        default=float(os.environ.get("OBSERVER_HEIGHT_M", "0")),
    )
    parser.add_argument(
        "--compress", action="store_true", default=_env_bool("COMPRESS")
    )
    parser.add_argument(
        "--log-dir",
        default=os.environ.get("LOG_DIR"),
        help="Optional shared directory for per-pod logs; writes pod-<nth>.log.",
    )
    parser.add_argument(
        "--pod-count",
        type=int,
        default=_env_int("POD_COUNT"),
        help="Total number of worker pods.",
    )
    parser.add_argument(
        "--pod-nth",
        type=int,
        default=None,
        help="This pod's 0-based index; defaults to $JOB_COMPLETION_INDEX.",
    )
    args = parser.parse_args(argv)

    if args.primary_dem is None:
        parser.error("--primary-dem or $PRIMARY_DEM is required")
    if args.output is None:
        parser.error("--output or $OUTPUT_DIR is required")
    if args.pod_count is None:
        parser.error("--pod-count or $POD_COUNT is required")

    pod_nth = args.pod_nth if args.pod_nth is not None else _pod_nth_from_env()
    if args.pod_count < 1 or not 0 <= pod_nth < args.pod_count:
        # Kubernetes cannot pass the Job's completions to the pod, so an
        # index outside POD_COUNT is the only mismatch a pod can detect.
        parser.error(
            f"pod index {pod_nth} is outside 0..POD_COUNT-1 (POD_COUNT={args.pod_count}); "
            "POD_COUNT must equal the Job's completions"
        )
    surrounding = [p for p in args.surrounding_dems.split(",") if p.strip()]

    original_stdout = sys.stdout
    original_stderr = sys.stderr
    log_handle = None
    if args.log_dir:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        log_path = Path(args.log_dir) / f"pod-{pod_nth}-{stamp}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        # A timestamped filename keeps each attempt's log unique, so a retried
        # pod cannot clobber a previous attempt's output.
        log_handle = log_path.open("w", buffering=1)
        sys.stdout = _Tee(original_stdout, log_handle)
        sys.stderr = _Tee(original_stderr, log_handle)
        print(f"pod {pod_nth}/{args.pod_count}: logging to {log_path.name}")

    print(f"worker: {_node_label()}")
    print(_describe_gpu())
    terminated = _install_sigterm_handler()
    from .file_format import HorizonTileStore

    removed = HorizonTileStore(args.output).remove_stale_staging_files(
        STALE_STAGING_SECONDS
    )
    if removed:
        print(f"removed {removed} orphaned staging files older than one hour")

    try:
        result = run_horizon_partition(
            args.primary_dem,
            args.output,
            surrounding_dems=surrounding,
            observer_height_m=args.observer_height_m,
            compress=args.compress,
            pod_count=args.pod_count,
            pod_nth=pod_nth,
            cancellation_requested=terminated.is_set,
        )
        print(f"SUCCESS: pod {pod_nth}/{args.pod_count} finished; horizons in {result}")
        print(_coverage(args.primary_dem, args.output, args.observer_height_m))
        return 0
    except OperationCancelledError:
        # SIGTERM is the worker's only cancellation source.
        print(f"TERMINATED: pod {pod_nth}/{args.pod_count} stopped by SIGTERM between patches")
        return EXIT_TERMINATED
    except BaseException:
        print(f"FAILED: pod {pod_nth}/{args.pod_count} raised the following:")
        traceback.print_exc()
        return 1
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        if log_handle is not None:
            log_handle.close()


if __name__ == "__main__":
    raise SystemExit(main())
