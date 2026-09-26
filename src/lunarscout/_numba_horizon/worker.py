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
import sys
import traceback
from pathlib import Path

from ..horizon import generate_horizons


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
) -> Path:
    """Generate the horizon patches assigned to one partition of a batch.

    Pod ``pod_nth`` of ``pod_count`` processes every patch whose enumeration
    index is congruent to ``pod_nth`` modulo ``pod_count``, so ``pod_count``
    pods cover every patch exactly once.  The first DEM in ``primary_dem`` plus
    ``surrounding_dems`` defines the output grid and extended terrain, exactly
    as in :func:`lunarscout.generate_horizons`.
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
    surrounding = [p for p in args.surrounding_dems.split(",") if p.strip()]

    original_stdout = sys.stdout
    original_stderr = sys.stderr
    log_handle = None
    if args.log_dir:
        log_path = Path(args.log_dir) / f"pod-{pod_nth}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_handle = log_path.open("w", buffering=1)
        sys.stdout = _Tee(original_stdout, log_handle)
        sys.stderr = _Tee(original_stderr, log_handle)
        print(f"pod {pod_nth}/{args.pod_count}: logging to {log_path}")

    try:
        result = run_horizon_partition(
            args.primary_dem,
            args.output,
            surrounding_dems=surrounding,
            observer_height_m=args.observer_height_m,
            compress=args.compress,
            pod_count=args.pod_count,
            pod_nth=pod_nth,
        )
        print("horizons written to", result)
        return 0
    except BaseException:
        traceback.print_exc()
        return 1
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        if log_handle is not None:
            log_handle.close()


if __name__ == "__main__":
    raise SystemExit(main())
