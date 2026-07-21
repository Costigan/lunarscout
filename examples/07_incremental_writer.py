"""Build a file-backed temporal series incrementally without a TemporalCube.

Inputs: deterministic synthetic DEM under --workspace.
Outputs: analysis/incremental_sun.temporal with manifest, VRT, and layer TIFFs.
Resources: small fixture; demonstrates progress, cancellation, and layer reads
without a GPU.
"""

from __future__ import annotations

import lunarscout as ls
import numpy as np

from _example_support import (
    ensure_synthetic_scenario,
    example_parser,
    synthetic_times,
)


def _on_progress(twp: "ls.TemporalWriteProgress") -> None:
    time_str = str(twp.last_time)
    print(f"  wrote layer {twp.layers_written} ({time_str}) -> {twp.layer_path.name}")


def main() -> None:
    args = example_parser(__doc__ or "").parse_args()
    scenario = ensure_synthetic_scenario(args.workspace)
    _dem, georef = ls.read_geotiff(scenario.dem_path())
    if georef is None:
        raise RuntimeError("Synthetic DEM unexpectedly lacks georeferencing.")

    times = synthetic_times()
    output = scenario.output_path("analysis/incremental_sun.temporal")

    print(f"Writing {times.time_count} layers to {output} ...")

    with ls.TemporalGeoTiffSeriesWriter(
        output,
        georef=georef,
        dtype=np.float32,
        signal_name="incremental_sun",
        units="fraction",
        provenance={"source": "Lunarscout incremental writer example"},
        progress_callback=_on_progress,
    ) as writer:
        rows, columns = np.indices((georef.height, georef.width), dtype=np.float32)
        for index in range(times.time_count):
            time = _utc_datetime_for(times, index)
            layer = np.clip(
                (0.35
                 + columns / max(1, georef.width - 1) * 0.45
                 + 0.08 * np.sin(index * np.pi / 3)
                 - rows * 0.001).astype(np.float32),
                0.0,
                1.0,
            )
            writer.write_layer(time, layer)

    series = writer.result
    if series is None:
        raise RuntimeError("Writer did not produce a series.")

    print(f"Completed: {series.root}")
    print(f"  shape={series.shape}, dtype={series.dtype}")
    print(f"  signal={series.signal_name}, units={series.units}")
    print(f"  VRT={series.vrt_path}")

    print("Reading back layers:")
    for index in range(series.time_count):
        time = series.time_for_layer(index)
        arr, _ = series.read_layer(index)
        print(
            f"  layer {index}: {time}  "
            f"min={arr.min():.4f}  max={arr.max():.4f}  mean={arr.mean():.4f}"
        )

    print("Nearest-time lookup:")
    nearest, _ = series.read_time("2027-01-01T02:20:00Z", method="nearest")
    print(f"  mean={nearest.mean():.4f}")

    series.close()


def _utc_datetime_for(times: ls.TimeRange, index: int) -> ...:
    """Return a UTC datetime for the given zero-based index into a TimeRange."""
    from datetime import datetime, timezone

    ts = times.values[index]
    return datetime.fromtimestamp(
        ts.astype("datetime64[us]").astype("int64") / 1_000_000, tz=timezone.utc
    )


if __name__ == "__main__":
    main()
