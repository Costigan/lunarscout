#!/usr/bin/env python3
"""Capture deterministic Phase 1 reference-ray oracle artifacts."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import subprocess
import tempfile
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
PROJECT = "scripts/numba_horizon/CSharpPhase1OracleCapture.csproj"
DEFAULT_NPZ = REPO_ROOT / "tests/data/numba_horizon/phase1_reference_rays.npz"
DEFAULT_METADATA = REPO_ROOT / "tests/data/numba_horizon/phase1_reference_rays.json"
FIXED_ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
SEGMENT_FIELDS = (
    ("start_pixel_x", "pixel column"),
    ("start_pixel_y", "pixel row"),
    ("x0", "pixel column"),
    ("y0", "pixel row"),
    ("a1", "pixel/km"),
    ("a2", "pixel/km^2"),
    ("a3", "pixel/km^3"),
    ("a4", "pixel/km^4"),
    ("b1", "pixel/km"),
    ("b2", "pixel/km^2"),
    ("b3", "pixel/km^3"),
    ("b4", "pixel/km^4"),
    ("s_start_km", "km"),
    ("s_end_km", "km"),
    ("s_start_chord_km", "km"),
    ("planar_to_chord_c1", "m/m"),
    ("planar_to_chord_c2", "1/m"),
    ("planar_to_chord_c3", "1/m^2"),
)
MAP_PARAMETER_FIELDS = (
    ("radius_m", "m"),
    ("scale", "dimensionless"),
    ("false_easting_m", "m"),
    ("false_northing_m", "m"),
    ("inverse_transform_determinant", "1/CRS unit^2"),
    ("transform_0", "CRS x origin"),
    ("transform_1", "CRS x per pixel column"),
    ("transform_2", "CRS x per pixel row"),
    ("transform_3", "CRS y origin"),
    ("transform_4", "CRS y per pixel column"),
    ("transform_5", "CRS y per pixel row"),
)
PROJECTION_PARAMETER_FIELDS = (
    ("radius_m", "m"),
    ("latitude_origin_rad", "rad"),
    ("longitude_origin_rad", "rad"),
    ("scale", "dimensionless"),
    ("false_easting_m", "m"),
    ("false_northing_m", "m"),
)
SUBPATCH_CENTER_FIELDS = (
    "index",
    "grid_row",
    "grid_column",
    "requested_center_column",
    "requested_center_row",
    "segment_center_column",
    "segment_center_row",
)
TRAVERSAL_TRACE_FIELDS = (
    "parameter_distance_km",
    "true_distance_m",
    "level",
    "cell_x",
    "cell_y",
    "pixel_x",
    "pixel_y",
    "maximum_elevation_m",
    "sample_elevation_m",
    "sample_slope",
    "advance_km",
    "action",
)


def array_sha256(array: np.ndarray) -> str:
    canonical = np.ascontiguousarray(array)
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


def add_array(
    arrays: dict[str, np.ndarray],
    manifest: dict[str, Any],
    name: str,
    values: Any,
    *,
    dtype: str,
    axes: list[str],
    units: str,
) -> None:
    array = np.asarray(values, dtype=dtype)
    arrays[name] = array
    manifest[name] = {
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "axes": axes,
        "units": units,
        "sha256_c_order_data": array_sha256(array),
    }


def write_deterministic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name in sorted(arrays):
            buffer = io.BytesIO()
            np.lib.format.write_array(buffer, arrays[name], allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", FIXED_ZIP_TIMESTAMP)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            archive.writestr(info, buffer.getvalue(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)


def convert_pyramid(
    pyramid: dict[str, Any],
    prefix: str,
    arrays: dict[str, np.ndarray],
    manifest: dict[str, Any],
) -> None:
    levels = pyramid.pop("levels")
    level0 = np.asarray(pyramid.pop("level0"), dtype="<f4")
    mips = np.asarray(pyramid.pop("mips"), dtype="<f4")
    for level in levels:
        if level["level"] == 0:
            values = level0.reshape(level["height"], level["width"])
        else:
            start = level["offset"]
            stop = start + level["width"] * level["height"]
            values = mips[start:stop].reshape(level["height"], level["width"])
        add_array(
            arrays,
            manifest,
            f"{prefix}__level_{level['level']}",
            values,
            dtype="<f4",
            axes=["y", "x"],
            units="m maximum elevation; -32000 denotes an all-invalid block above level 0",
        )
    add_array(
        arrays,
        manifest,
        f"{prefix}__level_metadata",
        [
            [level["level"], level["offset"], level["width"], level["height"]]
            for level in levels
        ],
        dtype="<i8",
        axes=["level", "field_level_offset_width_height"],
        units="indices and element counts; offsets for levels above 0 are relative to the concatenated mip buffer",
    )
    add_array(
        arrays,
        manifest,
        f"{prefix}__level_cell_sizes",
        [[level["cell_size_x"], level["cell_size_y"]] for level in levels],
        dtype="<f4",
        axes=["level", "axis_xy"],
        units="currently unset production metadata",
    )
    map_parameters = pyramid.pop("map_parameters")
    pyramid["map_parameter_fields"] = [
        {"name": name, "units": units} for name, units in MAP_PARAMETER_FIELDS
    ]
    add_array(
        arrays,
        manifest,
        f"{prefix}__map_parameters",
        [map_parameters[name] for name, _ in MAP_PARAMETER_FIELDS],
        dtype="<f4",
        axes=["map_parameter_field"],
        units="mixed; see pyramid map_parameter_fields",
    )
    projection_parameters = pyramid.pop("projection_parameters")
    pyramid["projection_parameter_fields"] = [
        {"name": name, "units": units}
        for name, units in PROJECTION_PARAMETER_FIELDS
    ]
    add_array(
        arrays,
        manifest,
        f"{prefix}__projection_parameters",
        [projection_parameters[name] for name, _ in PROJECTION_PARAMETER_FIELDS],
        dtype="<f4",
        axes=["projection_parameter_field"],
        units="mixed; see pyramid projection_parameter_fields",
    )
    pyramid["level_count"] = len(levels)


def convert(raw: dict[str, Any]) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    arrays: dict[str, np.ndarray] = {}
    manifest: dict[str, Any] = {}
    case_metadata: list[dict[str, Any]] = []
    pyramid_fixture_metadata: list[dict[str, Any]] = []
    subpatch_fixture_metadata: list[dict[str, Any]] = []
    horizon_buffer_fixture_metadata: list[dict[str, Any]] = []

    for case in raw["cases"]:
        case_id = case["id"]
        for dem in case["dems"]:
            prefix = f"{case_id}__dem_{dem['index']}"
            elevation = np.asarray(dem.pop("elevation_m"), dtype="<f4").reshape(
                dem["height"], dem["width"]
            )
            add_array(
                arrays,
                manifest,
                f"{prefix}__elevation_m",
                elevation,
                dtype="<f4",
                axes=["y", "x"],
                units="m",
            )
            add_array(
                arrays,
                manifest,
                f"{prefix}__geo_transform",
                dem.pop("geo_transform"),
                dtype="<f8",
                axes=["coefficient"],
                units="mixed CRS transform coefficients; see conventions",
            )

        for pyramid in case["pyramids"]:
            prefix = f"{case_id}__dem_{pyramid['dem_index']}__pyramid"
            convert_pyramid(pyramid, prefix, arrays, manifest)

        for pass_index, pass_data in enumerate(case["passes"]):
            prefix = f"{case_id}__pass_{pass_index}"
            trace = pass_data.pop("trace")
            slopes = pass_data.pop("slopes")
            add_array(arrays, manifest, f"{prefix}__slopes", slopes, dtype="<f8", axes=["sample"], units="dimensionless")
            for field, units in (
                ("distance_m", "m"),
                ("elevation_m", "m"),
                ("slope", "dimensionless"),
                ("pixel_x", "pixel column"),
                ("pixel_y", "pixel row"),
            ):
                add_array(
                    arrays,
                    manifest,
                    f"{prefix}__trace_{field}",
                    [sample[field] for sample in trace],
                    dtype="<f8",
                    axes=["sample"],
                    units=units,
                )
            add_array(
                arrays,
                manifest,
                f"{prefix}__direction_me",
                pass_data.pop("direction_me"),
                dtype="<f8",
                axes=["moon_centered_component_xyz"],
                units="dimensionless unit vector",
            )

        for fit_index, fit_data in enumerate(case["ray_fit_passes"]):
            prefix = f"{case_id}__ray_fit_pass_{fit_index}"
            samples = fit_data.pop("samples")
            for field, units in (
                ("distance_m", "m"),
                ("pixel_x", "pixel column"),
                ("pixel_y", "pixel row"),
                ("latitude_rad", "rad"),
                ("longitude_rad", "rad"),
                ("row", "pixel row"),
                ("column", "pixel column"),
                ("terrain_height_m", "m"),
            ):
                add_array(
                    arrays,
                    manifest,
                    f"{prefix}__sample_{field}",
                    [sample[field] for sample in samples],
                    dtype="<f8",
                    axes=["sample"],
                    units=units,
                )
            add_array(
                arrays,
                manifest,
                f"{prefix}__observer_vector_moon_centered_m",
                fit_data.pop("observer_vector_moon_centered_m"),
                dtype="<f8",
                axes=["moon_centered_component_xyz"],
                units="m",
            )
            add_array(
                arrays,
                manifest,
                f"{prefix}__nominal_direction_moon_centered",
                fit_data.pop("nominal_direction_moon_centered"),
                dtype="<f8",
                axes=["moon_centered_component_xyz"],
                units="dimensionless unit vector",
            )
            segment = fit_data.pop("segment")
            fit_data["segment_dem_id"] = segment.pop("dem_id")
            fit_data["segment_fields"] = [
                {"name": name, "units": units} for name, units in SEGMENT_FIELDS
            ]
            add_array(
                arrays,
                manifest,
                f"{prefix}__segment_values",
                [segment[name] for name, _ in SEGMENT_FIELDS],
                dtype="<f4",
                axes=["segment_field"],
                units="mixed; see case ray_fit_passes segment_fields",
            )
            fit_data["sample_count"] = len(samples)
        case_metadata.append(case)

    for fixture in raw["pyramid_fixtures"]:
        prefix = f"{fixture['id']}__pyramid_fixture"
        elevation = np.asarray(fixture.pop("elevation_m"), dtype="<f4").reshape(
            fixture["height"], fixture["width"]
        )
        add_array(
            arrays,
            manifest,
            f"{prefix}__elevation_m",
            elevation,
            dtype="<f4",
            axes=["y", "x"],
            units="m",
        )
        convert_pyramid(fixture["pyramid"], prefix, arrays, manifest)
        pyramid_fixture_metadata.append(fixture)

    for fixture in raw["subpatch_fixtures"]:
        prefix = f"{fixture['id']}__subpatch_fixture"
        configuration = fixture["configuration"]
        centers = fixture.pop("centers")
        fixture["center_fields"] = list(SUBPATCH_CENTER_FIELDS)
        add_array(
            arrays,
            manifest,
            f"{prefix}__centers",
            [[center[field] for field in SUBPATCH_CENTER_FIELDS] for center in centers],
            dtype="<i4",
            axes=["subpatch_center", "center_field"],
            units="pixel indices and coordinates; see center_fields",
        )
        grid_convergence = fixture.pop("grid_convergence")
        fixture["grid_convergence_fields"] = list(grid_convergence)
        add_array(
            arrays,
            manifest,
            f"{prefix}__grid_convergence",
            list(grid_convergence.values()),
            dtype="<f4",
            axes=["grid_convergence_field"],
            units="rad or rad/pixel; see grid_convergence_fields",
        )
        segments = fixture.pop("segments")
        expected_count = (
            configuration["azimuth_count"]
            * len(centers)
            * configuration["dem_count"]
        )
        if len(segments) != expected_count or fixture["segment_count"] != expected_count:
            raise ValueError(
                f"subpatch fixture {fixture['id']} has {len(segments)} segments; "
                f"expected {expected_count}"
            )
        fixture["segment_fields"] = [
            {"name": name, "units": units} for name, units in SEGMENT_FIELDS
        ]
        shape = (
            configuration["azimuth_count"],
            len(centers),
            configuration["dem_count"],
        )
        add_array(
            arrays,
            manifest,
            f"{prefix}__segment_values",
            np.asarray(
                [[segment[name] for name, _ in SEGMENT_FIELDS] for segment in segments],
                dtype="<f4",
            ).reshape(*shape, len(SEGMENT_FIELDS)),
            dtype="<f4",
            axes=["azimuth", "subpatch_center", "dem", "segment_field"],
            units="mixed; see segment_fields",
        )
        add_array(
            arrays,
            manifest,
            f"{prefix}__segment_dem_ids",
            np.asarray([segment["dem_id"] for segment in segments], dtype="<i4").reshape(shape),
            dtype="<i4",
            axes=["azimuth", "subpatch_center", "dem"],
            units="logical DEM index",
        )
        fixture["center_count"] = len(centers)
        subpatch_fixture_metadata.append(fixture)

    for fixture in raw["horizon_buffer_fixtures"]:
        prefix = f"{fixture['id']}__horizon_buffer_fixture"
        configuration = fixture["configuration"]
        for dem in fixture["dems"]:
            dem_prefix = f"{prefix}__dem_{dem['index']}"
            add_array(
                arrays,
                manifest,
                f"{dem_prefix}__elevation_m",
                np.asarray(dem.pop("elevation_m"), dtype="<f4").reshape(
                    dem["height"], dem["width"]
                ),
                dtype="<f4",
                axes=["y", "x"],
                units="m",
            )
            add_array(
                arrays,
                manifest,
                f"{dem_prefix}__geo_transform",
                dem["geo_transform"],
                dtype="<f8",
                axes=["geo_transform_element"],
                units="mixed affine CRS units",
            )
            convert_pyramid(
                dem["pyramid"],
                f"{dem_prefix}__pyramid",
                arrays,
                manifest,
            )
        shape = (
            configuration["tile_width"] * configuration["tile_height"],
            configuration["azimuth_count"],
        )
        per_dem = np.asarray(fixture.pop("per_dem_slopes"), dtype="<f4").reshape(
            configuration["dem_count"], *shape
        )
        add_array(
            arrays,
            manifest,
            f"{prefix}__per_dem_slopes",
            per_dem,
            dtype="<f4",
            axes=["dem", "pixel", "azimuth"],
            units="dimensionless slope",
        )
        add_array(
            arrays,
            manifest,
            f"{prefix}__final_slopes",
            np.asarray(fixture.pop("final_slopes"), dtype="<f4").reshape(shape),
            dtype="<f4",
            axes=["pixel", "azimuth"],
            units="dimensionless slope",
        )
        add_array(
            arrays,
            manifest,
            f"{prefix}__final_degrees",
            np.asarray(fixture.pop("final_degrees"), dtype="<f4").reshape(shape),
            dtype="<f4",
            axes=["pixel", "azimuth"],
            units="degree",
        )
        grid_convergence = fixture.pop("grid_convergence")
        fixture["grid_convergence_fields"] = list(grid_convergence)
        add_array(
            arrays,
            manifest,
            f"{prefix}__grid_convergence",
            list(grid_convergence.values()),
            dtype="<f4",
            axes=["grid_convergence_field"],
            units="rad or rad/pixel; see grid_convergence_fields",
        )
        traversal_trace = fixture.pop("traversal_trace")
        steps = traversal_trace.pop("steps")
        traversal_trace["fields"] = list(TRAVERSAL_TRACE_FIELDS)
        traversal_trace["step_count"] = len(steps)
        add_array(
            arrays,
            manifest,
            f"{prefix}__traversal_trace",
            [[float(step[field]) for field in TRAVERSAL_TRACE_FIELDS] for step in steps],
            dtype="<f4",
            axes=["step", "traversal_field"],
            units="mixed; see traversal_trace.fields",
        )
        fixture["traversal_trace"] = traversal_trace
        horizon_buffer_fixture_metadata.append(fixture)

    metadata = {
        "schema_version": 1,
        "artifact_kind": "lunarscout_phase1_reference_ray_oracles",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "baseline_commit": raw["baseline_commit"],
        "source_implementation": raw["implementation"],
        "pyramid_source_implementation": raw["pyramid_implementation"],
        "capture_accelerator": raw["accelerator"],
        "source_capture_project": PROJECT,
        "capture_script": str(Path(__file__).resolve().relative_to(REPO_ROOT)),
        "conventions": raw["conventions"],
        "cases": case_metadata,
        "pyramid_fixtures": pyramid_fixture_metadata,
        "subpatch_fixtures": subpatch_fixture_metadata,
        "horizon_buffer_fixtures": horizon_buffer_fixture_metadata,
        "arrays": manifest,
    }
    return arrays, metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-commit", required=True)
    parser.add_argument("--npz", type=Path, default=DEFAULT_NPZ)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    with tempfile.TemporaryDirectory(prefix="lunarscout-phase1-oracles-") as directory:
        raw_path = Path(directory) / "reference-oracles.json"
        command = ["dotnet", "run", "--project", PROJECT, "--", str(raw_path)]
        result = subprocess.run(
            command,
            cwd=REPO_ROOT,
            env={**os.environ, "LUNARSCOUT_BASELINE_COMMIT": args.baseline_commit},
            capture_output=True,
            check=False,
            text=True,
            timeout=300,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"C# oracle capture failed with exit code {result.returncode}:\n"
                f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
            )
        raw = json.loads(raw_path.read_text(encoding="utf-8"))

    arrays, metadata = convert(raw)
    write_deterministic_npz(args.npz, arrays)
    metadata["npz_sha256"] = hashlib.sha256(args.npz.read_bytes()).hexdigest()
    args.metadata.parent.mkdir(parents=True, exist_ok=True)
    args.metadata.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(args.npz)
    print(args.metadata)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
