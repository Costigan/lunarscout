"""CPU and CUDA backends agree on every horizon-derived product (review finding M-F).

The backends are not bit-identical: libNVVM contracts multiply-adds into FMAs
and the GPU's trigonometric functions differ from libm.  These tests run a
full 128 x 128 patch of rough synthetic terrain with real generated horizons
and 192 Sun and Earth positions that sweep every azimuth close to the horizon,
and bound the differences: rare one-count lightmap differences, rare
threshold-edge flips, and elevation differences that are only large where
the horizon jumps steeply between adjacent azimuth bins at the DEM edge.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import os
from pathlib import Path

import numpy as np
import pytest
import rasterio

import lunarscout as ls
from lunarscout._numba_horizon.psr import _pixel_frame
from lunarscout.products import _load_dem

pytestmark = pytest.mark.skipif(
    os.environ.get("LUNARSCOUT_REQUIRE_NUMBA_CUDA") != "1",
    reason="set LUNARSCOUT_REQUIRE_NUMBA_CUDA=1 for the explicit real-GPU probe",
)

_WKT = 'PROJCS["Moon_South_Pole_Stereographic",GEOGCS["Moon",DATUM["Moon",SPHEROID["Moon",1737400,0]],PRIMEM["Reference_Meridian",0],UNIT["degree",0.0174532925199433]],PROJECTION["Polar_Stereographic"],PARAMETER["latitude_of_origin",-90],PARAMETER["central_meridian",0],PARAMETER["scale_factor",1],PARAMETER["false_easting",0],PARAMETER["false_northing",0],UNIT["metre",1]]'
_TIME_COUNT = 192
_START = datetime(2027, 1, 1, tzinfo=timezone.utc)
_STEP = timedelta(hours=6)


@pytest.fixture(scope="module")
def scene(tmp_path_factory: pytest.TempPathFactory):
    root = tmp_path_factory.mktemp("parity")
    rng = np.random.default_rng(20261010)
    x, y = np.meshgrid(np.arange(128) * 20.0, np.arange(128) * 20.0)
    elevation = 0.05 * x
    for _ in range(40):
        cx, cy = rng.uniform(0.0, 2560.0, 2)
        sigma = rng.uniform(60.0, 400.0)
        elevation += rng.uniform(-150.0, 250.0) * np.exp(
            -((x - cx) ** 2 + (y - cy) ** 2) / (2.0 * sigma**2)
        )
    georef = ls.GeoReference(
        projection_wkt=_WKT,
        projection_proj4="+proj=stere +lat_0=-90 +lon_0=0 +k=1 +R=1737400 +units=m",
        affine_transform=(100_000.0, 20.0, 0.0, -100_000.0, 0.0, -20.0),
        width=128,
        height=128,
        pixel_size_x=20.0,
        pixel_size_y=-20.0,
        nodata=None,
    )
    dem_path = ls.write_geotiff(root / "dem.tif", elevation.astype(np.float32), georef)
    horizons = ls.generate_horizons(root / "horizons", [dem_path])

    rotation, translation = _pixel_frame(_load_dem(dem_path)[0], 64, 64)
    step = np.arange(_TIME_COUNT)

    def vectors(distance_m: float, low_deg: float, high_deg: float) -> np.ndarray:
        azimuth = np.deg2rad(step * 3.0 * 360.0 / _TIME_COUNT)
        elevation_rad = np.deg2rad(
            low_deg + (high_deg - low_deg) * (0.5 + 0.5 * np.sin(step * 0.37))
        )
        local = np.column_stack((
            np.cos(elevation_rad) * np.sin(azimuth),
            np.cos(elevation_rad) * np.cos(azimuth),
            np.sin(elevation_rad),
        ))
        return np.ascontiguousarray((local * distance_m - translation) @ rotation.T)

    return {
        "root": root,
        "dem": dem_path,
        "horizons": horizons,
        "times": tuple(_START + _STEP * index for index in range(_TIME_COUNT)),
        "sun": vectors(1.5e11, -1.0, 6.0),
        "earth": vectors(3.84e8, -2.0, 8.0),
    }


def _both(scene, name: str, function, **arguments) -> tuple[np.ndarray, np.ndarray]:
    results = []
    for backend in ("cpu", "cuda"):
        output = function(
            scene["dem"],
            scene["horizons"],
            scene["root"] / f"{name}-{backend}.tif",
            backend=backend,
            **arguments,
        )
        with rasterio.open(output) as dataset:
            assert dataset.tags()["LUNARSCOUT_COMPUTE_BACKENDS"] == f'["{backend}"]'
            results.append(dataset.read().astype(np.float64))
    return results[0], results[1]


def _differing_fraction(cpu: np.ndarray, cuda: np.ndarray) -> float:
    same = (cpu == cuda) | (np.isnan(cpu) & np.isnan(cuda))
    return float(np.mean(~same))


def test_lightmap_differs_by_at_most_one_count_rarely(scene) -> None:
    cpu, cuda = _both(
        scene, "lightmap", ls.generate_lightmap,
        times=scene["times"], sun_vectors_m=scene["sun"],
    )

    assert 0.0 < np.mean((cpu > 0) & (cpu < 255)) < 1.0  # partial sunlight occurs
    assert np.max(np.abs(cpu - cuda)) <= 1
    assert _differing_fraction(cpu, cuda) < 1e-4


def test_psr_agrees_apart_from_rare_threshold_flips(scene) -> None:
    cpu, cuda = _both(
        scene, "psr", ls.generate_psr,
        times=scene["times"], sun_vectors_m=scene["sun"],
    )

    assert len(np.unique(cpu)) > 1
    assert _differing_fraction(cpu, cuda) < 1e-3


@pytest.mark.parametrize(
    ("function", "vectors"),
    [
        (ls.generate_sun_elevation, "sun_vectors_m"),
        (ls.generate_earth_elevation, "earth_vectors_m"),
    ],
)
def test_elevations_agree_away_from_steep_edge_horizons(
    scene, function, vectors: str
) -> None:
    key = "sun" if vectors == "sun_vectors_m" else "earth"
    cpu, cuda = _both(
        scene, function.__name__, function,
        times=scene["times"], **{vectors: scene[key]},
    )

    difference = np.abs(cpu - cuda)
    assert np.max(difference[:, 1:-1, 1:-1]) < 1e-3
    assert np.max(difference) < 0.01


def test_safe_havens_agree_apart_from_rare_threshold_flips(scene) -> None:
    cpu, cuda = _both(
        scene, "safe_havens", ls.generate_safe_havens,
        times=scene["times"], sun_vectors_m=scene["sun"],
        earth_vectors_m=scene["earth"], earth_elevation_threshold_deg=0.0,
        sunlight_fraction_threshold=0.5,
    )

    assert np.nanmax(cpu) > 0
    assert _differing_fraction(cpu, cuda) < 1e-3


@pytest.mark.parametrize(
    ("function", "thresholds"),
    [
        (ls.mission_duration_from_sunlight, {"sunlight_fraction_threshold": 0.5}),
        (ls.mission_duration_from_sun_elevation, {"sun_elevation_threshold_deg": 0.5}),
        (
            ls.mission_duration_from_sunlight_and_earth_elevation,
            {"sunlight_fraction_threshold": 0.5, "earth_elevation_threshold_deg": 0.0},
        ),
        (
            ls.mission_duration_from_sun_and_earth_elevation,
            {"sun_elevation_threshold_deg": 0.5, "earth_elevation_threshold_deg": 0.0},
        ),
    ],
)
def test_mission_durations_agree_apart_from_rare_threshold_flips(
    scene, function, thresholds: dict
) -> None:
    if "earth_elevation_threshold_deg" in thresholds:
        thresholds = thresholds | {"earth_vectors_m": scene["earth"]}
    cpu, cuda = _both(
        scene, function.__name__, function,
        evaluation_start=scene["times"][0],
        evaluation_stop=scene["times"][-1],
        step=_STEP,
        candidate_start_intervals=((scene["times"][0], scene["times"][40]),),
        sun_vectors_m=scene["sun"],
        **thresholds,
    )

    assert np.nanmax(cpu) > 0
    assert _differing_fraction(cpu, cuda) < 1e-3
