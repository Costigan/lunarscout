"""Public ``generate_safe_haven_products`` behaviour (CPU backend)."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import textwrap

import numpy as np
import pytest
import rasterio

import lunarscout as ls
from lunarscout._numba_horizon.file_format import AZIMUTH_COUNT, HorizonTileStore
from lunarscout._numba_horizon.geometry import DemGrid, ProjectionParameters
from lunarscout._numba_horizon.pipeline import PatchDescriptor
from lunarscout._numba_horizon.psr import _pixel_frame
from lunarscout._numba_horizon.safe_haven_pipeline import patch_slope_exceeds
from lunarscout.terrain import slope as terrain_slope

_WKT = 'PROJCS["Moon_South_Pole_Stereographic",GEOGCS["Moon",DATUM["Moon",SPHEROID["Moon",1737400,0]],PRIMEM["Reference_Meridian",0],UNIT["degree",0.0174532925199433]],PROJECTION["Polar_Stereographic"],PARAMETER["latitude_of_origin",-90],PARAMETER["central_meridian",0],PARAMETER["scale_factor",1],PARAMETER["false_easting",0],PARAMETER["false_northing",0],UNIT["metre",1]]'
_PROJ4 = "+proj=stere +lat_0=-90 +lon_0=0 +k=1 +R=1737400 +units=m"


def _georef(width: int, height: int):
    return ls.GeoReference(
        projection_wkt=_WKT,
        projection_proj4=_PROJ4,
        affine_transform=(1000.0, 1.0, 0.0, -1000.0, 0.0, -1.0),
        width=width,
        height=height,
        pixel_size_x=1.0,
        pixel_size_y=-1.0,
        nodata=None,
    )


def _inputs(tmp_path: Path, elevation: np.ndarray) -> tuple[Path, Path]:
    """A DEM and zero-elevation horizons for every pixel of ``elevation``."""
    height, width = elevation.shape
    dem_path = tmp_path / "dem.tif"
    ls.write_geotiff(dem_path, elevation.astype(np.float32), _georef(width, height))
    horizons_path = tmp_path / "horizons"
    HorizonTileStore(horizons_path).write(
        0, 0, 0.0, np.zeros((width * height, AZIMUTH_COUNT), dtype=np.float32),
        compress=True, valid_width=width, valid_height=height,
    )
    return dem_path, horizons_path


def _body(elevation_deg: float) -> np.ndarray:
    """A Moon-ME position seen at ``elevation_deg`` due north of pixel (0, 0)."""
    dem = DemGrid(
        np.zeros((1, 1), dtype=np.float32),
        np.asarray((1000.0, 1.0, 0.0, -1000.0, 0.0, -1.0)),
        ProjectionParameters(1_737_400.0, -np.pi / 2.0, 0.0, 1.0, 0.0, 0.0),
    )
    rotation, translation = _pixel_frame(dem, 0, 0)
    angle = np.deg2rad(elevation_deg)
    local = np.asarray((0.0, np.cos(angle), np.sin(angle)))
    return (local * 150_000_000_000.0 - translation) @ rotation.T


SHADOW, LIT = -1.0, 5.0        # Sun elevation: below the horizon, or fully lit
BELOW, ABOVE = -1.0, 10.0      # Earth elevation: outage, or available

# 12-hour samples from January 30 to February 2; each letter is one sample.
_TIMES = tuple(
    f"2027-{month:02d}-{day:02d}T{hour:02d}:00:00Z"
    for month, day, hour in (
        (1, 30, 0), (1, 30, 12), (1, 31, 0), (1, 31, 12),
        (2, 1, 0), (2, 1, 12), (2, 2, 0), (2, 2, 12),
    )
)
_SUN = [LIT] * 4 + [SHADOW] * 4                     # shadow from February 1 00:00
_EARTH = [ABOVE] * 3 + [BELOW] * 3 + [ABOVE] * 2    # outage January 31 12:00 - February 1 12:00


def _vectors(elevations):
    return np.stack([_body(value) for value in elevations])


def _products(tmp_path, dem_path, horizons_path, name="out", **kwargs):
    options = dict(
        times=_TIMES, sun_vectors_m=_vectors(_SUN), earth_vectors_m=_vectors(_EARTH),
        backend="cpu",
    )
    options.update(kwargs)
    return ls.generate_safe_haven_products(dem_path, horizons_path, tmp_path / name, **options)


def _values(products):
    out = []
    for path in products:
        with rasterio.open(path) as dataset:
            out.append(dataset.read(1 if dataset.count == 1 else None))
    return out


def test_a_hibernation_is_scored_in_the_month_it_starts(tmp_path: Path) -> None:
    dem_path, horizons_path = _inputs(tmp_path, np.zeros((1, 1)))

    products = _products(tmp_path, dem_path, horizons_path)

    hibernation, joint_total, joint_longest = _values(products)
    # The outage starts January 31 12:00 (January). The shadow run starts February 1
    # and lasts four 12-hour samples; its whole 48 hours is scored in January, and
    # February, where no hibernation starts, is NODATA.
    assert hibernation.shape == (2, 1, 1)
    assert hibernation[0, 0, 0] == pytest.approx(48.0)
    assert np.isnan(hibernation[1, 0, 0])
    # Earth is available and the Sun full on the first three samples, all in January.
    assert joint_total[:, 0, 0].tolist() == [36.0, 0.0]
    assert joint_longest[:, 0, 0].tolist() == [36.0, 0.0]
    with rasterio.open(products.hibernation) as dataset:
        assert dataset.tags(1)["TIMESTAMP_UTC"].startswith("2027-01-01")


@pytest.mark.parametrize(
    ("max_slope_deg", "masked"), [(20.0, True), (30.0, False), (None, False)]
)
def test_steep_pixels_are_nodata_in_all_three_products(
    tmp_path: Path, max_slope_deg, masked: bool
) -> None:
    # A 0.5 m/pixel ramp on 1 m pixels is a 26.6 degree slope everywhere.
    ramp = np.tile(0.5 * np.arange(5.0), (5, 1))
    dem_path, horizons_path = _inputs(tmp_path, ramp)

    products = _products(tmp_path, dem_path, horizons_path, max_slope_deg=max_slope_deg)

    hibernation, joint_total, joint_longest = _values(products)
    for values in (hibernation, joint_total, joint_longest):
        # The four DEM corners follow gdaldem's -compute_edges rule, as lunarscout.slope
        # does (checked against it below), so look at the interior.
        assert bool(np.isnan(values[:, 1:4, 1:4]).all()) is masked
    if not masked:
        assert (hibernation[0] == 48.0).all() and np.isnan(hibernation[1]).all()
        assert (joint_total[0] == 36.0).all() and (joint_longest[0] == 36.0).all()


def test_the_patch_slope_mask_matches_the_public_slope_across_patch_edges() -> None:
    rng = np.random.default_rng(3)
    elevation = np.cumsum(np.cumsum(rng.normal(0.0, 0.4, (70, 131)), axis=0), axis=1)
    georef = _georef(131, 70)
    dem = DemGrid(
        elevation.astype(np.float32),
        np.asarray(georef.affine_transform),
        ProjectionParameters(1_737_400.0, -np.pi / 2.0, 0.0, 1.0, 0.0, 0.0),
    )
    reference, _ = terrain_slope(
        elevation.astype(np.float32), georef, output_nodata=np.nan, compute_edges=True
    )
    limit = float(np.nanmedian(reference))
    assert 0.0 < (reference > limit).mean() < 1.0

    for patch in (
        PatchDescriptor(0, 0, 0, 0, 0, 128, 70),
        PatchDescriptor(1, 128, 0, 1, 0, 3, 70),
    ):
        expected = ~(
            reference[:, patch.tile_x : patch.tile_x + patch.width] <= limit
        )
        np.testing.assert_array_equal(patch_slope_exceeds(dem, georef, patch, limit), expected)


def test_invalid_dem_neighbours_are_masked_like_steep_pixels(tmp_path: Path) -> None:
    elevation = np.zeros((5, 5))
    elevation[2, 2] = np.nan
    dem_path, horizons_path = _inputs(tmp_path, elevation)

    products = _products(tmp_path, dem_path, horizons_path)

    hibernation = _values(products)[0][0]
    assert np.isnan(hibernation[1:4, 1:4]).all()           # the node and its 8 neighbours
    assert np.isfinite(hibernation[0, 0])                   # flat and valid


@pytest.mark.parametrize(
    ("argument", "value", "code"),
    [
        ("merge_gap_hours", -1.0, "safe_haven_merge_gap_invalid"),
        ("merge_gap_hours", float("nan"), "safe_haven_merge_gap_invalid"),
        ("operating_sunlight_fraction", 1.5, "safe_haven_operating_fraction_invalid"),
        ("max_slope_deg", 0.0, "safe_haven_max_slope_invalid"),
        ("max_slope_deg", float("inf"), "safe_haven_max_slope_invalid"),
        ("name_prefix", "a/b", "safe_haven_name_prefix_invalid"),
        ("name_prefix", "", "safe_haven_name_prefix_invalid"),
    ],
)
def test_invalid_arguments_are_rejected_before_any_output(
    tmp_path: Path, argument: str, value, code: str
) -> None:
    dem_path, horizons_path = _inputs(tmp_path, np.zeros((1, 1)))

    with pytest.raises(ls.InputError) as raised:
        _products(tmp_path, dem_path, horizons_path, **{argument: value})

    assert raised.value.code == code
    assert not (tmp_path / "out").exists()


def test_name_prefix_names_the_three_files(tmp_path: Path) -> None:
    dem_path, horizons_path = _inputs(tmp_path, np.zeros((1, 1)))

    products = _products(tmp_path, dem_path, horizons_path, name_prefix="pole")

    assert [path.name for path in products] == [
        "pole_hibernation_hours.tif",
        "pole_joint_operating_total_hours.tif",
        "pole_joint_operating_longest_hours.tif",
    ]


def test_an_interrupted_run_resumes_and_matches_an_uninterrupted_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from lunarscout._numba_horizon.product_store import ResumableTiledProduct

    dem_path, horizons_path = _inputs(tmp_path, np.zeros((1, 1)))
    clean = _products(tmp_path, dem_path, horizons_path, name="clean")

    # The second product's write fails once, after the first product's patch is durable.
    original = ResumableTiledProduct.write_patch
    state = {"calls": 0}

    def flaky(self, *args, **kwargs):
        state["calls"] += 1
        if state["calls"] == 2:
            raise OSError("simulated disk failure")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(ResumableTiledProduct, "write_patch", flaky)
    with pytest.raises(ls.ProductStorageError):
        _products(tmp_path, dem_path, horizons_path, name="resumed")
    monkeypatch.setattr(ResumableTiledProduct, "write_patch", original)

    resumed = _products(tmp_path, dem_path, horizons_path, name="resumed")

    for left, right in zip(_values(clean), _values(resumed)):
        np.testing.assert_array_equal(left, right)


def test_the_public_function_in_a_fresh_cpu_process(tmp_path: Path) -> None:
    dem_path, horizons_path = _inputs(tmp_path, np.zeros((1, 1)))
    np.save(tmp_path / "sun.npy", _vectors(_SUN))
    np.save(tmp_path / "earth.npy", _vectors(_EARTH))
    code = textwrap.dedent(f"""
        import json, sys
        import numpy as np
        import rasterio
        import lunarscout as ls

        products = ls.generate_safe_haven_products(
            {str(dem_path)!r}, {str(horizons_path)!r}, {str(tmp_path / "fresh")!r},
            times={list(_TIMES)!r},
            sun_vectors_m=np.load({str(tmp_path / "sun.npy")!r}),
            earth_vectors_m=np.load({str(tmp_path / "earth.npy")!r}),
            backend="cpu",
        )
        values = []
        for path in products:
            with rasterio.open(path) as dataset:
                values.append(dataset.read()[:, 0, 0].tolist())
        print(json.dumps({{
            "values": values,
            "spice": "spiceypy" in sys.modules,
            "cuda": "numba.cuda" in sys.modules,
        }}))
    """)
    completed = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True,
        env={**__import__("os").environ, "PYTHONPATH": str(Path(ls.__file__).parents[1])},
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    result = json.loads(completed.stdout.strip().splitlines()[-1])
    # Explicit vectors and the CPU backend must not load SPICE or CUDA.
    assert result["spice"] is False and result["cuda"] is False
    hibernation, total, longest = result["values"]
    assert hibernation[0] == pytest.approx(48.0) and hibernation[1] != hibernation[1]
    assert total == [36.0, 0.0] and longest == [36.0, 0.0]
