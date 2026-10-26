"""Stage 1 fed from the stage-0 light store: it must give the products the horizon route gives."""

from __future__ import annotations

import os
import shutil

import numpy as np
import pytest
import rasterio

import blocks_scene as bs
import lunarscout as ls
import windows_scene as scene
from lunarscout._numba_horizon import safe_haven_stage1_store as from_store
from lunarscout._numba_horizon import safe_haven_stage0 as stage0
from lunarscout._numba_horizon.file_format import HorizonTileStore
from lunarscout._numba_horizon.lightmap_cpu import LightmapCpuSession
from lunarscout._numba_horizon.safe_haven_lightstore import LightStoreSettings
from lunarscout._numba_horizon.safe_haven_stage0 import LightStoreDirectory, Stage0Settings
from lunarscout.errors import InputError, OperationCancelledError
from lunarscout.products import _load_dem

START, LAST = "2027-01-01T00:00:00Z", "2027-03-31T22:00:00Z"        # 1,080 steps of two hours
STORE_START, STORE_STOP = "2027-01-01T00:00:00", "2027-04-01T00:00:00"


def make_store(directory, dem_path, horizons_path):
    directory.mkdir()
    dem, _georef = _load_dem(dem_path)
    settings = Stage0Settings(
        store=LightStoreSettings(steps_per_chunk=48, codec="zlib-1"), start_utc=STORE_START, stop_utc=STORE_STOP,
        lighting="cpu", workers=2)
    stage0.generate_light_store(
        directory, dem=dem, horizon_store=HorizonTileStore(horizons_path), settings=settings,
        sun_vectors_at=scene.sun_vectors, earth_vectors_at=scene.earth_vectors)
    return directory


def products(tmp_path, name, dem_path, *, horizons_path=None, light_store_path=None, **options):
    times = ls.times(START, LAST, step_hours=2.0)
    stamps = times.values
    options.setdefault("backend", "cpu")
    return ls.generate_safe_haven_products(
        dem_path, horizons_path, tmp_path / name, times=times, light_store_path=light_store_path,
        sun_vectors_m=scene.sun_vectors(stamps), earth_vectors_m=scene.earth_vectors(stamps), **options)


def rasters(paths):
    out = []
    for path in paths:
        with rasterio.open(path) as dataset:
            out.append(dataset.read())
    return out


@pytest.fixture(scope="module")
def blocks(tmp_path_factory):
    directory = tmp_path_factory.mktemp("blocks")
    dem_path, horizons_path = bs.write_inputs(directory)
    return dem_path, horizons_path, make_store(directory / "store", dem_path, horizons_path)


@pytest.fixture(scope="module")
def earthside(tmp_path_factory):
    """One patch with a corner whose Earth never sets: the degenerate-outage rule acts there."""
    directory = tmp_path_factory.mktemp("earthside")
    dem_path, horizons_path = scene.write_inputs(directory, earthside=True)
    return dem_path, horizons_path, make_store(directory / "store", dem_path, horizons_path)


@pytest.mark.parametrize("window_hours", [0.0, None])
def test_the_store_route_gives_the_horizon_route_products(blocks, tmp_path, window_hours) -> None:
    dem_path, horizons_path, store = blocks
    reference = products(tmp_path, "horizons", dem_path, horizons_path=horizons_path,
                         degenerate_outage_window_hours=window_hours)
    stored = products(tmp_path, "store", dem_path, light_store_path=store,
                      degenerate_outage_window_hours=window_hours)
    for kind, (want, got) in zip(("hibernation", "joint_total", "joint_longest"), zip(rasters(reference), rasters(stored))):
        assert want.shape[0] == 3, "three monthly bands"
        np.testing.assert_array_equal(got, want, err_msg=kind)
    hibernation, joint_total, _ = rasters(stored)
    assert np.isfinite(hibernation).any() and (joint_total > 0).any()           # not trivially equal


def test_the_degenerate_outage_rule_agrees_where_it_acts(earthside, tmp_path) -> None:
    dem_path, horizons_path, store = earthside
    window = 24.0
    reference = products(tmp_path, "horizons", dem_path, horizons_path=horizons_path, degenerate_outage_window_hours=window)
    without = products(tmp_path, "without", dem_path, horizons_path=horizons_path, degenerate_outage_window_hours=0.0)
    stored = products(tmp_path, "store", dem_path, light_store_path=store, degenerate_outage_window_hours=window)
    want, off = rasters(reference)[0], rasters(without)[0]
    assert not np.array_equal(want, off), "the rule changes this scene's result, so the test can see it"
    for expected, got in zip(rasters(reference), rasters(stored)):
        np.testing.assert_array_equal(got, expected)


def test_a_patch_without_a_file_is_nodata_and_the_rest_is_unchanged(blocks, tmp_path) -> None:
    dem_path, horizons_path, store = blocks
    copy = tmp_path / "store"
    shutil.copytree(store, copy)
    stage0.patch_file(copy, 1, 1).unlink()
    reference = rasters(products(tmp_path, "horizons", dem_path, horizons_path=horizons_path, nodata=255.0))
    stored = rasters(products(tmp_path, "store", dem_path, light_store_path=copy, nodata=255.0))
    for want, got in zip(reference, stored):
        assert np.all(got[:, 128:256, 128:256] == 255.0)
        got = got.copy()
        got[:, 128:256, 128:256] = want[:, 128:256, 128:256]
        np.testing.assert_array_equal(got, want)


@pytest.mark.parametrize("options, code", [
    (dict(sunlight_fraction_threshold=0.3), "safe_haven_store_thresholds"),
    (dict(operating_sunlight_fraction=0.9), "safe_haven_store_thresholds"),
    (dict(earth_elevation_threshold_deg=3.0), "safe_haven_store_thresholds"),
])
def test_other_thresholds_than_the_stores_are_refused(blocks, tmp_path, options, code) -> None:
    dem_path, _horizons, store = blocks
    with pytest.raises(InputError) as caught:
        products(tmp_path, "out", dem_path, light_store_path=store, **options)
    assert caught.value.code == code
    assert not (tmp_path / "out").exists() or not list((tmp_path / "out").glob("*.tif"))


def test_the_time_grid_must_be_the_stores(blocks, tmp_path) -> None:
    dem_path, _horizons, store = blocks
    for times, code in (
        (ls.times(START, LAST, step_hours=6.0), "safe_haven_store_step_mismatch"),
        (ls.times("2027-01-01T01:00:00Z", LAST, step_hours=2.0), "safe_haven_store_time_misaligned"),
        (ls.times("2026-12-31T00:00:00Z", LAST, step_hours=2.0), "safe_haven_store_time_range"),
        (ls.times(START, "2027-04-02T00:00:00Z", step_hours=2.0), "safe_haven_store_time_range"),
    ):
        stamps = times.values
        with pytest.raises(InputError) as caught:
            ls.generate_safe_haven_products(
                dem_path, None, tmp_path / "out", times=times, light_store_path=store, backend="cpu",
                sun_vectors_m=scene.sun_vectors(stamps), earth_vectors_m=scene.earth_vectors(stamps))
        assert caught.value.code == code


def test_a_store_for_another_dem_is_refused(blocks, earthside, tmp_path) -> None:
    dem_path, _horizons, _store = blocks
    _dem2, _horizons2, other_store = earthside
    with pytest.raises(Exception) as caught:
        products(tmp_path, "out", dem_path, light_store_path=other_store)
    assert getattr(caught.value, "code", "") == "light_store_manifest_mismatch"


def test_exactly_one_input_is_required(blocks, tmp_path) -> None:
    dem_path, horizons_path, store = blocks
    with pytest.raises(InputError) as both:
        products(tmp_path, "a", dem_path, horizons_path=horizons_path, light_store_path=store)
    with pytest.raises(InputError) as neither:
        products(tmp_path, "b", dem_path)
    assert (both.value.code, neither.value.code) == ("safe_haven_input_ambiguous", "safe_haven_input_ambiguous")


def test_a_cancellation_stops_the_store_route(blocks, tmp_path) -> None:
    dem_path, _horizons, store = blocks
    with pytest.raises(OperationCancelledError):
        products(tmp_path, "out", dem_path, light_store_path=store, cancellation_requested=lambda: True)


def test_the_store_route_reads_no_horizon(blocks, tmp_path, monkeypatch) -> None:
    dem_path, _horizons, store = blocks
    import lunarscout._numba_horizon.safe_haven_pipeline as pipeline

    monkeypatch.setattr(pipeline, "read_patch_horizons", lambda *a, **k: pytest.fail("a horizon was read"))
    monkeypatch.setattr(HorizonTileStore, "inventory_tiles", lambda *a, **k: pytest.fail("horizons were inventoried"))
    products(tmp_path, "out", dem_path, light_store_path=store)


def test_flat_elevation_is_the_kernels_value_at_a_valid_pixel_near_the_centre(blocks) -> None:
    dem_path, _horizons, _store = blocks
    dem, _ = _load_dem(dem_path)
    stamps = np.datetime64("2027-01-01T00:00:00", "us") + np.arange(50) * np.timedelta64(2, "h")
    vectors = scene.earth_vectors(stamps)
    session = LightmapCpuSession()
    series = from_store.flat_earth_elevation(dem, vectors, 128, 128, 0, 0, session)
    assert series.shape == (50,) and series.dtype == np.float32 and np.isfinite(series).all()
    centre = session.pixel_series(dem, np.zeros(1440, dtype=np.float32), vectors, row=64, column=64,
                                  margin=True, flat_elevation=True)
    np.testing.assert_allclose(series, centre, atol=1e-3)
    # A patch with no valid pixel has nothing to compute; the reducer skips every pixel anyway.
    dem.elevation_m[0:128, 0:128] = np.nan
    assert not from_store.flat_earth_elevation(dem, vectors, 128, 128, 0, 0, session).any()


def test_the_identity_names_the_stores_contents(blocks, earthside) -> None:
    dem_path, _h, store = blocks
    dem, _ = _load_dem(dem_path)
    first = from_store.store_identity(LightStoreDirectory(store, dem))
    assert first == from_store.store_identity(LightStoreDirectory(store, dem))
    assert first != from_store.store_identity(LightStoreDirectory(earthside[2]))


# --- the device route (real GPU: set LUNARSCOUT_REQUIRE_NUMBA_CUDA=1) ---------------------------------

gpu = pytest.mark.skipif(
    os.environ.get("LUNARSCOUT_REQUIRE_NUMBA_CUDA") != "1",
    reason="set LUNARSCOUT_REQUIRE_NUMBA_CUDA=1 for the real-GPU stage-1 tests",
)


@gpu
@pytest.mark.parametrize("window_hours", [0.0, 24.0])
def test_the_device_route_gives_the_cpu_route_products(blocks, earthside, tmp_path, window_hours) -> None:
    for name, (dem_path, _horizons, store) in (("blocks", blocks), ("earthside", earthside)):
        cpu = products(tmp_path, f"{name}-cpu", dem_path, light_store_path=store, degenerate_outage_window_hours=window_hours)
        cuda = products(tmp_path, f"{name}-cuda", dem_path, light_store_path=store, backend="cuda",
                        degenerate_outage_window_hours=window_hours)
        for want, got in zip(rasters(cpu), rasters(cuda)):
            np.testing.assert_array_equal(got, want, err_msg=name)


@gpu
def test_the_device_route_refuses_what_the_cpu_route_refuses_and_can_be_cancelled(blocks, tmp_path) -> None:
    dem_path, _horizons, store = blocks
    with pytest.raises(InputError) as caught:
        products(tmp_path, "thresholds", dem_path, light_store_path=store, backend="cuda",
                 sunlight_fraction_threshold=0.3)
    assert caught.value.code == "safe_haven_store_thresholds"
    with pytest.raises(OperationCancelledError):
        products(tmp_path, "cancelled", dem_path, light_store_path=store, backend="cuda",
                 cancellation_requested=lambda: True)
