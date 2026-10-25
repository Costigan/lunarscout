"""Stage 0: the thresholded-lighting store (file format, codecs, producer)."""

from __future__ import annotations

import sys

import numpy as np
import pytest

import windows_scene as scene
from lunarscout._numba_horizon import safe_haven_lightstore as store
from lunarscout._numba_horizon.file_format import PATCH_SIZE, HorizonTileStore, read_patch_horizons
from lunarscout._numba_horizon.lightmap_cpu import LightmapCpuSession
from lunarscout._numba_horizon.safe_haven_lightstore import LightStore, LightStoreSettings, LightStoreWriter
from lunarscout._numba_horizon.safe_haven_windows import PixelSignals
from lunarscout.errors import InputError, ProductStorageError
from lunarscout.products import _load_dem

CODECS = ["none", "zlib-1", "zstd-1", "zstd-3", "zstd-9", "lz4", "lz4-9", "snappy"]


def structured_planes(steps: int, seed: int = 3) -> np.ndarray:
    """Planes with long runs in time and space, so that they compress well but not trivially."""
    rng = np.random.default_rng(seed)
    rows = np.arange(PATCH_SIZE)[None, :, None]
    cols = np.arange(PATCH_SIZE)[None, None, :]
    time = np.arange(steps)[:, None, None]
    planes = []
    for plane in range(store.PLANE_COUNT):
        edge = (time * (plane + 1) * 0.37 + rows * 0.5 + cols * 0.25 + rng.integers(0, 5)) % 64
        lit = edge > (20 + 10 * plane)
        flips = rng.random((steps, PATCH_SIZE, PATCH_SIZE)) < 0.002
        planes.append(np.logical_xor(lit, flips))
    bits = np.stack(planes)                                   # (3, steps, 128, 128) bool
    words = np.zeros((store.PLANE_COUNT, steps, PATCH_SIZE, 2), dtype=np.uint64)
    for column in range(PATCH_SIZE):
        words[..., column // 64] |= bits[..., column].astype(np.uint64) << np.uint64(column % 64)
    return words


def settings_for(codec: str, **kwargs) -> LightStoreSettings:
    return LightStoreSettings(codec=codec, steps_per_chunk=kwargs.pop("steps_per_chunk", 96), **kwargs)


# --- encoding ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("codec", CODECS)
def test_every_codec_round_trips_across_chunk_edges(tmp_path, codec) -> None:
    try:
        settings = settings_for(codec)
    except ProductStorageError as error:                       # lz4 and snappy are optional
        pytest.skip(str(error))
    planes = structured_planes(250)                           # chunks of 96, 96 and 58 steps
    path = tmp_path / "patch.lit"
    size = store.write_patch(path, planes, settings, patch_row=4, patch_column=9, first_step=96 * 7)
    assert size == path.stat().st_size
    first = 96 * 7
    with LightStore(path, settings) as opened:
        assert (opened.patch, opened.first_step, opened.step_count, opened.chunk_count) == ((4, 9), first, 250, 3)
        assert np.array_equal(opened.read_steps(first, first + 250), planes)
        for lo, hi in ((0, 1), (95, 97), (0, 96), (96, 192), (190, 250), (249, 250), (50, 200), (94, 242)):
            assert np.array_equal(opened.read_steps(first + lo, first + hi), planes[:, lo:hi]), (lo, hi)
        assert opened.stored_bytes(first, first + 96) < opened.stored_bytes()      # a day-range reads fewer bytes
        assert opened.stored_bytes() == size - int(opened._offsets[0])


def test_a_real_codec_makes_the_file_much_smaller(tmp_path) -> None:
    planes = structured_planes(192)
    raw = store.write_patch(tmp_path / "a", planes, settings_for("none"), patch_row=0, patch_column=0, first_step=0)
    packed = store.write_patch(tmp_path / "b", planes, settings_for("zstd-3"), patch_row=0, patch_column=0, first_step=0)
    assert packed < raw / 4


def test_a_read_decodes_only_the_chunks_it_covers(tmp_path, monkeypatch) -> None:
    settings = settings_for("zlib-1")
    path = tmp_path / "patch.lit"
    store.write_patch(path, structured_planes(96 * 5), settings, patch_row=0, patch_column=0, first_step=0)
    decoded = []
    real = store.decode_chunk
    monkeypatch.setattr(store, "decode_chunk", lambda data, steps, s: decoded.append(steps) or real(data, steps, s))
    with LightStore(path) as opened:
        opened.read_steps(96 * 2 + 5, 96 * 3 + 5)
    assert len(decoded) == 2                                  # chunks 2 and 3 of 5


# --- refusals ---------------------------------------------------------------------------------------


@pytest.fixture
def written(tmp_path):
    settings = settings_for("zlib-1")
    path = tmp_path / "patch.lit"
    store.write_patch(path, structured_planes(100), settings, patch_row=1, patch_column=2, first_step=0)
    return path, settings


def test_settings_mismatch_is_refused_with_both_settings(written) -> None:
    path, settings = written
    other = settings_for("zlib-1", earth_elevation_deg=3.0)
    with pytest.raises(ProductStorageError) as error:
        LightStore(path, other)
    assert error.value.code == "light_store_settings_mismatch"
    assert error.value.details["file"]["earth_elevation_deg"] == 2.0
    assert error.value.details["expected"]["earth_elevation_deg"] == 3.0
    LightStore(path, settings).close()                        # the right ones open it
    LightStore(path).close()                                  # none given: no check


@pytest.mark.parametrize("damage", ["magic", "truncate_data", "truncate_header", "truncate_table", "empty", "offsets"])
def test_damaged_files_are_refused_not_misread(written, damage) -> None:
    path, _ = written
    data = bytearray(path.read_bytes())
    if damage == "magic":
        data[0] ^= 0xFF
    elif damage == "truncate_data":
        data = data[:-5]
    elif damage == "truncate_header":
        data = data[:30]
    elif damage == "truncate_table":
        data = data[:300]
    elif damage == "empty":
        data = b""
    elif damage == "offsets":
        header = int(np.frombuffer(bytes(data[8:16]), dtype="<u8")[0])
        table = 16 + header + (-header % 8)
        data[table + 8:table + 16] = (10).to_bytes(8, "little")               # offset 1 before offset 0
    path.write_bytes(bytes(data))
    with pytest.raises(ProductStorageError) as error:
        LightStore(path)
    assert error.value.code == "light_store_corrupt"


def test_a_damaged_chunk_is_refused_when_read(written) -> None:
    path, _ = written
    data = bytearray(path.read_bytes())
    with LightStore(path) as opened:
        start = int(opened._offsets[0])
    data[start + 4: start + 24] = bytes(20)
    path.write_bytes(bytes(data))
    with LightStore(path) as opened:
        with pytest.raises(Exception) as error:
            opened.read_steps(0, 10)
    assert error.type.__name__ in ("error", "ProductStorageError")             # zlib raises its own error


def test_missing_file_and_out_of_range_reads(tmp_path, written) -> None:
    with pytest.raises(ProductStorageError) as error:
        LightStore(tmp_path / "absent.lit")
    assert error.value.code == "light_store_unreadable"
    path, _ = written
    with LightStore(path) as opened:
        for first, stop in ((-1, 5), (0, 101), (5, 5), (7, 3)):
            with pytest.raises(InputError) as bad:
                opened.read_steps(first, stop)
            assert bad.value.code == "light_store_range"


def test_unknown_and_missing_codecs(monkeypatch) -> None:
    with pytest.raises(InputError) as error:
        LightStoreSettings(codec="bogus")
    assert error.value.code == "light_store_codec_unknown"
    with pytest.raises(InputError):
        LightStoreSettings(codec="zstd-x")
    monkeypatch.setitem(sys.modules, "zstandard", None)
    with pytest.raises(ProductStorageError) as missing:
        LightStoreSettings(codec="zstd-3")
    assert missing.value.code == "light_store_codec_unavailable"


@pytest.mark.parametrize("bad", [
    dict(step_minutes=7), dict(step_minutes=0), dict(steps_per_chunk=0),
    dict(shadow_fraction=0.9, operating_fraction=0.5), dict(operating_fraction=1.5),
])
def test_invalid_settings(bad) -> None:
    with pytest.raises(InputError):
        LightStoreSettings(**bad)


# --- writing ----------------------------------------------------------------------------------------


def test_a_failed_write_leaves_no_file_and_keeps_the_old_one(tmp_path) -> None:
    settings = settings_for("zlib-1", steps_per_chunk=10)
    path = tmp_path / "patch.lit"
    first = structured_planes(30)
    store.write_patch(path, first, settings, patch_row=0, patch_column=0, first_step=0)
    before = path.read_bytes()
    with pytest.raises(RuntimeError):
        with LightStoreWriter(path, settings, patch_row=0, patch_column=0, first_step=0, step_count=30) as writer:
            writer.write_chunk(first[:, :10])
            raise RuntimeError("the lighting failed")
    assert path.read_bytes() == before
    assert sorted(entry.name for entry in tmp_path.iterdir()) == ["patch.lit"]            # no staging file left


def test_closing_early_and_writing_wrong_chunks_are_errors(tmp_path) -> None:
    settings = settings_for("none", steps_per_chunk=10)
    planes = structured_planes(25)
    path = tmp_path / "patch.lit"
    with pytest.raises(ProductStorageError) as early:
        with LightStoreWriter(path, settings, patch_row=0, patch_column=0, first_step=0, step_count=25) as writer:
            writer.write_chunk(planes[:, :10])
    assert early.value.code == "light_store_incomplete" and not path.exists()
    with pytest.raises(InputError) as wrong:
        with LightStoreWriter(path, settings, patch_row=0, patch_column=0, first_step=0, step_count=25) as writer:
            writer.write_chunk(planes[:, :9])
    assert wrong.value.code == "light_store_chunk_steps" and not path.exists()
    with pytest.raises(InputError) as many:
        with LightStoreWriter(path, settings, patch_row=0, patch_column=0, first_step=0, step_count=25) as writer:
            for start in (0, 10, 20):
                writer.write_chunk(planes[:, start:start + 10])
            writer.write_chunk(planes[:, :5])
    assert many.value.code == "light_store_too_many_chunks" and not path.exists()
    with pytest.raises(InputError) as shape:
        with LightStoreWriter(path, settings, patch_row=0, patch_column=0, first_step=0, step_count=25) as writer:
            writer.write_chunk(planes[0, :10])
    assert shape.value.code == "light_store_chunk_shape"


def test_a_file_may_start_at_any_step(tmp_path) -> None:
    settings = settings_for("zlib-1", steps_per_chunk=10)
    planes = structured_planes(25)
    store.write_patch(tmp_path / "a", planes, settings, patch_row=0, patch_column=0, first_step=237)
    with LightStore(tmp_path / "a") as opened:
        assert np.array_equal(opened.read_steps(237 + 8, 237 + 23), planes[:, 8:23])


def test_count_and_directory_are_checked_before_anything_is_written(tmp_path) -> None:
    settings = settings_for("none", steps_per_chunk=10)
    with pytest.raises(InputError):
        LightStoreWriter(tmp_path / "a", settings, patch_row=0, patch_column=0, first_step=0, step_count=0)
    with pytest.raises(ProductStorageError) as directory:
        LightStoreWriter(tmp_path / "no" / "a", settings, patch_row=0, patch_column=0, first_step=0, step_count=10)
    assert directory.value.code == "light_store_directory_missing"
    assert list(tmp_path.iterdir()) == []


# --- the producer against the per-pixel reference ---------------------------------------------------


@pytest.fixture(scope="module")
def lit_scene(tmp_path_factory):
    directory = tmp_path_factory.mktemp("lightstore_scene")
    dem_path, horizons_path = scene.write_inputs(directory, earthside=True, ridge=True)
    dem, _georef = _load_dem(dem_path)
    horizon_store = HorizonTileStore(horizons_path)
    horizons = read_patch_horizons(horizon_store, 0, 0, 0.0)
    stamps = scene.EPOCH + np.arange(0, 150) * np.timedelta64(2, "h") + np.timedelta64(7 * 24, "h")
    return dict(dem=dem, horizons=horizons, signals=PixelSignals(dem, horizon_store), stamps=stamps,
                sun=scene.sun_vectors(stamps), earth=scene.earth_vectors_ridge(stamps))


def test_planes_equal_the_per_pixel_thresholds_and_the_edges_are_clear(lit_scene) -> None:
    settings = LightStoreSettings(shadow_fraction=0.2, operating_fraction=1.0, earth_elevation_deg=2.0)
    planes = store.light_patch_planes(
        LightmapCpuSession(time_batch_size=32), lit_scene["dem"], lit_scene["horizons"], 0, 0,
        lit_scene["sun"], lit_scene["earth"], settings,
    )
    assert planes.shape == (3, 150, PATCH_SIZE, 2) and planes.dtype == np.uint64
    bits = np.zeros((3, 150, PATCH_SIZE, PATCH_SIZE), dtype=bool)
    for column in range(PATCH_SIZE):
        bits[..., column] = (planes[..., column // 64] >> np.uint64(column % 64)) & np.uint64(1)
    size = scene.SIZE
    for row in range(size):
        for column in range(size):
            fraction = lit_scene["signals"].series(row, column, lit_scene["sun"], margin=False).astype(np.float64)
            margin = lit_scene["signals"].series(row, column, lit_scene["earth"], margin=True).astype(np.float64)
            assert np.array_equal(bits[0, :, row, column], fraction >= 0.2), (row, column)
            assert np.array_equal(bits[1, :, row, column], fraction >= 1.0 - 1e-6), (row, column)
            assert np.array_equal(bits[2, :, row, column], margin >= 2.0), (row, column)
    assert not bits[:, :, size:, :].any() and not bits[:, :, :, size:].any()           # outside the DEM
    assert bits[0].any() and not bits[0].all()                                          # the scene exercises each plane
    assert (bits[0] != bits[1]).any() and not (bits[1] & ~bits[0]).any()                # full implies partial
    assert bits[2].any() and not bits[2].all()


def test_a_produced_patch_survives_the_file(tmp_path, lit_scene) -> None:
    settings = LightStoreSettings(steps_per_chunk=48, codec="zstd-3")
    planes = store.light_patch_planes(
        LightmapCpuSession(time_batch_size=16), lit_scene["dem"], lit_scene["horizons"], 0, 0,
        lit_scene["sun"], lit_scene["earth"], settings,
    )
    path = tmp_path / "scene.lit"
    store.write_patch(path, planes, settings, patch_row=0, patch_column=0, first_step=0)
    with LightStore(path, settings) as opened:
        assert np.array_equal(opened.read_steps(0, 150), planes)
        assert np.array_equal(opened.read_steps(40, 60), planes[:, 40:60])


def test_the_producer_rejects_bad_vectors_and_patches(lit_scene) -> None:
    session = LightmapCpuSession(time_batch_size=8)
    with pytest.raises(InputError) as shape:
        store.light_patch_planes(session, lit_scene["dem"], lit_scene["horizons"], 0, 0,
                                 lit_scene["sun"], lit_scene["earth"][:-1])
    assert shape.value.code == "light_store_vectors_invalid"
    with pytest.raises(InputError) as outside:
        store.light_patch_planes(session, lit_scene["dem"], lit_scene["horizons"], 5, 5,
                                 lit_scene["sun"], lit_scene["earth"])
    assert outside.value.code == "light_store_patch_outside"


def test_a_session_with_its_own_planes_method_is_used_and_given_the_patch_geometry(lit_scene) -> None:
    calls = []

    class FakeSession:
        def light_patch_planes(self, dem, horizons, sun, earth, settings, **geometry):
            calls.append((sun.shape, earth.shape, settings, geometry))
            return np.zeros((3, sun.shape[0], PATCH_SIZE, 2), dtype=np.uint64)

    settings = LightStoreSettings(shadow_fraction=0.3)
    result = store.light_patch_planes(FakeSession(), lit_scene["dem"], lit_scene["horizons"], 0, 0,
                                      lit_scene["sun"], lit_scene["earth"], settings)
    assert result.shape == (3, 150, PATCH_SIZE, 2)
    shapes_sun, shapes_earth, used, geometry = calls[0]
    assert shapes_sun == shapes_earth == (150, 3) and used is settings
    assert geometry == dict(tile_y=0, tile_x=0, valid_height=scene.SIZE, valid_width=scene.SIZE)


def test_plane_thresholds_carry_the_stage_one_fuzz() -> None:
    shadow, operating, earth = store.plane_thresholds(LightStoreSettings(
        shadow_fraction=0.25, operating_fraction=0.9, earth_elevation_deg=3.5))
    assert (shadow, earth) == (0.25, 3.5) and operating == pytest.approx(0.9 - 1e-6)
