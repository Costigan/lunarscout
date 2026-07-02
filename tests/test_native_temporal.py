from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import lunarscout as ls
import lunarscout.native_temporal as native_temporal


def _georef() -> ls.GeoReference:
    return ls.GeoReference(
        projection_wkt='PROJCS["test",GEOGCS["g",DATUM["d",SPHEROID["s",1,0]],PRIMEM["p",0],UNIT["degree",0.0174532925199433]],PROJECTION["Equirectangular"],PARAMETER["standard_parallel_1",0],PARAMETER["central_meridian",0],PARAMETER["false_easting",0],PARAMETER["false_northing",0],UNIT["metre",1]]',
        projection_proj4="+proj=eqc +R=1 +units=m +no_defs",
        affine_transform=(0.0, 1.0, 0.0, 2.0, 0.0, -1.0),
        width=2,
        height=2,
        pixel_size_x=1.0,
        pixel_size_y=-1.0,
        nodata=-9999.0,
    )


def _inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    root = tmp_path / "scenario"
    root.mkdir()
    dem = root / "dem.tif"
    dem.touch()
    horizons = root / "horizons"
    horizons.mkdir(parents=True)
    return root, dem, horizons


def _components(*, incomplete: bool = False):
    class Request:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class SignalSpec:
        def __init__(self, *, signal):
            self.signal = signal

    def stream(_client, request, **_kwargs):
        assert request.signals[0].signal == "sun_fraction_u8"
        yield SimpleNamespace(
            rank=4,
            patch_row=0,
            patch_col=0,
            height=1,
            width=2,
            time_offset=0,
            time_count=2,
        ), np.asarray([[[[0, 255]]], [[[128, 64]]]], dtype=np.uint8)
        if not incomplete:
            yield SimpleNamespace(
                rank=4,
                patch_row=1,
                patch_col=0,
                height=1,
                width=2,
                time_offset=0,
                time_count=2,
            ), np.asarray([[[[255, 0]]], [[[32, 16]]]], dtype=np.uint8)

    return SimpleNamespace(
        Request=Request,
        Client=object,
        SignalSpec=SignalSpec,
        stream=stream,
    )


def _generate(tmp_path: Path, *, storage="memory", **kwargs):
    root, dem, horizons = _inputs(tmp_path)
    return native_temporal.generate_temporal_signal(
        signal="sun_fraction",
        scenario_root=root,
        dem_path=dem,
        horizons_path=horizons,
        times=ls.times("2027-01-01", "2027-01-01T01:00:00", step_hours=1),
        georef=_georef(),
        storage=storage,
        _client=object(),
        _components=_components(),
        **kwargs,
    )


def test_memory_generation_returns_fractional_temporal_cube(tmp_path: Path) -> None:
    progress: list[native_temporal.NativeTemporalProgress] = []

    cube = _generate(tmp_path, progress_callback=progress.append)

    assert isinstance(cube, ls.TemporalCube)
    assert cube.dtype == np.dtype(np.float32)
    np.testing.assert_allclose(
        cube.values,
        np.asarray(
            [
                [[0.0, 1.0], [1.0, 0.0]],
                [[128 / 255, 64 / 255], [32 / 255, 16 / 255]],
            ],
            dtype=np.float32,
        ),
    )
    assert cube.georef.nodata is None
    assert progress[0].stage == "preflight"
    assert progress[-1].stage == "complete"
    assert progress[-1].percent == 100.0


def test_file_generation_uses_scratch_and_returns_completed_series(
    tmp_path: Path,
) -> None:
    output = tmp_path / "sun.temporal"
    scratch = tmp_path / "scratch"

    series = _generate(
        tmp_path,
        storage="geotiff_series",
        output_path=output,
        scratch_directory=scratch,
    )

    assert isinstance(series, ls.TemporalGeoTiffSeries)
    assert series.signal_name == "sun_fraction"
    assert series.units == "fraction"
    assert series.manifest["provenance"]["generator"] == "MoonlibBridge"
    file_values = np.stack(
        [series.read_layer(index)[0] for index in range(series.shape[0])]
    )
    expected_u8 = np.asarray(
        [
            [[0, 255], [255, 0]],
            [[128, 64], [32, 16]],
        ],
        dtype=np.uint8,
    )
    expected = np.empty(expected_u8.shape, dtype=np.float32)
    np.multiply(expected_u8, 1.0 / 255.0, out=expected, casting="unsafe")
    assert file_values.dtype == np.float32
    assert np.array_equal(file_values, expected)
    memory_root = tmp_path / "memory"
    memory_root.mkdir()
    assert np.array_equal(file_values, _generate(memory_root).values)
    assert not list(scratch.glob(".lunarscout-native-temporal-*"))


def test_memory_preflight_rejects_unsafe_allocation_before_native_start(
    tmp_path: Path,
) -> None:
    root, dem, horizons = _inputs(tmp_path)

    with pytest.raises(ls.NativeAllocationError) as raised:
        native_temporal.generate_temporal_signal(
            signal="sun_fraction",
            scenario_root=root,
            dem_path=dem,
            horizons_path=horizons,
            times=ls.times("2027-01-01", "2027-01-01T01:00:00", step_hours=1),
            georef=_georef(),
            storage="memory",
            max_in_memory_bytes=31,
        )

    assert raised.value.code == "native_temporal_memory_limit_exceeded"
    assert raised.value.details == {"estimated_bytes": 32, "limit_bytes": 31}


def test_file_storage_requires_output_path(tmp_path: Path) -> None:
    root, dem, horizons = _inputs(tmp_path)

    with pytest.raises(ls.NativeInputError) as raised:
        native_temporal.generate_temporal_signal(
            signal="sun_fraction",
            scenario_root=root,
            dem_path=dem,
            horizons_path=horizons,
            times=ls.times("2027-01-01", "2027-01-01", step_hours=1),
            georef=_georef(),
            storage="geotiff_series",
        )

    assert raised.value.code == "native_temporal_output_required"


def test_incomplete_native_stream_fails_without_publishing_output(
    tmp_path: Path,
) -> None:
    root, dem, horizons = _inputs(tmp_path)
    output = tmp_path / "sun.temporal"

    with pytest.raises(ls.NativeTemporalError) as raised:
        native_temporal.generate_temporal_signal(
            signal="sun_fraction",
            scenario_root=root,
            dem_path=dem,
            horizons_path=horizons,
            times=ls.times("2027-01-01", "2027-01-01T01:00:00", step_hours=1),
            georef=_georef(),
            storage="geotiff_series",
            output_path=output,
            _client=object(),
            _components=_components(incomplete=True),
        )

    assert raised.value.code == "native_temporal_stream_incomplete"
    assert not output.exists()
    assert not list(tmp_path.glob(".sun.temporal.staging-*"))


def test_cancellation_cleans_scratch_and_staging(tmp_path: Path) -> None:
    root, dem, horizons = _inputs(tmp_path)
    output = tmp_path / "sun.temporal"
    calls = 0

    def cancelled() -> bool:
        nonlocal calls
        calls += 1
        return calls >= 4

    with pytest.raises(ls.NativeTemporalError) as raised:
        native_temporal.generate_temporal_signal(
            signal="sun_fraction",
            scenario_root=root,
            dem_path=dem,
            horizons_path=horizons,
            times=ls.times("2027-01-01", "2027-01-01T01:00:00", step_hours=1),
            georef=_georef(),
            storage="geotiff_series",
            output_path=output,
            cancellation_requested=cancelled,
            _client=object(),
            _components=_components(),
        )

    assert raised.value.code == "native_temporal_cancelled"
    assert not output.exists()
    assert not list(tmp_path.glob(".sun.temporal.staging-*"))
    assert not list(tmp_path.glob(".lunarscout-native-temporal-*"))


def test_cancelled_overwrite_preserves_output_and_restart_succeeds(
    tmp_path: Path,
) -> None:
    root, dem, horizons = _inputs(tmp_path)
    output = tmp_path / "sun.temporal"
    arguments = {
        "signal": "sun_fraction",
        "scenario_root": root,
        "dem_path": dem,
        "horizons_path": horizons,
        "times": ls.times("2027-01-01", "2027-01-01T01:00:00", step_hours=1),
        "georef": _georef(),
        "storage": "geotiff_series",
        "output_path": output,
        "_client": object(),
        "_components": _components(),
    }
    baseline = native_temporal.generate_temporal_signal(**arguments)
    baseline.close()
    before = {
        path.relative_to(output): path.read_bytes()
        for path in output.rglob("*")
        if path.is_file()
    }
    calls = 0

    def cancelled() -> bool:
        nonlocal calls
        calls += 1
        return calls >= 4

    with pytest.raises(ls.NativeTemporalError) as raised:
        native_temporal.generate_temporal_signal(
            **arguments, overwrite=True, cancellation_requested=cancelled
        )

    assert raised.value.code == "native_temporal_cancelled"
    after = {
        path.relative_to(output): path.read_bytes()
        for path in output.rglob("*")
        if path.is_file()
    }
    assert after == before
    assert not list(tmp_path.glob(".sun.temporal.staging-*"))
    assert not list(tmp_path.glob(".lunarscout-native-temporal-*"))

    restarted = native_temporal.generate_temporal_signal(**arguments, overwrite=True)
    assert restarted.shape == (2, 2, 2)
    expected_restart = np.empty((2, 2), dtype=np.float32)
    np.multiply(
        np.asarray([[128, 64], [32, 16]], dtype=np.uint8),
        1.0 / 255.0,
        out=expected_restart,
        casting="unsafe",
    )
    assert np.array_equal(
        restarted.read_layer(1)[0],
        expected_restart,
    )


def test_estimate_reports_exact_output_shape_and_dtype() -> None:
    estimate = native_temporal.estimate_temporal_allocation(
        signal="earth_over_horizon_deg",
        times=ls.times("2027-01-01", "2027-01-01T02:00:00", step_hours=1),
        georef=_georef(),
        storage="memory",
        max_in_memory_bytes=100,
    )

    assert estimate.shape == (3, 2, 2)
    assert estimate.dtype == np.dtype(np.float32)
    assert estimate.estimated_bytes == 48
    assert estimate.limit_bytes == 100


def test_streaming_client_is_built_around_moonlib_bridge(monkeypatch) -> None:
    bridge = object()
    moonlib = SimpleNamespace(MoonlibBridge=lambda: bridge)
    bootstrap = SimpleNamespace(import_moonlib=lambda **_kwargs: moonlib)
    monkeypatch.setattr("lunarscout.native._bootstrap_module", lambda: bootstrap)

    class Client:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    client = native_temporal._create_streaming_client(Client)

    assert client.kwargs == {"bridge": bridge, "moonlib_module": moonlib}


class _FakeFillPipeline:
    class FillLightmapBufferState:
        Filled = "Filled"
        Error = "Error"

    class FillLightmapRunState:
        Running = "Running"
        Completed = "Completed"

    class FillLightmapBuffersRequest:
        def __init__(
            self,
            dem_path,
            horizon_dir,
            timestamps,
            patch_width,
            patch_height,
            max_read_parallelism,
            max_compute_parallelism,
            queue_capacity,
            use_spice_sun_vectors,
        ):
            self.dem_path = dem_path
            self.horizon_dir = horizon_dir
            self.timestamps = timestamps
            self.patch_width = patch_width
            self.patch_height = patch_height
            self.max_read_parallelism = max_read_parallelism
            self.max_compute_parallelism = max_compute_parallelism
            self.queue_capacity = queue_capacity
            self.use_spice_sun_vectors = use_spice_sun_vectors

    class FillLightmapAvailableBuffer:
        def __init__(self, buffer_id, pointer, byte_length):
            self.BufferId = buffer_id
            self.Pointer = pointer
            self.ByteLength = byte_length


class _FakeFilled:
    def __init__(self, *, buffer_id: int, tile_id: int, state: str = "Filled"):
        self.BufferId = buffer_id
        self.TileId = tile_id
        self.PatchRow = tile_id
        self.PatchCol = 0
        self.Width = 2
        self.Height = 2
        self.TimeCount = 2
        self.State = state
        self.Message = "tile failed" if state == "Error" else None


class _FakePollResult:
    def __init__(self, *, filled, state="Running"):
        self.FilledBuffers = filled
        self.State = state
        self.Progress01 = 1.0 if state == "Completed" else 0.0
        self.Message = None


class _FakeFiller:
    def __init__(self):
        self.started = None
        self.polls = 0
        self.offered_ids: list[list[int]] = []
        self.disposed = False

    def Start(self, request):
        self.started = request

    def Poll(self, offered, _timeout):
        self.polls += 1
        ids = [int(item.BufferId) for item in offered]
        self.offered_ids.append(ids)
        if self.polls == 1:
            return _FakePollResult(filled=[_FakeFilled(buffer_id=ids[0], tile_id=0)])
        if self.polls == 2:
            return _FakePollResult(
                filled=[_FakeFilled(buffer_id=ids[0], tile_id=1)],
                state="Completed",
            )
        return _FakePollResult(filled=[], state="Completed")

    def Dispose(self):
        self.disposed = True


def test_stream_lightmap_buffers_reuses_python_owned_pool(tmp_path: Path) -> None:
    root, dem, horizons = _inputs(tmp_path)
    (horizons / "horizon_00000_00000_000.cbin").touch()
    (horizons / "horizon_00001_00000_000.cbin").touch()
    filler = _FakeFiller()
    progress: list[native_temporal.NativeTemporalProgress] = []

    patches = list(
        native_temporal.stream_lightmap_buffers(
            dem_path=dem,
            horizons_path=horizons,
            times=ls.times("2027-01-01", "2027-01-01T01:00:00", step_hours=1),
            buffer_count=1,
            patch_width=2,
            patch_height=2,
            progress_callback=progress.append,
            _pipeline=_FakeFillPipeline,
            _filler=filler,
            _array_factory=lambda _cls, values: values,
        )
    )

    assert filler.started.dem_path == str(dem.resolve())
    assert filler.offered_ids == [[0], [0]]
    assert filler.disposed is True
    assert [patch.tile_id for patch in patches] == [0, 1]
    assert patches[0].values is patches[1].values
    assert patches[0].values.shape == (2, 2, 2)
    assert patches[0].values.dtype == np.uint8
    assert progress[-1].stage == "complete"


def test_stream_lightmap_buffers_surfaces_tile_error(tmp_path: Path) -> None:
    class ErrorFiller(_FakeFiller):
        def Poll(self, offered, _timeout):
            self.polls += 1
            ids = [int(item.BufferId) for item in offered]
            self.offered_ids.append(ids)
            return _FakePollResult(
                filled=[_FakeFilled(buffer_id=ids[0], tile_id=0, state="Error")]
            )

    root, dem, horizons = _inputs(tmp_path)
    (horizons / "horizon_00000_00000_000.cbin").touch()
    filler = ErrorFiller()

    with pytest.raises(ls.NativeTemporalError) as raised:
        list(
            native_temporal.stream_lightmap_buffers(
                dem_path=dem,
                horizons_path=horizons,
                times=ls.times("2027-01-01", "2027-01-01", step_hours=1),
                buffer_count=1,
                patch_width=2,
                patch_height=2,
                _pipeline=_FakeFillPipeline,
                _filler=filler,
                _array_factory=lambda _cls, values: values,
            )
        )

    assert raised.value.code == "native_temporal_fill_buffers_tile_failed"
    assert filler.disposed is True


def test_scenario_sun_fraction_uses_standard_paths_and_safe_output(
    tmp_path: Path,
) -> None:
    root = tmp_path / "scenario"
    (root / "horizons").mkdir(parents=True)
    georef = _georef()
    ls.write_geotiff(root / "dem.tif", np.zeros((2, 2), dtype=np.float32), georef)
    scenario = ls.open_scenario(root)

    series = scenario.sun_fraction(
        times=ls.times("2027-01-01", "2027-01-01T01:00:00", step_hours=1),
        storage="geotiff_series",
        output="analysis/sun.temporal",
        _client=object(),
        _components=_components(),
    )

    assert series.root == root / "analysis" / "sun.temporal"
    assert series.shape == (2, 2, 2)
