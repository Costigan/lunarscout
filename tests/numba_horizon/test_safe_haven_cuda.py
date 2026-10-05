"""Opt-in real-device coverage for resident safe-haven reduction."""

from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess
import sys
import textwrap

import numpy as np
import pytest

from lunarscout._numba_horizon.geometry import DemGrid, ProjectionParameters
from lunarscout._numba_horizon.safe_haven import reduce_safe_haven_patch_stream
from lunarscout._numba_horizon.safe_haven_cuda import (
    SafeHavenCudaCancelled,
    SafeHavenCudaSession,
)


pytestmark = pytest.mark.skipif(
    os.environ.get("LUNARSCOUT_REQUIRE_NUMBA_CUDA") != "1",
    reason="set LUNARSCOUT_REQUIRE_NUMBA_CUDA=1 for resident CUDA reduction tests",
)


@pytest.fixture(scope="module")
def inputs():
    dem = DemGrid(
        np.zeros((2, 3), dtype=np.float32),
        np.array((1000., 20., 0., -1000., 0., -20.)),
        ProjectionParameters(1737400., -np.pi / 2, 0., 1., 0., 0.),
    )
    bounds = tuple(datetime(2027, month, 1, tzinfo=timezone.utc) for month in (1, 2, 3, 4))
    months = tuple(zip(bounds[:-1], bounds[1:]))
    mapping = np.array([0] * 4 + [1] * 4 + [2] * 3, dtype=np.int32)
    fractions = np.full((11, 2, 3), 0.1, dtype=np.float32)
    earth = np.full_like(fractions, 3.0)
    # Six pixels: three-month run; run crossing an unqualified month;
    # threshold equality; no outage; permanent outage; censored final run.
    fractions[10, 0, 0] = 0.8
    earth[[2, 5, 9], 0, 0] = 1.0
    fractions[6:, 0, 1] = 0.8
    earth[[2, 7], 0, 1] = 1.0
    fractions[:, 0, 2] = np.float32(0.2)
    earth[[2, 5, 9], 0, 2] = 1.0
    earth[:, 1, 1] = 1.0
    fractions[:3, 1, 2] = 0.8
    earth[:, 1, 2] = np.float32(2.0)
    earth[[3, 5, 9], 1, 2] = 1.0
    return dem, np.zeros((128, 128, 1440), dtype=np.float32), months, mapping, fractions, earth


class SignalKernel:
    """Supply controlled device signals while exercising real reducer kernels."""

    def __init__(self, fractions, earth):
        self.fractions = fractions
        self.earth = earth

    def __getitem__(self, _launch):
        def launch(*args):
            start, count = args[5:7]
            source = self.earth if args[-1] else self.fractions
            buffer = args[13] if args[-1] else args[12]
            # NaNs in padded pixels must not poison valid-pixel reduction.
            batch = np.full((count, 128, 128), np.nan, dtype=np.float32)
            batch[:, :2, :3] = source[start:start + count]
            buffer[:count].copy_to_device(batch.reshape(count, 128 * 128))
        return launch


def calculate(session, inputs, *, cancellation_requested=None):
    dem, horizons, months, mapping, fractions, earth = inputs
    session._kernel = SignalKernel(fractions, earth)
    vectors = np.ones((len(mapping), 3), dtype=np.float64)
    return session.reduce_patch(
        dem, horizons, vectors, vectors, tile_y=0, tile_x=0,
        valid_height=2, valid_width=3, month_bands=months,
        month_index_of=mapping, time_count=len(mapping), sunlight_threshold=0.2,
        earth_threshold_deg=2.0, time_step_hours=2.5,
        cancellation_requested=cancellation_requested,
    )


def reference(inputs):
    _, _, months, mapping, fractions, earth = inputs
    return reduce_safe_haven_patch_stream(
        iter(fractions), iter(earth), len(mapping), months,
        month_index_of=mapping, sunlight_threshold=0.2,
        earth_threshold_deg=2.0, time_step_hours=2.5,
    )


@pytest.mark.parametrize("batch_size", [1, 3, 8])
def test_month_batch_boundary_parity_and_patch_reset(inputs, batch_size):
    session = SafeHavenCudaSession(time_batch_size=batch_size)
    expected = np.stack(reference(inputs))
    np.testing.assert_array_equal(calculate(session, inputs), expected)
    np.testing.assert_array_equal(expected[:, 0, 0], [25., 25., 25.])
    np.testing.assert_array_equal(expected[:, 0, 1], [15., 0., np.nan])
    np.testing.assert_array_equal(expected[:, 0, 2], [0., 0., 0.])
    np.testing.assert_array_equal(expected[:, 1, 2], [20., 20., 20.])
    assert np.isnan(expected[:, 1, :2]).all()
    allocation = session._duration_output
    changed = (*inputs[:4], np.full_like(inputs[4], 0.8), inputs[5])
    np.testing.assert_array_equal(calculate(session, changed), reference(changed))
    assert session._duration_output is allocation
    # Mutating an existing mapping must invalidate its content cache.
    mapping = inputs[3].copy()
    altered = (*inputs[:3], mapping, inputs[4], inputs[5])
    calculate(session, altered)
    previous = session._month_map
    mapping[-2:] = 1
    np.testing.assert_array_equal(calculate(session, altered), reference(altered))
    assert session._month_map is not previous


@pytest.mark.parametrize("signal", ["fraction", "earth"])
def test_nonfinite_device_signals_are_rejected(inputs, signal):
    fraction, earth = inputs[4].copy(), inputs[5].copy()
    (fraction if signal == "fraction" else earth)[2, 0, 1] = np.nan
    modified = (*inputs[:4], fraction, earth)
    with pytest.raises(ValueError, match="non-finite"):
        calculate(SafeHavenCudaSession(time_batch_size=3), modified)


def test_cuda_cancellation_and_reused_session(inputs):
    session = SafeHavenCudaSession(time_batch_size=3)
    checks = 0
    def cancelled():
        nonlocal checks
        checks += 1
        return checks >= 2
    with pytest.raises(SafeHavenCudaCancelled):
        calculate(session, inputs, cancellation_requested=cancelled)
    np.testing.assert_array_equal(calculate(session, inputs), reference(inputs))


def test_only_finished_months_and_error_flag_copy_to_host(inputs, monkeypatch):
    from numba.cuda.cudadrv.devicearray import DeviceNDArray
    session = SafeHavenCudaSession(time_batch_size=3)
    copied = []
    original = DeviceNDArray.copy_to_host
    def record_copy(array, *args, **kwargs):
        copied.append(array.shape)
        return original(array, *args, **kwargs)
    monkeypatch.setattr(DeviceNDArray, "copy_to_host", record_copy)
    np.testing.assert_array_equal(calculate(session, inputs), reference(inputs))
    assert copied == [(1,), (3, 128 * 128)]


def test_public_byte_product_in_fresh_process(tmp_path):
    repo = Path(__file__).resolve().parents[2]
    environment = dict(os.environ, PYTHONPATH=str(repo / "src"))
    code = textwrap.dedent('''
        from pathlib import Path
        import sys
        import numpy as np
        import rasterio
        from rasterio.transform import Affine
        from pyproj import CRS
        import lunarscout as ls
        from lunarscout._numba_horizon.file_format import HorizonTileStore
        from lunarscout._numba_horizon.psr import _pixel_frame
        from lunarscout._numba_horizon import lightmap_cuda, safe_haven_pipeline
        from lunarscout.products import _load_dem

        dem_path = Path('dem.tif')
        with rasterio.open(dem_path, 'w', driver='GTiff', width=3, height=2,
                           count=1, dtype='float32', crs=CRS.from_user_input('ESRI:103878'),
                           transform=Affine(20, 0, 1000, 0, -20, -1000)) as dataset:
            dataset.write(np.zeros((2, 3), dtype=np.float32), 1)
        horizons = np.zeros((6, 1440), dtype=np.float32)
        horizons[0] = 45.0  # Undefined months must become byte nodata 255.
        HorizonTileStore(Path('horizons')).write(0, 0, 0., horizons,
                                               compress=True, valid_width=3, valid_height=2)
        dem, _ = _load_dem(dem_path)
        rotation, translation = _pixel_frame(dem, 0, 0)
        def vectors(elevations, distance):
            angles = np.deg2rad(elevations)
            local = np.column_stack((np.zeros(6), np.cos(angles), np.sin(angles))) * distance
            return (local - translation) @ rotation.T
        sun = vectors([-5, -5, -5, -5, 5, 5], 150e9)
        earth = vectors([1, 10, 1, 10, 10, 10], 384e6)
        def encode(hours):
            return np.where(np.isnan(hours), 255, np.clip(np.ceil(hours / 2), 0, 255)).astype(np.uint8)
        kwargs = dict(times=ls.times('2027-01-31T12:00:00Z', '2027-02-01T18:00:00Z', step_hours=6),
                      sun_vectors_m=sun, earth_vectors_m=earth, nodata=255,
                      output_transform=encode, output_dtype=np.uint8,
                      output_transform_id='test-two-hour-byte-v1', compress=True)
        cpu = ls.generate_safe_havens(dem_path, 'horizons', 'cpu.tif', backend='cpu', **kwargs)
        def forbidden(*args, **kwargs):
            raise AssertionError('Host lighting/reduction path called by CUDA product')
        safe_haven_pipeline.reduce_safe_haven_patch_stream = forbidden
        lightmap_cuda.LightmapCudaSession.iter_patch_fraction_tiles = forbidden
        lightmap_cuda.LightmapCudaSession.iter_patch_margin_tiles = forbidden
        gpu = ls.generate_safe_havens(dem_path, 'horizons', 'cuda.tif', backend='cuda', **kwargs)
        assert 'spiceypy' not in sys.modules
        with rasterio.open(cpu) as expected, rasterio.open(gpu) as actual:
            np.testing.assert_array_equal(actual.read(), expected.read())
            assert actual.dtypes == ('uint8', 'uint8')
            assert actual.nodata == 255
            assert actual.block_shapes == [(128, 128), (128, 128)]
            assert actual.compression.value.upper() == 'DEFLATE'
            assert actual.tags()['LUNARSCOUT_COMPUTE_BACKENDS'] == '["cuda"]'
            assert (actual.read()[:, 0, 0] == 255).all()
            assert (actual.read()[:, 1, :] == 12).all()
    ''')
    completed = subprocess.run([sys.executable, "-c", code], cwd=tmp_path,
                               env=environment, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stdout + completed.stderr
