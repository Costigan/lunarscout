from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import lunarscout.map_algebra._writer as writer_module
from lunarscout.errors import OperationCancelledError
from lunarscout.map_algebra import (
    source,
    slope as ma_slope,
    resample_to,
    write,
)

from .test_writer import _write_tiff
from .conftest import north_up_georef  # noqa: F401


_PROGRESS_LOG = list[tuple[int, int, int]]()


def _make_progress_callback(
    log: list[tuple[int, int, int]],
    *,
    fail_after: int | None = None,
) -> object:
    def cb(completed: int, total: int, window_idx: int) -> None:
        log.append((completed, total, window_idx))
        if fail_after is not None and completed >= fail_after:
            raise RuntimeError("progress callback failure")

    return cb


class TestProgress:
    def test_progress_monotonic_and_exact_completion(self, tmp_path):
        p = _write_tiff(tmp_path, "src.tif", np.ones((257, 257), dtype=np.float32))
        out = tmp_path / "out.tif"
        expr = source(p) + 1
        progress_log: list[tuple[int, int, int]] = []
        result = write(out, expr, progress_callback=_make_progress_callback(progress_log))
        assert result == out
        assert len(progress_log) > 0
        prev = 0
        for completed, total, idx in progress_log:
            assert completed == prev + 1 or (prev == 0 and completed == prev + 1) or (
                completed == total and idx == -1
            ), f"non-monotonic: {completed=} {prev=} {total=}"
            assert completed <= total
            prev = max(prev, completed)
        final = progress_log[-1]
        assert final[0] == final[1]
        assert final[2] == final[1] - 1
        assert sum(completed == total for completed, total, _ in progress_log) == 1
        assert prev == final[0]

    def test_progress_callback_failure_does_not_publish(self, tmp_path):
        p = _write_tiff(tmp_path, "src.tif", np.ones((300, 300), dtype=np.float32))
        out = tmp_path / "out.tif"
        expr = source(p) + 1
        progress_log: list[tuple[int, int, int]] = []
        with pytest.raises(RuntimeError, match="progress callback failure"):
            write(
                out, expr,
                progress_callback=_make_progress_callback(progress_log, fail_after=2),
                window_width=100,
                window_height=100,
                checkpoint_interval=2,
            )
        assert not out.exists()
        staging = writer_module._staging_tiff_path(out.resolve())
        journal = writer_module._staging_journal_path(out.resolve())
        assert staging.exists()
        assert journal.exists()

        write(
            out, expr,
            window_width=100,
            window_height=100,
            checkpoint_interval=2,
        )
        import rasterio
        with rasterio.open(out) as ds:
            np.testing.assert_array_equal(
                ds.read(1), np.full((300, 300), 2.0, dtype=np.float32),
            )
            assert np.all(ds.read_masks(1) == 255)

    def test_progress_callback_does_not_affect_scientific_result(self, tmp_path):
        p = _write_tiff(tmp_path, "src.tif", np.ones((10, 10), dtype=np.float32))
        out1 = tmp_path / "out1.tif"
        out2 = tmp_path / "out2.tif"
        expr = source(p) + 1
        progress_log: list[tuple[int, int, int]] = []
        write(out1, expr)
        write(out2, expr, progress_callback=_make_progress_callback(progress_log))
        import rasterio
        with rasterio.open(out1) as d1, rasterio.open(out2) as d2:
            np.testing.assert_array_equal(d1.read(1), d2.read(1))

    def test_no_progress_without_callback(self, tmp_path):
        p = _write_tiff(tmp_path, "src.tif", np.ones((10, 10), dtype=np.float32))
        out = tmp_path / "out.tif"
        expr = source(p) + 1
        result = write(out, expr)
        assert result == out


class TestCancellation:
    def test_cancellation_before_execution(self, tmp_path):
        p = _write_tiff(tmp_path, "src.tif", np.ones((10, 10), dtype=np.float32))
        out = tmp_path / "out.tif"
        expr = source(p) + 1
        with pytest.raises(OperationCancelledError, match="cancelled before execution"):
            write(out, expr, cancellation_requested=lambda: True)
        assert not out.exists()

    def test_cancellation_after_several_windows(self, tmp_path):
        p = _write_tiff(tmp_path, "src.tif", np.ones((300, 300), dtype=np.float32))
        out = tmp_path / "out.tif"
        expr = source(p) + 1
        cancel_calls = [0]

        def cancel_check() -> bool:
            cancel_calls[0] += 1
            return cancel_calls[0] > 3

        with pytest.raises(OperationCancelledError):
            write(out, expr, cancellation_requested=cancel_check,
                  window_width=100, window_height=100)
        assert not out.exists()
        staging = out.with_name("." + out.name + ".lunarscout-partial.tif")
        journal = out.with_name("." + out.name + ".lunarscout-partial.journal.json")
        assert staging.exists()
        assert journal.exists()

    def test_cancellation_with_existing_destination(self, tmp_path):
        p = _write_tiff(tmp_path, "src.tif", np.ones((10, 10), dtype=np.float32))
        out = tmp_path / "out.tif"
        expr = source(p) + 2
        write(out, expr)
        import rasterio
        with rasterio.open(out) as ds:
            orig = ds.read(1).copy()

        expr2 = source(p) + 5
        cancel_calls = [0]

        def cancel_check() -> bool:
            cancel_calls[0] += 1
            return True

        with pytest.raises(OperationCancelledError):
            write(out, expr2, cancellation_requested=cancel_check, overwrite=True)
        assert out.exists()
        with rasterio.open(out) as ds:
            data = ds.read(1)
            np.testing.assert_array_equal(data, orig)

    def test_cancellation_cleans_up_resources(self, tmp_path):
        p = _write_tiff(tmp_path, "src.tif", np.ones((300, 300), dtype=np.float32))
        out = tmp_path / "out.tif"
        expr = source(p) + 1
        cancel_calls = [0]

        def cancel_check() -> bool:
            cancel_calls[0] += 1
            return cancel_calls[0] > 2

        with pytest.raises(OperationCancelledError):
            write(out, expr, cancellation_requested=cancel_check,
                  window_width=100, window_height=100)
        staging = out.with_name("." + out.name + ".lunarscout-partial.tif")
        journal = out.with_name("." + out.name + ".lunarscout-partial.journal.json")
        assert staging.exists()
        assert journal.exists()
        import rasterio
        with rasterio.open(staging, "r+") as ds:
            assert ds.width == 300

    def test_cancellation_structured_error_code(self, tmp_path):
        p = _write_tiff(tmp_path, "src.tif", np.ones((10, 10), dtype=np.float32))
        out = tmp_path / "out.tif"
        with pytest.raises(OperationCancelledError) as exc_info:
            write(out, source(p) + 1, cancellation_requested=lambda: True)
        assert exc_info.value.code == "map_algebra_cancelled"
        assert "completed_windows" in exc_info.value.details
        assert "total_windows" in exc_info.value.details


class TestJournaling:
    def test_journal_created_and_updated(self, tmp_path):
        p = _write_tiff(tmp_path, "src.tif", np.ones((300, 300), dtype=np.float32))
        out = tmp_path / "out.tif"
        expr = source(p) + 1
        journal = out.with_name("." + out.name + ".lunarscout-partial.journal.json")

        calls = [0]

        def cancel() -> bool:
            calls[0] += 1
            return calls[0] > 3

        with pytest.raises(OperationCancelledError):
            write(
                out, expr,
                window_width=100,
                window_height=100,
                checkpoint_interval=1,
                cancellation_requested=cancel,
            )
        payload = json.loads(journal.read_text())
        assert payload["journal_format"] == 2
        assert payload["layout"] == "row_major_contiguous_prefix"
        assert payload["completed_windows"] == 2
        assert payload["total_windows"] == 9

    def test_journal_not_reused_for_different_expression(self, tmp_path):
        p = _write_tiff(tmp_path, "src.tif", np.ones((300, 300), dtype=np.float32))
        out = tmp_path / "out.tif"

        expr1 = source(p) + 1
        write(out, expr1, window_width=100, window_height=100)

        result2 = write(out, source(p) + 2, overwrite=True,
                        window_width=100, window_height=100)
        assert result2 == out

    def test_journal_not_reused_for_different_window_size(self, tmp_path):
        p = _write_tiff(tmp_path, "src.tif", np.ones((200, 200), dtype=np.float32))
        out = tmp_path / "out.tif"
        expr = source(p) + 1

        write(out, expr, window_width=100, window_height=100)

        result2 = write(out, expr, overwrite=True, window_width=50, window_height=50)
        assert result2 == out

    def test_journal_not_reused_for_different_dtype(self, tmp_path):
        p = _write_tiff(tmp_path, "src.tif",
                        np.ones((100, 100), dtype=np.float32))
        out = tmp_path / "out.tif"
        expr = source(p) + 1

        write(out, expr)

        result2 = write(out, expr, overwrite=True, dtype="float64")
        assert result2 == out

    def test_journal_not_reused_for_different_invalid_value(self, tmp_path):
        p = _write_tiff(tmp_path, "src.tif", np.ones((100, 100), dtype=np.float32))
        out = tmp_path / "out.tif"
        expr = source(p) + 1

        write(out, expr, invalid_value=-9999.0)

        result2 = write(out, expr, overwrite=True, invalid_value=-1.0)
        assert result2 == out

    def test_journal_not_reused_for_different_grid(self, tmp_path):
        p1 = _write_tiff(tmp_path, "src1.tif", np.ones((100, 100), dtype=np.float32))
        p2 = _write_tiff(tmp_path, "src2.tif", np.ones((200, 200), dtype=np.float32))
        out = tmp_path / "out.tif"
        expr1 = source(p1) + 1
        expr2 = source(p2) + 1

        write(out, expr1)
        result2 = write(out, expr2, overwrite=True)
        assert result2 == out

    def test_resume_skips_completed_windows(self, tmp_path):
        p = _write_tiff(tmp_path, "src.tif", np.ones((300, 300), dtype=np.float32))
        out = tmp_path / "out.tif"
        expr = source(p) + 1
        staging = out.with_name("." + out.name + ".lunarscout-partial.tif")
        journal = out.with_name("." + out.name + ".lunarscout-partial.journal.json")

        calls = [0]

        def cancel() -> bool:
            calls[0] += 1
            return calls[0] > 3

        with pytest.raises(OperationCancelledError):
            write(
                out, expr,
                window_width=100,
                window_height=100,
                checkpoint_interval=1,
                cancellation_requested=cancel,
            )
        assert json.loads(journal.read_text())["completed_windows"] == 2

        real_execute = writer_module._execute_window
        executed: list[int] = []

        def record_execute(plan, cache, idx, x0, y0, width, height):
            executed.append(idx)
            return real_execute(plan, cache, idx, x0, y0, width, height)

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(writer_module, "_execute_window", record_execute)
            result = write(
                out, expr,
                window_width=100,
                window_height=100,
                checkpoint_interval=1,
            )
        assert result == out
        assert executed == list(range(2, 9))
        assert not staging.exists()
        assert not journal.exists()

    def test_resume_without_restart_artifacts_does_full_calculation(self, tmp_path):
        p = _write_tiff(tmp_path, "src.tif", np.ones((100, 100), dtype=np.float32))
        out = tmp_path / "out.tif"
        expr = source(p) + 1

        result = write(out, expr)
        assert result == out
        staging = out.with_name("." + out.name + ".lunarscout-partial.tif")
        journal = out.with_name("." + out.name + ".lunarscout-partial.journal.json")
        assert not staging.exists()
        assert not journal.exists()

    def test_malformed_journal_handled(self, tmp_path):
        p = _write_tiff(tmp_path, "src.tif", np.ones((100, 100), dtype=np.float32))
        out = tmp_path / "out.tif"
        journal = out.with_name("." + out.name + ".lunarscout-partial.journal.json")
        journal.write_text("not valid json {{{")
        expr = source(p) + 1

        result = write(out, expr)
        assert result == out

    def test_truncated_journal_handled(self, tmp_path):
        p = _write_tiff(tmp_path, "src.tif", np.ones((100, 100), dtype=np.float32))
        out = tmp_path / "out.tif"
        journal = out.with_name("." + out.name + ".lunarscout-partial.journal.json")
        journal.write_text('{"journal_format": 1, "identity": "sha256:abc", "windows')
        expr = source(p) + 1

        result = write(out, expr)
        assert result == out

    def test_stale_journal_identity_ignored(self, tmp_path):
        p = _write_tiff(tmp_path, "src.tif", np.ones((100, 100), dtype=np.float32))
        out = tmp_path / "out.tif"
        journal = out.with_name("." + out.name + ".lunarscout-partial.journal.json")
        stale_journal = {
            "journal_format": 1,
            "identity": "sha256:different_identity_not_matching",
            "windows": [{"idx": 0}, {"idx": 1}],
        }
        journal.write_text(json.dumps(stale_journal))
        expr = source(p) + 1

        result = write(out, expr)
        assert result == out

    def test_out_of_range_journal_window_ignored(self, tmp_path):
        p = _write_tiff(tmp_path, "src.tif", np.ones((100, 100), dtype=np.float32))
        out = tmp_path / "out.tif"

        journal = out.with_name("." + out.name + ".lunarscout-partial.journal.json")
        expr = source(p) + 1

        from lunarscout.map_algebra._planner import plan_expression
        plan = plan_expression(expr, window_width=128, window_height=128)
        jid = writer_module._build_journal_identity(
            expr, np.dtype(np.float32), np.nan, 128, 128,
        )
        bad_journal = {
            "journal_format": 2,
            "identity": jid,
            "layout": "row_major_contiguous_prefix",
            "completed_windows": 999,
            "total_windows": 1,
        }
        journal.write_text(json.dumps(bad_journal))
        staging = out.with_name("." + out.name + ".lunarscout-partial.tif")
        ds = writer_module._create_staged_tiff(
            staging, 100, 100, np.dtype(np.float32), plan.grid,
            nodata=np.nan, journal_identity=jid,
        )
        ds.close()

        result = write(out, expr, overwrite=True)
        assert result == out
        import rasterio
        with rasterio.open(out) as output:
            np.testing.assert_array_equal(
                output.read(1), np.full((100, 100), 2.0, dtype=np.float32),
            )

    def test_incompatible_journal_format_ignored(self, tmp_path):
        p = _write_tiff(tmp_path, "src.tif", np.ones((100, 100), dtype=np.float32))
        out = tmp_path / "out.tif"
        journal = out.with_name("." + out.name + ".lunarscout-partial.journal.json")
        journal.write_text('{"journal_format": 999, "identity": "test", "windows": []}')
        expr = source(p) + 1

        result = write(out, expr)
        assert result == out

    def test_journal_cleanup_after_publication(self, tmp_path):
        p = _write_tiff(tmp_path, "src.tif", np.ones((300, 300), dtype=np.float32))
        out = tmp_path / "out.tif"
        expr = source(p) + 1
        staging = out.with_name("." + out.name + ".lunarscout-partial.tif")
        journal = out.with_name("." + out.name + ".lunarscout-partial.journal.json")

        result = write(out, expr, window_width=100, window_height=100)
        assert result == out
        assert not staging.exists()
        assert not journal.exists()
        assert out.exists()

    def test_non_divisible_edge_windows(self, tmp_path):
        p = _write_tiff(tmp_path, "src.tif", np.ones((250, 250), dtype=np.float32))
        out = tmp_path / "out.tif"
        expr = source(p) + 1

        result = write(out, expr, window_width=128, window_height=128)
        assert result == out
        import rasterio
        with rasterio.open(out) as ds:
            data = ds.read(1)
            assert data.shape == (250, 250)
            np.testing.assert_array_equal(data, np.full((250, 250), 2.0, dtype=np.float32))

    def test_progress_reports_non_divisible_edge_windows(self, tmp_path):
        p = _write_tiff(tmp_path, "src.tif", np.ones((127, 127), dtype=np.float32))
        out = tmp_path / "out.tif"
        expr = source(p) + 1
        progress_log: list[tuple[int, int, int]] = []
        result = write(out, expr, progress_callback=_make_progress_callback(progress_log))
        assert result == out
        final = progress_log[-1]
        assert final[0] == final[1]


class TestEagerVsWindowedParity:
    def test_eager_windowed_parity_after_cancel_and_resume(self, tmp_path):
        p = _write_tiff(tmp_path, "src.tif",
                        np.arange(200, dtype=np.float32).reshape(20, 10))
        out = tmp_path / "out.tif"
        expr = source(p) + 1

        result = write(out, expr)
        import rasterio
        from lunarscout.map_algebra import compute
        eager = compute(source(p) + 1)

        with rasterio.open(out) as ds:
            data = ds.read(1)
            np.testing.assert_array_equal(data, eager.values)


class TestCleanup:
    def test_successful_cleanup(self, tmp_path):
        p = _write_tiff(tmp_path, "src.tif", np.ones((100, 100), dtype=np.float32))
        out = tmp_path / "out.tif"
        staging = out.with_name("." + out.name + ".lunarscout-partial.tif")
        journal = out.with_name("." + out.name + ".lunarscout-partial.journal.json")
        manifest = out.with_suffix(out.suffix + ".manifest.json")
        backup = staging.with_name(staging.name + ".previous")

        result = write(out, source(p) + 1)
        assert result == out
        assert not staging.exists()
        assert not journal.exists()
        assert not backup.exists()
        assert out.exists()
        assert manifest.exists()

    def test_failed_overwrite_preserves_previous(self, tmp_path):
        p = _write_tiff(tmp_path, "src.tif", np.ones((10, 10), dtype=np.float32))
        out = tmp_path / "out.tif"
        expr = source(p) + 2
        write(out, expr)
        import rasterio
        with rasterio.open(out) as ds:
            orig = ds.read(1).copy()

        cancel_calls = [0]

        def cancel_check() -> bool:
            cancel_calls[0] += 1
            return True

        with pytest.raises(OperationCancelledError):
            write(out, source(p) + 5, cancellation_requested=cancel_check,
                  overwrite=True)
        assert out.exists()
        with rasterio.open(out) as ds:
            data = ds.read(1)
            np.testing.assert_array_equal(data, orig)


class TestNoRegression:
    def test_basic_write_without_lifecycle_params(self, tmp_path):
        p = _write_tiff(tmp_path, "src.tif", np.ones((10, 10), dtype=np.float32))
        out = tmp_path / "out.tif"
        result = write(out, source(p) + 1)
        assert result == out
        import rasterio
        with rasterio.open(out) as ds:
            data = ds.read(1)
            np.testing.assert_array_equal(data, np.full((10, 10), 2.0, dtype=np.float32))

    def test_overwrite_without_lifecycle_params(self, tmp_path):
        p = _write_tiff(tmp_path, "src.tif", np.ones((2, 2), dtype=np.float32))
        out = tmp_path / "out.tif"
        write(out, source(p) + 1)
        write(out, source(p) + 5, overwrite=True)
        import rasterio
        with rasterio.open(out) as ds:
            data = ds.read(1)
            np.testing.assert_array_equal(data, np.full((2, 2), 6.0, dtype=np.float32))

    def test_manifest_matches_restart_id(self, tmp_path):
        p = _write_tiff(tmp_path, "src.tif", np.ones((2, 2), dtype=np.float32))
        out = tmp_path / "out.tif"
        expr = source(p) + 1
        write(out, expr)
        mtime_before = out.stat().st_mtime
        result = write(out, expr)
        assert result == out
        assert out.stat().st_mtime == mtime_before

    def test_start_fresh(self, tmp_path):
        p = _write_tiff(tmp_path, "src.tif", np.ones((2, 2), dtype=np.float32))
        out = tmp_path / "out.tif"
        expr = source(p) + 1
        write(out, expr)
        result = write(out, expr, start_fresh=True)
        assert result == out

    def test_compound_expression_write(self, tmp_path):
        p = _write_tiff(tmp_path, "src.tif", np.ones((2, 2), dtype=np.float32))
        out = tmp_path / "out.tif"
        from lunarscout.map_algebra import cast
        expr = cast((source(p) + 1) * 3, "int32", casting="unsafe")
        result = write(out, expr, dtype="int32")
        import rasterio
        with rasterio.open(out) as ds:
            data = ds.read(1)
            np.testing.assert_array_equal(data, np.full((2, 2), 6, dtype=np.int32))

    def test_lossy_dtype_conversion_rejected_preflight(self, tmp_path):
        p = _write_tiff(tmp_path, "src.tif", np.ones((2, 2), dtype=np.float32))
        out = tmp_path / "out.tif"
        with pytest.raises(Exception):
            write(out, source(p) + 1.5, dtype="int8")


from .conftest import _georef  # noqa: E402


class TestTerrainAndResampleWriteLifecycle:
    def test_terrain_write_with_progress(self, tmp_path, north_up_georef):
        from lunarscout.map_algebra import raster
        dem_data = np.arange(
            north_up_georef.height * north_up_georef.width, dtype=np.float32,
        ).reshape(north_up_georef.height, north_up_georef.width)
        dem = raster(dem_data, north_up_georef, units="metres")
        out = tmp_path / "slope.tif"
        progress_log: list[tuple[int, int, int]] = []
        result = write(
            out, ma_slope(dem.expression()),
            progress_callback=_make_progress_callback(progress_log),
        )
        assert result == out
        assert len(progress_log) > 0

    def test_resample_write_with_progress(self, tmp_path, north_up_georef):
        from lunarscout.map_algebra import raster
        dem_data = np.ones(
            (north_up_georef.height, north_up_georef.width), dtype=np.float32,
        )
        dem = raster(dem_data, north_up_georef)
        shifted = _georef(
            width=5, height=4,
            origin_x=north_up_georef.affine_transform[0] + 5,
            origin_y=north_up_georef.affine_transform[3] + 5,
        )
        expr = resample_to(dem.expression(), shifted, resampling="nearest")
        out = tmp_path / "resampled.tif"
        cancel_calls = [0]

        def cancel() -> bool:
            cancel_calls[0] += 1
            return cancel_calls[0] > 3

        with pytest.raises(OperationCancelledError):
            write(
                out, expr,
                window_width=2,
                window_height=2,
                checkpoint_interval=1,
                cancellation_requested=cancel,
            )
        progress_log: list[tuple[int, int, int]] = []
        result = write(
            out, expr,
            window_width=2,
            window_height=2,
            checkpoint_interval=1,
            progress_callback=_make_progress_callback(progress_log),
        )
        assert result == out

    def test_eager_vs_windowed_terrain_parity_with_lifecycle(
        self, tmp_path, north_up_georef,
    ):
        from lunarscout.map_algebra import compute, raster
        dem_data = np.arange(
            north_up_georef.height * north_up_georef.width, dtype=np.float32,
        ).reshape(north_up_georef.height, north_up_georef.width)
        dem = raster(dem_data, north_up_georef, units="metres")

        eager = compute(ma_slope(dem.expression()))
        out = tmp_path / "slope.tif"
        cancel_calls = [0]

        def cancel() -> bool:
            cancel_calls[0] += 1
            return cancel_calls[0] > 3

        with pytest.raises(OperationCancelledError):
            write(
                out, ma_slope(dem.expression()),
                cancellation_requested=cancel,
                checkpoint_interval=1,
                window_width=3, window_height=3,
            )
        progress_log: list[tuple[int, int, int]] = []
        write(
            out, ma_slope(dem.expression()),
            progress_callback=_make_progress_callback(progress_log),
            window_width=3, window_height=3,
        )
        import rasterio
        with rasterio.open(out) as ds:
            data = ds.read(1)
            mask = ds.read_masks(1)
        valid = mask.astype(bool)
        np.testing.assert_allclose(
            data[valid], eager.values[valid], rtol=1e-5, atol=1e-5,
        )
        np.testing.assert_array_equal(valid, eager.valid)


class TestLifecycleRegressionFaults:
    def test_planner_reports_enforced_lifecycle_capabilities(self, tmp_path):
        p = _write_tiff(tmp_path, "src.tif", np.ones((2, 2), dtype=np.float32))
        expr = source(p) + 1
        from lunarscout.map_algebra._planner import plan_expression

        plan = plan_expression(expr, window_width=1, window_height=1)
        assert plan.total_windows == 4
        assert plan.journal_available
        assert plan.supports_progress
        assert plan.supports_cancellation
        assert plan.resumable_stages == ("windowed_execution",)
        inputs = plan.journal_identity_inputs(expr, np.dtype("float32"), np.nan)
        assert inputs["destination_grid"]["affine_transform"]
        assert inputs["window_layout"] == {
            "width": 1, "height": 1, "order": "row_major",
        }
        assert inputs["write_options"]["compression"] == "deflate"
        from lunarscout.map_algebra import plan as public_plan
        public = public_plan(expr)["planner"]
        assert public["journal_available"]
        assert public["supports_progress"]
        assert public["supports_cancellation"]
        assert public["resumable_stages"] == ("windowed_execution",)
        assert public["default_write_journal_identity"].startswith("sha256:")

    def test_changed_dtype_discards_incompatible_stage(self, tmp_path):
        p = _write_tiff(tmp_path, "src.tif", np.ones((300, 300), dtype=np.float32))
        out = tmp_path / "out.tif"
        calls = [0]

        def cancel() -> bool:
            calls[0] += 1
            return calls[0] > 3

        with pytest.raises(OperationCancelledError):
            write(
                out, source(p) + 1,
                window_width=100,
                window_height=100,
                checkpoint_interval=1,
                cancellation_requested=cancel,
            )

        import rasterio
        staging = writer_module._staging_tiff_path(out.resolve())
        with rasterio.open(staging) as ds:
            assert ds.dtypes[0] == "float32"

        write(
            out, source(p) + 1,
            dtype="float64",
            window_width=100,
            window_height=100,
        )
        with rasterio.open(out) as ds:
            assert ds.dtypes[0] == "float64"
            np.testing.assert_array_equal(ds.read(1), np.full((300, 300), 2.0))

    @pytest.mark.parametrize(
        "change", ["expression", "nodata", "window_layout", "checkpoint_interval"],
    )
    def test_changed_execution_identity_recomputes_every_window(
        self, tmp_path, monkeypatch, change,
    ):
        p = _write_tiff(tmp_path, "src.tif", np.ones((300, 300), dtype=np.float32))
        out = tmp_path / "out.tif"
        calls = [0]

        def cancel() -> bool:
            calls[0] += 1
            return calls[0] > 3

        with pytest.raises(OperationCancelledError):
            write(
                out, source(p) + 1,
                window_width=100,
                window_height=100,
                invalid_value=-9999.0,
                checkpoint_interval=1,
                cancellation_requested=cancel,
            )

        expression = source(p) + (2 if change == "expression" else 1)
        kwargs = {
            "window_width": 50 if change == "window_layout" else 100,
            "window_height": 50 if change == "window_layout" else 100,
            "invalid_value": -1.0 if change == "nodata" else -9999.0,
            "checkpoint_interval": 2 if change == "checkpoint_interval" else 1,
        }
        expected_total = 36 if change == "window_layout" else 9
        executed: list[int] = []
        real_execute = writer_module._execute_window

        def record_execute(plan, cache, idx, x0, y0, width, height):
            executed.append(idx)
            return real_execute(plan, cache, idx, x0, y0, width, height)

        monkeypatch.setattr(writer_module, "_execute_window", record_execute)
        write(out, expression, **kwargs)
        assert executed == list(range(expected_total))

    def test_changed_write_options_discard_stage(self, tmp_path, monkeypatch):
        p = _write_tiff(tmp_path, "src.tif", np.ones((300, 300), dtype=np.float32))
        out = tmp_path / "out.tif"
        calls = [0]

        def cancel() -> bool:
            calls[0] += 1
            return calls[0] > 3

        with pytest.raises(OperationCancelledError):
            write(
                out, source(p) + 1,
                window_width=100,
                window_height=100,
                checkpoint_interval=1,
                cancellation_requested=cancel,
            )

        monkeypatch.setitem(writer_module._GEOTIFF_WRITE_OPTIONS, "compression", "lzw")
        executed: list[int] = []
        real_execute = writer_module._execute_window

        def record_execute(plan, cache, idx, x0, y0, width, height):
            executed.append(idx)
            return real_execute(plan, cache, idx, x0, y0, width, height)

        monkeypatch.setattr(writer_module, "_execute_window", record_execute)
        write(out, source(p) + 1, window_width=100, window_height=100)
        assert executed == list(range(9))
        import rasterio
        with rasterio.open(out) as ds:
            assert ds.compression.value.lower() == "lzw"

    def test_changed_same_size_grid_discards_stage(self, tmp_path, monkeypatch):
        p = _write_tiff(tmp_path, "src.tif", np.ones((300, 300), dtype=np.float32))
        out = tmp_path / "out.tif"
        calls = [0]

        def cancel() -> bool:
            calls[0] += 1
            return calls[0] > 3

        with pytest.raises(OperationCancelledError):
            write(
                out, source(p) + 1,
                window_width=100,
                window_height=100,
                checkpoint_interval=1,
                cancellation_requested=cancel,
            )

        from lunarscout.map_algebra import raster
        shifted = _georef(width=300, height=300, origin_x=4321.0, origin_y=9876.0)
        expression = raster(
            np.ones((300, 300), dtype=np.float32), shifted,
        ).expression() + 1
        executed: list[int] = []
        real_execute = writer_module._execute_window

        def record_execute(plan, cache, idx, x0, y0, width, height):
            executed.append(idx)
            return real_execute(plan, cache, idx, x0, y0, width, height)

        monkeypatch.setattr(writer_module, "_execute_window", record_execute)
        write(out, expression, window_width=100, window_height=100)
        assert executed == list(range(9))
        import rasterio
        from rasterio.transform import Affine
        with rasterio.open(out) as ds:
            assert ds.transform.almost_equals(
                Affine.from_gdal(*shifted.affine_transform),
            )

    def test_failure_between_value_and_mask_write_resumes_safely(
        self, tmp_path, monkeypatch,
    ):
        p = _write_tiff(tmp_path, "src.tif", np.ones((300, 300), dtype=np.float32))
        out = tmp_path / "out.tif"
        real_write_window = writer_module._write_window
        injected = [False]

        def fail_before_mask(ds, idx, x0, y0, w, h, values, valid, fill, dtype):
            if idx == 2 and not injected[0]:
                injected[0] = True
                ds.write(values.astype(dtype, copy=False), 1,
                         window=((y0, y0 + h), (x0, x0 + w)))
                raise OSError("injected failure before validity write")
            return real_write_window(
                ds, idx, x0, y0, w, h, values, valid, fill, dtype,
            )

        monkeypatch.setattr(writer_module, "_write_window", fail_before_mask)
        with pytest.raises(OSError, match="before validity"):
            write(
                out, source(p) + 1,
                window_width=100,
                window_height=100,
                checkpoint_interval=1,
            )
        journal = writer_module._staging_journal_path(out.resolve())
        assert json.loads(journal.read_text())["completed_windows"] == 2

        monkeypatch.setattr(writer_module, "_write_window", real_write_window)
        write(
            out, source(p) + 1,
            window_width=100,
            window_height=100,
            checkpoint_interval=1,
        )
        import rasterio
        with rasterio.open(out) as ds:
            np.testing.assert_array_equal(
                ds.read(1), np.full((300, 300), 2.0, dtype=np.float32),
            )
            assert np.all(ds.read_masks(1) == 255)

    def test_journal_update_failure_preserves_previous_checkpoint(
        self, tmp_path, monkeypatch,
    ):
        p = _write_tiff(tmp_path, "src.tif", np.ones((300, 300), dtype=np.float32))
        out = tmp_path / "out.tif"
        real_write_journal = writer_module._write_journal
        injected = [False]

        def fail_second_checkpoint(path, identity, completed, total):
            if completed == 2 and not injected[0]:
                injected[0] = True
                raise OSError("injected journal update failure")
            return real_write_journal(path, identity, completed, total)

        monkeypatch.setattr(writer_module, "_write_journal", fail_second_checkpoint)
        with pytest.raises(OSError, match="journal update"):
            write(
                out, source(p) + 1,
                window_width=100,
                window_height=100,
                checkpoint_interval=1,
            )
        journal = writer_module._staging_journal_path(out.resolve())
        staging = writer_module._staging_tiff_path(out.resolve())
        assert staging.exists()
        assert json.loads(journal.read_text())["completed_windows"] == 1

        monkeypatch.setattr(writer_module, "_write_journal", real_write_journal)
        executed: list[int] = []
        real_execute = writer_module._execute_window

        def record_execute(plan, cache, idx, x0, y0, width, height):
            executed.append(idx)
            return real_execute(plan, cache, idx, x0, y0, width, height)

        monkeypatch.setattr(writer_module, "_execute_window", record_execute)
        write(
            out, source(p) + 1,
            window_width=100,
            window_height=100,
            checkpoint_interval=1,
        )
        assert executed == list(range(1, 9))

    def test_complete_staging_is_published_over_existing_destination(
        self, tmp_path,
    ):
        p = _write_tiff(tmp_path, "src.tif", np.ones((200, 200), dtype=np.float32))
        out = tmp_path / "out.tif"
        write(out, source(p) + 1)

        def fail_at_completion(completed: int, total: int, idx: int) -> None:
            if completed == total:
                raise RuntimeError("injected final callback failure")

        with pytest.raises(RuntimeError, match="final callback"):
            write(
                out, source(p) + 5,
                overwrite=True,
                window_width=100,
                window_height=100,
                progress_callback=fail_at_completion,
            )
        import rasterio
        with rasterio.open(out) as ds:
            np.testing.assert_array_equal(
                ds.read(1), np.full((200, 200), 2.0, dtype=np.float32),
            )

        write(
            out, source(p) + 5,
            overwrite=True,
            window_width=100,
            window_height=100,
        )
        with rasterio.open(out) as ds:
            np.testing.assert_array_equal(
                ds.read(1), np.full((200, 200), 6.0, dtype=np.float32),
            )

    def test_interrupted_publication_backups_are_rolled_back_and_resumed(
        self, tmp_path,
    ):
        p = _write_tiff(tmp_path, "src.tif", np.ones((200, 200), dtype=np.float32))
        out = tmp_path / "out.tif"
        old_expr = source(p) + 1
        new_expr = source(p) + 5
        write(out, old_expr)

        def fail_at_completion(completed: int, total: int, idx: int) -> None:
            if completed == total:
                raise RuntimeError("leave complete stage")

        with pytest.raises(RuntimeError, match="complete stage"):
            write(
                out, new_expr,
                overwrite=True,
                window_width=100,
                window_height=100,
                progress_callback=fail_at_completion,
            )

        resolved = out.resolve()
        stage = writer_module._staging_tiff_path(resolved)
        stage_manifest = writer_module._staging_manifest_path(resolved)
        manifest = writer_module._manifest_path(resolved)
        backup = stage.with_name(stage.name + ".previous")
        backup_manifest = stage_manifest.with_name(stage_manifest.name + ".previous")
        restart_id = writer_module._build_restart_identity(
            new_expr, np.dtype(np.float32), np.nan,
        )
        writer_module._write_manifest_atomic(stage_manifest, restart_id)

        os_replace = writer_module.os.replace
        os_replace(resolved, backup)
        os_replace(manifest, backup_manifest)
        os_replace(stage, resolved)

        write(
            out, new_expr,
            overwrite=True,
            window_width=100,
            window_height=100,
        )
        import rasterio
        with rasterio.open(out) as ds:
            np.testing.assert_array_equal(
                ds.read(1), np.full((200, 200), 6.0, dtype=np.float32),
            )
        assert not backup.exists()
        assert not backup_manifest.exists()

    def test_integer_nodata_tag_is_published(self, tmp_path):
        p = _write_tiff(
            tmp_path, "src.tif", np.array([[1, 2]], dtype=np.uint8), dtype="uint8",
        )
        out = tmp_path / "out.tif"
        write(out, source(p), invalid_value=0)
        import rasterio
        with rasterio.open(out) as ds:
            assert ds.tags(1)[writer_module.LUNARSCOUT_NODATA_VALUE] == "0"

    @pytest.mark.parametrize("checkpoint_interval", [0, -1, True, 1.5])
    def test_invalid_checkpoint_interval_rejected(self, tmp_path, checkpoint_interval):
        p = _write_tiff(tmp_path, "src.tif", np.ones((2, 2), dtype=np.float32))
        with pytest.raises(Exception) as exc_info:
            write(
                tmp_path / "out.tif",
                source(p),
                checkpoint_interval=checkpoint_interval,
            )
        assert exc_info.value.code == "map_algebra_invalid_checkpoint_interval"
