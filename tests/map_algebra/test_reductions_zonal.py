from __future__ import annotations

import json
import numpy as np
import pytest

from lunarscout.errors import MapAlgebraError
from lunarscout.georeference import GeoReference
from lunarscout.map_algebra import (
    RasterStatistics,
    ZonalStatistics,
    histogram,
    percentile,
    raster as ma_raster,
    statistics,
    unique_counts,
    zonal_raster,
    zonal_stats,
)
from tests.map_algebra.conftest import MOON_WKT, MOON_PROJ4


def _georef(h: int, w: int) -> GeoReference:
    return GeoReference(
        projection_wkt=MOON_WKT, projection_proj4=MOON_PROJ4,
        affine_transform=(1000.0, 20.0, 0.0, 2000.0, 0.0, -20.0),
        width=w, height=h, pixel_size_x=20.0, pixel_size_y=-20.0, nodata=None,
    )


def _make(values, valid=None):
    g = _georef(values.shape[0], values.shape[1])
    if valid is not None:
        return ma_raster(values, g, valid=valid)
    return ma_raster(values, g)


class TestStatistics:
    def test_basic(self):
        r = _make(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
        s = statistics(r)
        assert s.count == 4
        assert s.invalid_count == 0
        assert s.sum == 10.0
        assert s.mean == 2.5
        assert s.min_val == 1.0
        assert s.max_val == 4.0
        assert s.range_val == 3.0
        assert s.std > 0

    def test_with_invalid(self):
        v = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        valid = np.array([[True, False], [True, True]])
        r = _make(v, valid=valid)
        s = statistics(r)
        assert s.count == 3
        assert s.invalid_count == 1

    def test_all_invalid_raises(self):
        r = _make(np.ones((2, 2), dtype=np.float32), valid=np.zeros((2, 2), dtype=np.bool_))
        with pytest.raises(MapAlgebraError):
            statistics(r)

    def test_to_dict(self):
        r = _make(np.ones((2, 2), dtype=np.float32))
        d = statistics(r).to_dict()
        assert d["count"] == 4
        assert "mean" in d

    def test_uint64_extrema_range_and_spread_remain_distinct(self):
        base = 2**63
        r = _make(np.array([[base, base + 2, base + 4]], dtype=np.uint64))
        result = statistics(r)
        assert result.sum == 3 * base + 6
        assert result.min_val == base
        assert result.max_val == base + 4
        assert result.range_val == 4
        assert result.variance > 0.0
        assert result.std > 0.0


class TestHistogram:
    def test_default_bins(self):
        r = _make(np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32))
        counts, edges = histogram(r, bins=3)
        assert len(counts) == 3

    def test_explicit_bins(self):
        r = _make(np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32))
        counts, edges = histogram(r, bins=np.array([0.0, 2.0, 4.0, 6.0]))
        assert np.array_equal(counts, np.array([1, 2, 1], dtype=np.int64))

    def test_empty(self):
        r = _make(np.ones((2, 2), dtype=np.float32), valid=np.zeros((2, 2), dtype=np.bool_))
        counts, edges = histogram(r)
        assert len(counts) == 0

    def test_uint64_uses_explicit_integer_edges_without_precision_loss(self):
        base = 2**63
        r = _make(np.array([[base, base + 2, base + 4]], dtype=np.uint64))
        edges = np.array(
            [base - 1, base + 1, base + 3, base + 5], dtype=np.uint64,
        )
        counts, returned_edges = histogram(r, bins=edges)
        np.testing.assert_array_equal(counts, np.ones(3, dtype=np.int64))
        np.testing.assert_array_equal(returned_edges, edges)

    def test_uint64_rejects_inexact_automatic_edges(self):
        base = 2**63
        r = _make(np.array([[base, base + 2, base + 4]], dtype=np.uint64))
        with pytest.raises(MapAlgebraError) as error:
            histogram(r, bins=3)
        assert error.value.code == "map_algebra_inexact_histogram_edges"


class TestPercentile:
    def test_median(self):
        r = _make(np.array([[1.0, 2.0, 3.0]], dtype=np.float32))
        p = percentile(r, 50)
        assert abs(float(p) - 2.0) < 1e-5

    def test_list(self):
        r = _make(np.arange(1.0, 11.0, dtype=np.float32).reshape(2, 5))
        p = percentile(r, [25, 50, 75])
        assert len(p) == 3

    def test_empty_raises(self):
        r = _make(np.ones((2, 2), dtype=np.float32), valid=np.zeros((2, 2), dtype=np.bool_))
        with pytest.raises(MapAlgebraError):
            percentile(r, 50)

    def test_uint64_observed_percentile_is_exact(self):
        value = 2**63 + 3
        r = _make(np.array([[value - 2, value, value + 2]], dtype=np.uint64))
        result = percentile(r, 50)
        assert isinstance(result, np.uint64)
        assert int(result) == value

    def test_invalid_method_is_structured(self):
        r = _make(np.array([[1.0]], dtype=np.float32))
        with pytest.raises(MapAlgebraError) as error:
            percentile(r, 50, method="bogus")  # type: ignore[arg-type]
        assert error.value.code == "map_algebra_invalid_percentile_method"


class TestUniqueCounts:
    def test_basic(self):
        r = _make(np.array([[1, 1, 2], [2, 3, 3]], dtype=np.int32))
        vals, cnts = unique_counts(r)
        assert list(vals) == [1, 2, 3]
        assert list(cnts) == [2, 2, 2]

    def test_max_unique(self):
        r = _make(np.array([[1, 2, 3, 4, 5, 6]], dtype=np.int32))
        with pytest.raises(MapAlgebraError):
            unique_counts(r, max_unique=5)

    def test_empty(self):
        r = _make(np.ones((2, 2), dtype=np.int32), valid=np.zeros((2, 2), dtype=np.bool_))
        vals, cnts = unique_counts(r)
        assert len(vals) == 0


class TestZonalStats:
    def test_basic(self):
        vals = _make(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
        zones = _make(np.array([[1, 1], [2, 2]], dtype=np.int32))
        zs = zonal_stats(vals, zones, statistics=["mean", "sum"])
        assert list(zs.zone_ids) == [1, 2]
        assert zs.values["mean"][0] == 1.5
        assert zs.values["sum"][1] == 7.0

    def test_invalid_values_excluded_from_mean(self):
        vals = np.array([[1.0, 2.0], [99.0, 4.0]], dtype=np.float32)
        vld = np.ones((2, 2), dtype=np.bool_)
        vld[1, 0] = False
        values_raster = ma_raster(vals, _georef(2, 2), valid=vld)
        zones = _make(np.array([[1, 1], [2, 2]], dtype=np.int32))
        zs = zonal_stats(values_raster, zones, statistics=["count", "valid_count", "mean"])
        assert zs.values["count"][1] == 2
        assert zs.values["valid_count"][1] == 1
        assert zs.values["mean"][1] == 4.0

    def test_boolean_zones(self):
        vals = _make(np.ones((3, 3), dtype=np.float32))
        zones = _make(np.array([[1, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=np.bool_))
        zs = zonal_stats(vals, zones, statistics=["count"])
        assert list(zs.zone_ids) == [0, 1]

    def test_empty_zones(self):
        vals = _make(np.ones((2, 2), dtype=np.float32))
        zones = _make(np.ones((2, 2), dtype=np.int32), valid=np.zeros((2, 2), dtype=np.bool_))
        zs = zonal_stats(vals, zones, statistics=["mean"])
        assert len(zs.zone_ids) == 0

    def test_empty_zones_preserve_declared_integer_statistic_dtypes(self):
        vals = _make(np.ones((2, 2), dtype=np.uint64))
        zones = _make(
            np.ones((2, 2), dtype=np.uint32),
            valid=np.zeros((2, 2), dtype=np.bool_),
        )
        zs = zonal_stats(vals, zones, statistics=["count", "min", "range"])
        assert zs.zone_ids.dtype == np.dtype(np.uint32)
        assert zs.values["count"].dtype == np.dtype(np.int64)
        assert zs.values["min"].dtype == np.dtype(np.uint64)
        assert zs.values["range"].dtype == np.dtype(np.uint64)

    def test_default_stats(self):
        vals = _make(np.array([[1.0, 2.0]], dtype=np.float32))
        zones = _make(np.array([[1, 1]], dtype=np.int32))
        zs = zonal_stats(vals, zones)
        assert len(zs.columns) >= 10

    def test_to_json(self):
        vals = _make(np.array([[1.0]], dtype=np.float32))
        zones = _make(np.array([[1]], dtype=np.int32))
        zs = zonal_stats(vals, zones, statistics=["mean"])
        j = json.loads(zs.to_json())
        assert j["zone_ids"] == [1]

    def test_to_records(self):
        vals = _make(np.array([[1.0, 2.0]], dtype=np.float32))
        zones = _make(np.array([[1, 2]], dtype=np.int32))
        zs = zonal_stats(vals, zones, statistics=["mean"])
        recs = zs.to_records()
        assert len(recs) == 2
        assert recs[0]["zone"] == 1
        with pytest.raises(TypeError):
            recs[0]["zone"] = 99

    def test_zone_nodata(self):
        vals = _make(np.array([[1.0, 2.0, 3.0]], dtype=np.float32))
        zones = _make(np.array([[1, -9999, 1]], dtype=np.int32))
        zs = zonal_stats(vals, zones, statistics=["mean"], zone_nodata=-9999)
        assert list(zs.zone_ids) == [1]

    def test_include_zone_ids_uses_zone_dtype(self):
        vals = _make(np.array([[1.0]], dtype=np.float32))
        zones = _make(np.array([[1]], dtype=np.uint64))
        zs = zonal_stats(vals, zones, statistics=["mean"], include_zone_ids=[1, 2])
        assert list(zs.zone_ids) == [1, 2]

    def test_write_csv(self, tmp_path):
        vals = _make(np.array([[1.0, 2.0]], dtype=np.float32))
        zones = _make(np.array([[1, 2]], dtype=np.int32))
        zs = zonal_stats(vals, zones, statistics=["mean"])
        path = tmp_path / "out.csv"
        zs.write_csv(str(path))
        assert path.exists()
        content = path.read_text()
        assert "zone" in content

    def test_float_zones_rejected(self):
        vals = _make(np.ones((2, 2), dtype=np.float32))
        zones = _make(np.ones((2, 2), dtype=np.float32))
        with pytest.raises(MapAlgebraError):
            zonal_stats(vals, zones)

    def test_unknown_statistic_raises(self):
        vals = _make(np.ones((2, 2), dtype=np.float32))
        zones = _make(np.ones((2, 2), dtype=np.int32))
        with pytest.raises(MapAlgebraError, match="Unknown zonal statistic"):
            zonal_stats(vals, zones, statistics=["bogus"])  # type: ignore[list-item]

    def test_zonal_percentile(self):
        vals = _make(np.array([[1.0, 2.0, 3.0, 4.0, 5.0]], dtype=np.float32))
        zones = _make(np.array([[1, 1, 1, 1, 1]], dtype=np.int32))
        zs = zonal_stats(vals, zones, statistics=["median", "p25", "p75", "p90"])
        assert zs.values["median"][0] == 3.0
        assert zs.values["p25"][0] == 2.0
        assert zs.values["p75"][0] == 4.0

    def test_include_zone_ids(self):
        vals = _make(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
        zones = _make(np.array([[1, 1], [2, 2]], dtype=np.int32))
        zs = zonal_stats(vals, zones, statistics=["mean"], include_zone_ids=[3, 4])
        assert list(zs.zone_ids) == [1, 2, 3, 4]
        assert not zs.valid["mean"][2]
        assert not zs.valid["mean"][3]

    def test_uint64_zone_ids(self):
        vals = _make(np.array([[1.0, 2.0]], dtype=np.float32))
        large = np.uint64(2**63 + 5)
        zones = _make(np.array([[1, large]], dtype=np.uint64))
        zs = zonal_stats(vals, zones, statistics=["mean"])
        assert list(zs.zone_ids) == [1, large]
        j = json.loads(zs.to_json())
        assert j["zone_ids"] == [1, int(large)]

    def test_zonal_invalid_values_excluded_from_mean(self):
        vals = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        valid = np.ones((2, 2), dtype=np.bool_)
        valid[0, 0] = False
        values_raster = ma_raster(vals, _georef(2, 2), valid=valid)
        zones = _make(np.array([[1, 1], [1, 1]], dtype=np.int32))
        zs = zonal_stats(values_raster, zones, statistics=["count", "valid_count", "invalid_count", "mean"])
        assert zs.values["count"][0] == 4
        assert zs.values["valid_count"][0] == 3
        assert zs.values["invalid_count"][0] == 1
        assert zs.values["mean"][0] == 3.0

    def test_count_is_int(self):
        vals = _make(np.ones((2, 2), dtype=np.float32))
        zones = _make(np.ones((2, 2), dtype=np.int32))
        zs = zonal_stats(vals, zones, statistics=["count"])
        assert zs.values["count"].dtype == np.int64

    def test_count_units_are_dimensionless(self):
        vals = _make(np.ones((2, 2), dtype=np.float32)).with_units("metres")
        zones = _make(np.ones((2, 2), dtype=np.int32))
        zs = zonal_stats(vals, zones, statistics=["count", "mean"])
        assert zs.units == {"count": None, "mean": "metres"}

    def test_uint64_value_extrema_and_spread_remain_distinct(self):
        base = 2**63
        vals = _make(np.array([[base, base + 2, base + 4]], dtype=np.uint64))
        zones = _make(np.ones((1, 3), dtype=np.uint8))
        zs = zonal_stats(
            vals,
            zones,
            statistics=["sum", "min", "max", "range", "variance", "std"],
        )
        assert zs.values["sum"].dtype == np.dtype(np.uint64)
        assert int(zs.values["sum"][0]) == (3 * base + 6) % 2**64
        assert int(zs.values["min"][0]) == base
        assert int(zs.values["max"][0]) == base + 4
        assert int(zs.values["range"][0]) == 4
        assert zs.values["variance"][0] > 0.0
        assert zs.values["std"][0] > 0.0
        record = zs.to_records()[0]
        assert record["min"] == base
        assert record["max"] == base + 4


class TestZonalRaster:
    def test_basic(self):
        vals = _make(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
        zones = _make(np.array([[1, 1], [2, 2]], dtype=np.int32))
        result = zonal_raster(vals, zones, statistic="mean")
        assert result.values[0, 0] == 1.5
        assert result.values[1, 1] == 3.5

    def test_integer_statistic_preserves_dtype_and_units(self):
        base = 2**63
        vals = _make(np.array([[base, base + 2]], dtype=np.uint64)).with_units("counts")
        zones = _make(np.ones((1, 2), dtype=np.uint8))
        result = zonal_raster(vals, zones, statistic="min")
        assert result.dtype == np.dtype(np.uint64)
        assert int(result.values[0, 0]) == base
        assert result.units == "counts"
