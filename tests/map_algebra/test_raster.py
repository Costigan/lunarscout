from __future__ import annotations

import sys

import numpy as np
import pytest
import rasterio
from affine import Affine

from lunarscout.errors import RasterValidationError
from lunarscout.georeference import GeoReference
from lunarscout.map_algebra import (
    from_existing,
    from_masked_array,
    raster,
    read,
    to_existing,
)
from lunarscout.raster import Raster


def assert_raster_equal(
    left: Raster,
    right: Raster,
) -> None:
    """Test assertion helper with detailed mismatch reporting."""
    if not left.same_grid(right):
        raise AssertionError(
            f"Raster grids differ:\n"
            f"  left: {left.georef}\n"
            f"  right: {right.georef}"
        )
    valid_equal = bool(np.array_equal(left.valid, right.valid))
    if not valid_equal:
        mismatch = int(np.sum(left.valid != right.valid))
        raise AssertionError(
            f"Validity masks differ at {mismatch} pixel(s)."
        )
    valid = left.valid
    values_equal = bool(np.array_equal(left.values[valid], right.values[valid]))
    if not values_equal:
        diff = np.where(valid, left.values != right.values, False)
        mismatch_count = int(np.sum(diff))
        max_diff = float(np.max(np.abs((left.values - right.values).astype(np.float64)[diff])))
        raise AssertionError(
            f"Raster values differ at {mismatch_count} valid pixel(s); "
            f"max absolute difference = {max_diff:.6g}."
        )


# ---------------------------------------------------------------------------
# Construction and validation
# ---------------------------------------------------------------------------


class TestRasterConstruction:
    def test_basic_construction(self, float_raster_values, north_up_georef, all_valid_mask):
        r = raster(float_raster_values, north_up_georef, valid=all_valid_mask)
        assert r.values is float_raster_values
        assert r.georef is north_up_georef
        assert r.valid is all_valid_mask
        assert r.shape == (north_up_georef.height, north_up_georef.width)

    def test_rejects_3d_array(self, north_up_georef):
        values = np.zeros((2, 3, 4), dtype=np.float32)
        with pytest.raises(RasterValidationError, match="two-dimensional"):
            Raster(values=values, georef=north_up_georef, valid=np.ones((2, 3), dtype=np.bool_))

    def test_rejects_object_dtype(self, north_up_georef):
        values = np.array([["a", "b"], ["c", "d"]], dtype=object)
        with pytest.raises(RasterValidationError, match="supported numeric or boolean"):
            Raster(values=values, georef=north_up_georef, valid=np.ones((2, 2), dtype=np.bool_))

    def test_rejects_complex_dtype(self, north_up_georef):
        values = np.zeros((2, 2), dtype=np.complex64)
        with pytest.raises(RasterValidationError, match="supported numeric or boolean"):
            Raster(values=values, georef=north_up_georef, valid=np.ones((2, 2), dtype=np.bool_))

    def test_rejects_string_dtype(self, north_up_georef):
        values = np.zeros((2, 2), dtype="U10")
        with pytest.raises(RasterValidationError, match="supported numeric or boolean"):
            Raster(values=values, georef=north_up_georef, valid=np.ones((2, 2), dtype=np.bool_))

    def test_rejects_shape_mismatch_values(self, north_up_georef, all_valid_mask):
        values = np.zeros((3, 3), dtype=np.float32)
        assert north_up_georef.height == 8 and north_up_georef.width == 10
        with pytest.raises(RasterValidationError, match="shape does not match"):
            Raster(values=values, georef=north_up_georef, valid=all_valid_mask)

    def test_rejects_shape_mismatch_valid(self, float_raster_values, north_up_georef):
        valid = np.ones((3, 3), dtype=np.bool_)
        with pytest.raises(RasterValidationError, match="shape does not match"):
            Raster(values=float_raster_values, georef=north_up_georef, valid=valid)

    def test_rejects_non_bool_valid_dtype(self, float_raster_values, north_up_georef):
        valid = np.ones((north_up_georef.height, north_up_georef.width), dtype=np.uint8)
        with pytest.raises(RasterValidationError, match="bool dtype"):
            Raster(values=float_raster_values, georef=north_up_georef, valid=valid)

    def test_supports_boolean_dtype(self, north_up_georef):
        values = np.ones((north_up_georef.height, north_up_georef.width), dtype=np.bool_)
        valid = np.ones((north_up_georef.height, north_up_georef.width), dtype=np.bool_)
        r = Raster(values=values, georef=north_up_georef, valid=valid)
        assert r.dtype == np.dtype(np.bool_)

    def test_supports_all_integer_dtypes(self, north_up_georef):
        for dtype_name in ("uint8", "int8", "uint16", "int16", "uint32", "int32", "uint64", "int64"):
            values = np.ones((2, 2), dtype=dtype_name)
            georef = _make_2x2_georef(north_up_georef)
            r = Raster(values=values, georef=georef, valid=np.ones((2, 2), dtype=np.bool_))
            assert r.dtype == np.dtype(dtype_name)

    def test_supports_float_dtypes(self, north_up_georef):
        for dtype_name in ("float32", "float64"):
            values = np.ones((2, 2), dtype=dtype_name)
            georef = _make_2x2_georef(north_up_georef)
            r = Raster(values=values, georef=georef, valid=np.ones((2, 2), dtype=np.bool_))
            assert r.dtype == np.dtype(dtype_name)

    def test_direct_construction_without_factory(self, float_raster_values, north_up_georef, all_valid_mask):
        r = Raster(values=float_raster_values, georef=north_up_georef, valid=all_valid_mask)
        assert r.values is float_raster_values
        assert r.valid is all_valid_mask

    def test_no_implicit_copy_on_construction(self, north_up_georef, all_valid_mask):
        values = np.ones((north_up_georef.height, north_up_georef.width), dtype=np.float32)
        r = Raster(values=values, georef=north_up_georef, valid=all_valid_mask)
        r.values[0, 0] = 99.0
        assert values[0, 0] == 99.0


class TestRasterBoolUnavailable:
    def test_bool_raises_typeerror(self, north_up_georef):
        georef = _make_2x2_georef(north_up_georef)
        r = raster(np.ones((2, 2), dtype=np.float32), georef, valid=np.ones((2, 2), dtype=np.bool_))
        with pytest.raises(TypeError, match="implicit truth testing"):
            bool(r)

    def test_if_raster_raises_typeerror(self, north_up_georef):
        georef = _make_2x2_georef(north_up_georef)
        r = raster(np.ones((2, 2), dtype=np.float32), georef, valid=np.ones((2, 2), dtype=np.bool_))
        with pytest.raises(TypeError, match="implicit truth testing"):
            if r:
                pass

    def test_all_false_raster_raises_typeerror(self, north_up_georef):
        georef = _make_2x2_georef(north_up_georef)
        r = raster(np.zeros((2, 2), dtype=np.bool_), georef, valid=np.ones((2, 2), dtype=np.bool_))
        with pytest.raises(TypeError, match="implicit truth testing"):
            bool(r)


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


class TestRasterProperties:
    def test_shape(self, float_raster_values, north_up_georef, all_valid_mask):
        r = raster(float_raster_values, north_up_georef, valid=all_valid_mask)
        assert r.shape == (8, 10)
        assert r.height == 8
        assert r.width == 10

    def test_dtype(self, float_raster_values, north_up_georef, all_valid_mask):
        r = raster(float_raster_values, north_up_georef, valid=all_valid_mask)
        assert r.dtype == np.dtype(np.float32)

    def test_nbytes(self, float_raster_values, north_up_georef, all_valid_mask):
        r = raster(float_raster_values, north_up_georef, valid=all_valid_mask)
        expected = float_raster_values.nbytes + all_valid_mask.nbytes
        assert r.nbytes == expected

    def test_all_valid(self, float_raster_values, north_up_georef, all_valid_mask):
        r = raster(float_raster_values, north_up_georef, valid=all_valid_mask)
        assert r.all_valid is True

    def test_not_all_valid(self, float_raster_values, north_up_georef):
        valid = np.zeros((8, 10), dtype=np.bool_)
        valid[2:6, 3:7] = True
        r = raster(float_raster_values, north_up_georef, valid=valid)
        assert r.all_valid is False

    def test_invalid_count(self, float_raster_values, north_up_georef):
        valid = np.zeros((8, 10), dtype=np.bool_)
        valid[0, 0] = True
        valid[1, 1] = True
        r = raster(float_raster_values, north_up_georef, valid=valid)
        assert r.invalid_count == 80 - 2

    def test_invalid_count_all_valid(self, float_raster_values, north_up_georef, all_valid_mask):
        r = raster(float_raster_values, north_up_georef, valid=all_valid_mask)
        assert r.invalid_count == 0

    def test_units_default_none(self, float_raster_values, north_up_georef, all_valid_mask):
        r = raster(float_raster_values, north_up_georef, valid=all_valid_mask)
        assert r.units is None

    def test_name_default_none(self, float_raster_values, north_up_georef, all_valid_mask):
        r = raster(float_raster_values, north_up_georef, valid=all_valid_mask)
        assert r.name is None


# ---------------------------------------------------------------------------
# Named construction
# ---------------------------------------------------------------------------


class TestRasterNamedConstruction:
    def test_with_units(self, float_raster_values, north_up_georef, all_valid_mask):
        r = raster(float_raster_values, north_up_georef, valid=all_valid_mask, units="meters")
        assert r.units == "meters"

    def test_with_name(self, float_raster_values, north_up_georef, all_valid_mask):
        r = raster(float_raster_values, north_up_georef, valid=all_valid_mask, name="elevation")
        assert r.name == "elevation"


# ---------------------------------------------------------------------------
# Copy / readonly
# ---------------------------------------------------------------------------


class TestRasterCopyReadonly:
    def test_copy_returns_new_array(self, float_raster_values, north_up_georef, all_valid_mask):
        r = raster(float_raster_values, north_up_georef, valid=all_valid_mask)
        c = r.copy()
        assert c.values is not r.values
        assert c.valid is not r.valid
        np.testing.assert_array_equal(c.values, r.values)
        np.testing.assert_array_equal(c.valid, r.valid)

    def test_readonly_values_not_writeable(self, float_raster_values, north_up_georef, all_valid_mask):
        r = raster(float_raster_values, north_up_georef, valid=all_valid_mask)
        ro = r.readonly()
        assert not ro.values.flags.writeable
        assert not ro.valid.flags.writeable

    def test_copy_preserves_metadata(self, float_raster_values, north_up_georef, all_valid_mask):
        r = raster(float_raster_values, north_up_georef, valid=all_valid_mask, units="meters", name="test")
        c = r.copy()
        assert c.georef is r.georef
        assert c.units == "meters"
        assert c.name == "test"

    def test_copy_preserves_provenance(self, north_up_georef):
        georef = _make_2x2_georef(north_up_georef)
        r = raster(
            np.ones((2, 2), dtype=np.float32),
            georef,
            valid=np.ones((2, 2), dtype=np.bool_),
            validity_provenance="explicit-caller",
        )
        c = r.copy()
        assert c.validity_provenance == "explicit-caller"


# ---------------------------------------------------------------------------
# Filled / masked
# ---------------------------------------------------------------------------


class TestRasterFilledMasked:
    def test_filled_replaces_invalid_with_fill(self, north_up_georef):
        values = np.ones((3, 3), dtype=np.float32) * 100.0
        valid = np.ones((3, 3), dtype=np.bool_)
        valid[1, 1] = False
        georef = _make_3x3_georef(north_up_georef)
        r = raster(values, georef, valid=valid)
        filled = r.filled(-1.0)
        assert filled[1, 1] == -1.0
        assert filled[0, 0] == 100.0
        assert filled.dtype == np.dtype(np.float32)

    def test_masked_returns_masked_array(self, north_up_georef):
        values = np.ones((3, 3), dtype=np.float32) * 100.0
        valid = np.ones((3, 3), dtype=np.bool_)
        valid[1, 1] = False
        georef = _make_3x3_georef(north_up_georef)
        r = raster(values, georef, valid=valid)
        ma = r.masked()
        assert isinstance(ma, np.ma.MaskedArray)
        assert bool(ma.mask[1, 1]) is True
        assert bool(ma.mask[0, 0]) is False

    @pytest.mark.parametrize("fill", [-1, 256, 1.5, True, "5"])
    def test_filled_rejects_lossy_uint8_fill_without_mutation(
        self, north_up_georef, fill,
    ):
        values = np.array([[7, 99], [8, 9]], dtype=np.uint8)
        valid = np.array([[True, False], [True, True]], dtype=np.bool_)
        raster_value = raster(
            values, _make_2x2_georef(north_up_georef), valid=valid,
        )
        before = values.copy()
        with pytest.raises(RasterValidationError) as error:
            raster_value.filled(fill)
        assert error.value.code == "raster_unrepresentable_nodata"
        np.testing.assert_array_equal(values, before)

    def test_filled_preserves_exact_uint64_and_never_mutates_source(
        self, north_up_georef,
    ):
        fill = 2**63 + 123
        values = np.array([[7, 99], [8, 9]], dtype=np.uint64)
        valid = np.array([[True, False], [True, True]], dtype=np.bool_)
        raster_value = raster(
            values, _make_2x2_georef(north_up_georef), valid=valid,
        )
        filled = raster_value.filled(fill)
        assert int(filled[0, 1]) == fill
        assert int(values[0, 1]) == 99


# ---------------------------------------------------------------------------
# Metadata helpers
# ---------------------------------------------------------------------------


class TestRasterMetadataHelpers:
    def test_with_name(self, float_raster_values, north_up_georef, all_valid_mask):
        r = raster(float_raster_values, north_up_georef, valid=all_valid_mask, name="old")
        r2 = r.with_name("new")
        assert r2.name == "new"
        assert r.name == "old"
        assert r2.values is r.values

    def test_with_units(self, float_raster_values, north_up_georef, all_valid_mask):
        r = raster(float_raster_values, north_up_georef, valid=all_valid_mask, units="old")
        r2 = r.with_units("new")
        assert r2.units == "new"
        assert r.units == "old"

    def test_with_validity(self, float_raster_values, north_up_georef, all_valid_mask):
        r = raster(float_raster_values, north_up_georef, valid=all_valid_mask)
        new_valid = np.zeros_like(all_valid_mask)
        new_valid[0:2, 0:2] = True
        r2 = r.with_validity(new_valid)
        assert r2.valid is new_valid
        assert r2.values is r.values

    def test_with_validity_rejects_wrong_shape(self, float_raster_values, north_up_georef, all_valid_mask):
        r = raster(float_raster_values, north_up_georef, valid=all_valid_mask)
        wrong_valid = np.ones((3, 3), dtype=np.bool_)
        with pytest.raises(RasterValidationError, match="shape"):
            r.with_validity(wrong_valid)

    def test_with_validity_rejects_wrong_dtype(self, float_raster_values, north_up_georef, all_valid_mask):
        r = raster(float_raster_values, north_up_georef, valid=all_valid_mask)
        wrong_valid = np.ones(all_valid_mask.shape, dtype=np.uint8)
        with pytest.raises(RasterValidationError, match="bool dtype"):
            r.with_validity(wrong_valid)

    def test_with_georef(self, float_raster_values, north_up_georef, all_valid_mask):
        r = raster(float_raster_values, north_up_georef, valid=all_valid_mask)
        r2 = r.with_georef(north_up_georef)
        assert r2.georef is north_up_georef

    def test_with_georef_rejects_shape_mismatch(self, float_raster_values, north_up_georef, all_valid_mask, small_georef):
        r = raster(float_raster_values, north_up_georef, valid=all_valid_mask)
        with pytest.raises(RasterValidationError, match="shape"):
            r.with_georef(small_georef)


# ---------------------------------------------------------------------------
# Grid comparison
# ---------------------------------------------------------------------------


class TestRasterGridComparison:
    def test_same_grid_true(self, float_raster_values, north_up_georef, all_valid_mask):
        r1 = raster(float_raster_values, north_up_georef, valid=all_valid_mask)
        r2 = raster(float_raster_values.copy(), north_up_georef, valid=all_valid_mask.copy())
        assert r1.same_grid(r2) is True

    def test_same_grid_false_shifted(self, float_raster_values, north_up_georef, shifted_georef, all_valid_mask):
        r1 = raster(float_raster_values, north_up_georef, valid=all_valid_mask)
        r2 = raster(float_raster_values, shifted_georef, valid=all_valid_mask.copy())
        assert r1.same_grid(r2) is False

    def test_same_grid_false_rotated(self, float_raster_values, north_up_georef, rotated_georef, all_valid_mask):
        r1 = raster(float_raster_values, north_up_georef, valid=all_valid_mask)
        r2_values = np.zeros((rotated_georef.height, rotated_georef.width), dtype=np.float32)
        r2_valid = np.ones((rotated_georef.height, rotated_georef.width), dtype=np.bool_)
        r2 = raster(r2_values, rotated_georef, valid=r2_valid)
        assert r1.same_grid(r2) is False

    def test_same_grid_false_differing_crs(self, north_up_georef, differing_crs_georef):
        r1 = raster(np.ones((8, 10), dtype=np.float32), north_up_georef, valid=np.ones((8, 10), dtype=np.bool_))
        r2 = raster(np.ones((8, 10), dtype=np.float32), differing_crs_georef, valid=np.ones((8, 10), dtype=np.bool_))
        assert r1.same_grid(r2) is False

    def test_same_metadata_true(self, float_raster_values, north_up_georef, all_valid_mask):
        r1 = raster(float_raster_values, north_up_georef, valid=all_valid_mask, units="m", name="a")
        r2 = raster(float_raster_values.copy(), north_up_georef, valid=all_valid_mask.copy(), units="m", name="a")
        assert r1.same_metadata(r2) is True

    def test_same_metadata_false_different_units(self, float_raster_values, north_up_georef, all_valid_mask):
        r1 = raster(float_raster_values, north_up_georef, valid=all_valid_mask, units="m", name="a")
        r2 = raster(float_raster_values, north_up_georef, valid=all_valid_mask, units="deg", name="a")
        assert r1.same_metadata(r2) is False

    def test_same_metadata_false_different_name(self, float_raster_values, north_up_georef, all_valid_mask):
        r1 = raster(float_raster_values, north_up_georef, valid=all_valid_mask, name="a")
        r2 = raster(float_raster_values, north_up_georef, valid=all_valid_mask, name="b")
        assert r1.same_metadata(r2) is False


# ---------------------------------------------------------------------------
# array_equal
# ---------------------------------------------------------------------------


class TestRasterArrayEqual:
    def test_array_equal_true(self, north_up_georef):
        values = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        valid = np.ones((2, 2), dtype=np.bool_)
        georef = _make_2x2_georef(north_up_georef)
        r1 = raster(values, georef, valid=valid)
        r2 = raster(values.copy(), georef, valid=valid.copy())
        assert r1.array_equal(r2) is True

    def test_array_equal_false_values(self, north_up_georef):
        georef = _make_2x2_georef(north_up_georef)
        r1 = raster(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32), georef, valid=np.ones((2, 2), dtype=np.bool_))
        r2 = raster(np.array([[5.0, 2.0], [3.0, 4.0]], dtype=np.float32), georef, valid=np.ones((2, 2), dtype=np.bool_))
        assert r1.array_equal(r2) is False

    def test_array_equal_false_validity(self, north_up_georef):
        georef = _make_2x2_georef(north_up_georef)
        valid1 = np.ones((2, 2), dtype=np.bool_)
        valid2 = np.ones((2, 2), dtype=np.bool_)
        valid2[0, 0] = False
        r1 = raster(np.ones((2, 2), dtype=np.float32), georef, valid=valid1)
        r2 = raster(np.ones((2, 2), dtype=np.float32), georef, valid=valid2)
        assert r1.array_equal(r2) is False

    def test_array_equal_invalid_payload_ignored(self, north_up_georef):
        georef = _make_2x2_georef(north_up_georef)
        valid = np.ones((2, 2), dtype=np.bool_)
        valid[0, 0] = False
        v1 = np.array([[99.0, 1.0], [1.0, 1.0]], dtype=np.float32)
        v2 = np.array([[77.0, 1.0], [1.0, 1.0]], dtype=np.float32)
        r1 = raster(v1, georef, valid=valid)
        r2 = raster(v2, georef, valid=valid)
        assert r1.array_equal(r2) is True

    def test_array_equal_equal_invalid_payload(self, north_up_georef):
        georef = _make_2x2_georef(north_up_georef)
        valid = np.ones((2, 2), dtype=np.bool_)
        valid[0, 0] = False
        v1 = np.array([[99.0, 1.0], [1.0, 1.0]], dtype=np.float32)
        v2 = np.array([[77.0, 1.0], [1.0, 1.0]], dtype=np.float32)
        r1 = raster(v1, georef, valid=valid)
        r2 = raster(v2, georef, valid=valid)
        assert r1.array_equal(r2, equal_invalid_payload=True) is False


# ---------------------------------------------------------------------------
# allclose
# ---------------------------------------------------------------------------


class TestRasterAllclose:
    def test_allclose_true(self, north_up_georef):
        georef = _make_2x2_georef(north_up_georef)
        r1 = raster(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32), georef, valid=np.ones((2, 2), dtype=np.bool_))
        r2 = raster(np.array([[1.000001, 2.0], [3.0, 4.0]], dtype=np.float32), georef, valid=np.ones((2, 2), dtype=np.bool_))
        assert r1.allclose(r2) is True

    def test_allclose_false(self, north_up_georef):
        georef = _make_2x2_georef(north_up_georef)
        r1 = raster(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32), georef, valid=np.ones((2, 2), dtype=np.bool_))
        r2 = raster(np.array([[100.0, 2.0], [3.0, 4.0]], dtype=np.float32), georef, valid=np.ones((2, 2), dtype=np.bool_))
        assert r1.allclose(r2) is False

    def test_allclose_equal_nan(self, north_up_georef):
        georef = _make_2x2_georef(north_up_georef)
        v1 = np.array([[np.nan, 2.0], [3.0, 4.0]], dtype=np.float32)
        v2 = np.array([[np.nan, 2.0], [3.0, 4.0]], dtype=np.float32)
        r1 = raster(v1, georef, valid=np.ones((2, 2), dtype=np.bool_))
        r2 = raster(v2, georef, valid=np.ones((2, 2), dtype=np.bool_))
        assert r1.allclose(r2, equal_nan=True) is True
        assert r1.allclose(r2, equal_nan=False) is False


# ---------------------------------------------------------------------------
# Equality operators
# ---------------------------------------------------------------------------


class TestRasterEqualityOperators:
    def test_eq_returns_raster(self, north_up_georef):
        georef = _make_2x2_georef(north_up_georef)
        r1 = raster(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32), georef, valid=np.ones((2, 2), dtype=np.bool_))
        r2 = raster(np.array([[1.0, 2.0], [3.0, 5.0]], dtype=np.float32), georef, valid=np.ones((2, 2), dtype=np.bool_))
        result = r1 == r2
        assert isinstance(result, Raster)
        assert result.dtype == np.dtype(np.bool_)
        np.testing.assert_array_equal(result.values, np.array([[True, True], [True, False]]))

    def test_ne_returns_raster(self, north_up_georef):
        georef = _make_2x2_georef(north_up_georef)
        r1 = raster(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32), georef, valid=np.ones((2, 2), dtype=np.bool_))
        r2 = raster(np.array([[1.0, 2.0], [3.0, 5.0]], dtype=np.float32), georef, valid=np.ones((2, 2), dtype=np.bool_))
        result = r1 != r2
        assert isinstance(result, Raster)
        np.testing.assert_array_equal(result.values, np.array([[False, False], [False, True]]))

    def test_eq_invalid_pixels_are_false(self, north_up_georef):
        georef = _make_2x2_georef(north_up_georef)
        valid = np.ones((2, 2), dtype=np.bool_)
        valid[0, 0] = False
        r1 = raster(np.array([[99.0, 2.0], [3.0, 4.0]], dtype=np.float32), georef, valid=valid)
        r2 = raster(np.array([[99.0, 2.0], [3.0, 4.0]], dtype=np.float32), georef, valid=valid)
        result = r1 == r2
        np.testing.assert_array_equal(result.values, np.array([[False, True], [True, True]]))
        assert not result.valid[0, 0]

    def test_unhashable(self, north_up_georef):
        georef = _make_2x2_georef(north_up_georef)
        r = raster(np.ones((2, 2), dtype=np.float32), georef, valid=np.ones((2, 2), dtype=np.bool_))
        with pytest.raises(TypeError):
            hash(r)

    def test_eq_with_non_raster(self, north_up_georef):
        georef = _make_2x2_georef(north_up_georef)
        r = raster(np.ones((2, 2), dtype=np.float32), georef, valid=np.ones((2, 2), dtype=np.bool_))
        result = r == "not a raster"
        assert result is False


# ---------------------------------------------------------------------------
# Validity from nodata
# ---------------------------------------------------------------------------


class TestValidFromNodata:
    def test_nodata_none_all_valid(self, north_up_georef):
        values = np.array([[1.0, -9999.0], [3.0, 4.0]], dtype=np.float32)
        georef = _make_2x2_georef(north_up_georef)
        r = raster(values, georef, nodata=None)
        assert r.all_valid is True

    def test_nodata_auto_uses_georef(self, north_up_georef):
        values = np.array([[1.0, -9999.0], [3.0, 4.0]], dtype=np.float32)
        georef = _make_2x2_georef(north_up_georef, nodata=-9999.0)
        r = raster(values, georef, nodata="auto")
        assert not r.valid[0, 1]
        assert r.valid[0, 0]

    def test_nodata_explicit(self, north_up_georef):
        values = np.array([[1.0, -9999.0], [3.0, 4.0]], dtype=np.float32)
        georef = _make_2x2_georef(north_up_georef)
        r = raster(values, georef, nodata=-9999.0)
        assert not r.valid[0, 1]

    def test_nan_nodata(self, north_up_georef):
        values = np.array([[1.0, np.nan], [3.0, 4.0]], dtype=np.float32)
        georef = _make_2x2_georef(north_up_georef, nodata=np.nan)
        r = raster(values, georef, nodata="auto")
        assert not r.valid[0, 1]
        assert r.valid[0, 0]

    @pytest.mark.parametrize("nodata", [np.inf, -np.inf, 0.1])
    def test_float32_rejects_non_exact_or_infinite_nodata(
        self, north_up_georef, nodata,
    ):
        values = np.ones((2, 2), dtype=np.float32)
        with pytest.raises(RasterValidationError) as error:
            raster(values, _make_2x2_georef(north_up_georef), nodata=nodata)
        assert error.value.code == "raster_unrepresentable_nodata"

    def test_uint64_nodata_beyond_float_exact_range_is_exact(
        self, north_up_georef,
    ):
        nodata = 2**63 + 123
        values = np.array([[nodata, nodata + 1], [0, 1]], dtype=np.uint64)
        result = raster(
            values, _make_2x2_georef(north_up_georef), nodata=nodata,
        )
        np.testing.assert_array_equal(
            result.valid,
            np.array([[False, True], [True, True]], dtype=np.bool_),
        )

    def test_explicit_valid_overrides_nodata(self, north_up_georef):
        values = np.array([[1.0, -9999.0], [3.0, 4.0]], dtype=np.float32)
        georef = _make_2x2_georef(north_up_georef, nodata=-9999.0)
        explicit = np.ones((2, 2), dtype=np.bool_)
        explicit[0, 0] = False
        r = raster(values, georef, valid=explicit)
        assert not r.valid[0, 0]
        assert r.valid[0, 1]

    def test_integer_nodata_with_float_value_rejected(self, north_up_georef):
        georef = _make_2x2_georef(north_up_georef)
        values = np.ones((2, 2), dtype=np.uint8)
        with pytest.raises(RasterValidationError, match="integer"):
            raster(values, georef, nodata=1.5)

    def test_integer_nodata_out_of_range_rejected(self, north_up_georef):
        georef = _make_2x2_georef(north_up_georef)
        values = np.ones((2, 2), dtype=np.uint8)
        with pytest.raises(RasterValidationError, match="out of range"):
            raster(values, georef, nodata=300)


# ---------------------------------------------------------------------------
# Masked-array handling
# ---------------------------------------------------------------------------


class TestMaskedArrayPreservation:
    def test_masked_array_mask_preserved(self, north_up_georef, masked_float_raster):
        georef = north_up_georef
        r = raster(masked_float_raster, georef, nodata=None)
        assert not r.valid[0, 1]
        assert not r.valid[1, 2]
        assert r.values[0, 0] == float(masked_float_raster.data[0, 0])

    def test_masked_array_combined_with_nodata(self, masked_float_raster):
        georef = _make_2x2_georef(_georef_8x10(), nodata=-9999.0)
        small = masked_float_raster[:2, :2].copy()
        r = raster(small, georef, nodata="auto")
        assert not r.valid[0, 0]
        if small._mask[0, 1]:
            assert not r.valid[0, 1]

    def test_masked_array_with_explicit_valid_raises(self, north_up_georef, masked_float_raster):
        with pytest.raises(RasterValidationError, match="Cannot supply both"):
            raster(masked_float_raster, north_up_georef, valid=np.ones((8, 10), dtype=np.bool_))

    def test_masked_array_provenance(self, north_up_georef):
        ma = np.ma.array(np.ones((2, 2), dtype=np.float32), mask=[[False, True], [False, False]])
        georef = _make_2x2_georef(north_up_georef)
        r = raster(ma, georef, nodata=None)
        assert r.validity_provenance == "masked-array+nodata"

    def test_masked_array_nodata_is_validated_before_comparison(
        self, north_up_georef,
    ):
        values = np.ma.array(
            np.array([[1, 2], [3, 4]], dtype=np.uint8),
            mask=np.array([[False, True], [False, False]], dtype=np.bool_),
        )
        with pytest.raises(RasterValidationError) as error:
            raster(values, _make_2x2_georef(north_up_georef), nodata=300)
        assert error.value.code == "raster_unrepresentable_nodata"


# ---------------------------------------------------------------------------
# from_masked_array
# ---------------------------------------------------------------------------


class TestFromMaskedArray:
    def test_basic(self, north_up_georef):
        data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        mask = np.array([[False, True], [False, False]])
        ma = np.ma.array(data, mask=mask)
        georef = _make_2x2_georef(north_up_georef)
        r = from_masked_array(ma, georef)
        assert not r.valid[0, 1]
        assert r.valid[0, 0]
        assert r.values[0, 0] == 1.0
        assert r.validity_provenance == "masked-array"

    def test_no_mask_all_valid(self, north_up_georef):
        ma = np.ma.array(np.ones((2, 2), dtype=np.float32))
        georef = _make_2x2_georef(north_up_georef)
        r = from_masked_array(ma, georef)
        assert r.all_valid is True

    def test_preserves_units_name(self, north_up_georef):
        ma = np.ma.array(np.ones((2, 2), dtype=np.float32))
        georef = _make_2x2_georef(north_up_georef)
        r = from_masked_array(ma, georef, units="meters", name="test")
        assert r.units == "meters"
        assert r.name == "test"


# ---------------------------------------------------------------------------
# from_existing
# ---------------------------------------------------------------------------


class TestFromExisting:
    def test_basic(self, north_up_georef):
        values = np.array([[1.0, -9999.0], [3.0, 4.0]], dtype=np.float32)
        georef = _make_2x2_georef(north_up_georef, nodata=-9999.0)
        r = from_existing(values, georef)
        assert not r.valid[0, 1]
        assert r.valid[0, 0]
        assert r.validity_provenance == "nodata"

    def test_no_nodata_all_valid(self, north_up_georef):
        values = np.ones((2, 2), dtype=np.float32)
        georef = _make_2x2_georef(north_up_georef, nodata=None)
        r = from_existing(values, georef)
        assert r.all_valid is True
        assert r.validity_provenance == "all_valid"


# ---------------------------------------------------------------------------
# to_existing
# ---------------------------------------------------------------------------


class TestToExisting:
    def test_basic(self, north_up_georef):
        georef = _make_2x2_georef(north_up_georef, nodata=None)
        values = np.array([[1.0, 99.0], [3.0, 4.0]], dtype=np.float32)
        valid = np.ones((2, 2), dtype=np.bool_)
        valid[0, 1] = False
        r = raster(values, georef, valid=valid)
        out_values, out_georef = to_existing(r, nodata=-9999.0)
        assert out_values[0, 1] == -9999.0
        assert out_values[0, 0] == 1.0
        assert out_georef.nodata == -9999.0

    def test_nodata_none_copies_values(self, north_up_georef):
        georef = _make_2x2_georef(north_up_georef)
        values = np.ones((2, 2), dtype=np.float32)
        r = raster(values, georef, valid=np.ones((2, 2), dtype=np.bool_))
        out_values, out_georef = to_existing(r, nodata=None)
        assert out_georef.nodata is None
        assert out_values is not r.values

    def test_rejects_unrepresentable_nodata_uint8(self, north_up_georef):
        georef = _make_2x2_georef(north_up_georef)
        values = np.ones((2, 2), dtype=np.uint8)
        r = raster(values, georef, valid=np.ones((2, 2), dtype=np.bool_))
        with pytest.raises(RasterValidationError, match="out of range"):
            to_existing(r, nodata=-1)

    def test_rejects_out_of_range_nodata_uint8(self, north_up_georef):
        georef = _make_2x2_georef(north_up_georef)
        values = np.ones((2, 2), dtype=np.uint8)
        r = raster(values, georef, valid=np.ones((2, 2), dtype=np.bool_))
        with pytest.raises(RasterValidationError, match="out of range"):
            to_existing(r, nodata=300)

    def test_rejects_float_nodata_for_uint8(self, north_up_georef):
        georef = _make_2x2_georef(north_up_georef)
        values = np.ones((2, 2), dtype=np.uint8)
        r = raster(values, georef, valid=np.ones((2, 2), dtype=np.bool_))
        with pytest.raises(RasterValidationError, match="integer"):
            to_existing(r, nodata=1.5)


# ---------------------------------------------------------------------------
# Validity provenance
# ---------------------------------------------------------------------------


class TestValidityProvenance:
    def test_explicit_valid_provenance(self, north_up_georef):
        georef = _make_2x2_georef(north_up_georef)
        r = raster(np.ones((2, 2), dtype=np.float32), georef, valid=np.ones((2, 2), dtype=np.bool_))
        assert r.validity_provenance == "explicit-caller"

    def test_nodata_provenance(self, north_up_georef):
        georef = _make_2x2_georef(north_up_georef, nodata=-9999.0)
        r = raster(np.ones((2, 2), dtype=np.float32), georef)
        assert r.validity_provenance == "nodata"

    def test_all_valid_provenance(self, north_up_georef):
        georef = _make_2x2_georef(north_up_georef, nodata=None)
        r = raster(np.ones((2, 2), dtype=np.float32), georef, nodata=None)
        assert r.validity_provenance == "all_valid"

    def test_explicit_provenance_override(self, north_up_georef):
        georef = _make_2x2_georef(north_up_georef)
        r = raster(
            np.ones((2, 2), dtype=np.float32),
            georef,
            valid=np.ones((2, 2), dtype=np.bool_),
            validity_provenance="custom",
        )
        assert r.validity_provenance == "custom"

    def test_from_masked_array_provenance(self, north_up_georef):
        ma = np.ma.array(np.ones((2, 2), dtype=np.float32))
        georef = _make_2x2_georef(north_up_georef)
        r = from_masked_array(ma, georef)
        assert r.validity_provenance == "masked-array"

    def test_read_provenance_includes_geotiff_prefix(self, tmp_path, north_up_georef):
        georef = _make_2x2_georef(north_up_georef, nodata=0.0)
        values = np.array([[1.0, 0.0], [3.0, 4.0]], dtype=np.float32)
        path = tmp_path / "prov.tif"
        profile = {
            "driver": "GTiff",
            "width": 2,
            "height": 2,
            "count": 1,
            "dtype": "float32",
            "crs": georef.projection_wkt,
            "transform": Affine.from_gdal(*georef.affine_transform),
            "nodata": 0.0,
        }
        with rasterio.open(path, "w", **profile) as ds:
            ds.write(values, 1)
        r = read(path)
        assert r.validity_provenance is not None
        assert r.validity_provenance.startswith("geotiff:")


# ---------------------------------------------------------------------------
# read() – GeoTIFF source
# ---------------------------------------------------------------------------


class TestRasterRead:
    def test_read_round_trip_values_and_validity(self, tmp_path, north_up_georef):
        georef = _make_2x2_georef(north_up_georef, nodata=0.0)
        values = np.array([[1.0, 0.0], [3.0, 4.0]], dtype=np.float32)
        path = tmp_path / "test.tif"
        profile = {
            "driver": "GTiff",
            "width": 2,
            "height": 2,
            "count": 1,
            "dtype": "float32",
            "crs": georef.projection_wkt,
            "transform": Affine.from_gdal(*georef.affine_transform),
            "nodata": 0.0,
        }
        with rasterio.open(path, "w", **profile) as ds:
            ds.write(values, 1)

        r = read(path)
        assert not r.valid[0, 1]
        assert r.valid[0, 0]
        np.testing.assert_array_equal(r.values, values)

    def test_read_unreferenced_raises(self, tmp_path):
        path = tmp_path / "unref.tif"
        with rasterio.open(path, "w", driver="GTiff", width=2, height=2, count=1, dtype="float32") as ds:
            ds.write(np.ones((2, 2), dtype=np.float32), 1)

        from lunarscout.errors import GeoTiffOpenError

        with pytest.raises(GeoTiffOpenError, match="not georeferenced"):
            read(path)

    def test_read_uses_default_name_from_stem(self, tmp_path, north_up_georef):
        georef = _make_2x2_georef(north_up_georef, nodata=None)
        values = np.ones((2, 2), dtype=np.float32)
        path = tmp_path / "elevation.tif"
        profile = {
            "driver": "GTiff",
            "width": 2,
            "height": 2,
            "count": 1,
            "dtype": "float32",
            "crs": georef.projection_wkt,
            "transform": Affine.from_gdal(*georef.affine_transform),
        }
        with rasterio.open(path, "w", **profile) as ds:
            ds.write(values, 1)
        r = read(path)
        assert r.name == "elevation"

    def test_read_explicit_name_overrides_stem(self, tmp_path, north_up_georef):
        georef = _make_2x2_georef(north_up_georef, nodata=None)
        values = np.ones((2, 2), dtype=np.float32)
        path = tmp_path / "test.tif"
        profile = {
            "driver": "GTiff",
            "width": 2,
            "height": 2,
            "count": 1,
            "dtype": "float32",
            "crs": georef.projection_wkt,
            "transform": Affine.from_gdal(*georef.affine_transform),
        }
        with rasterio.open(path, "w", **profile) as ds:
            ds.write(values, 1)
        r = read(path, name="custom_name")
        assert r.name == "custom_name"

    def test_read_valid_zero_not_confused_with_nodata(self, tmp_path, north_up_georef):
        georef = _make_2x2_georef(north_up_georef, nodata=0)
        values = np.array([[255, 0], [255, 128]], dtype=np.uint8)
        path = tmp_path / "valid_zero.tif"
        profile = {
            "driver": "GTiff",
            "width": 2,
            "height": 2,
            "count": 1,
            "dtype": "uint8",
            "crs": georef.projection_wkt,
            "transform": Affine.from_gdal(*georef.affine_transform),
            "nodata": 0,
        }
        with rasterio.open(path, "w", **profile) as ds:
            ds.write(values, 1)
        r = read(path)
        assert not r.valid[0, 1]
        assert r.valid[0, 0]
        assert r.values[0, 0] == 255
        assert r.values[0, 1] == 0


# ---------------------------------------------------------------------------
# Representation
# ---------------------------------------------------------------------------


class TestRasterRepr:
    def test_repr_includes_shape_and_dtype(self, north_up_georef):
        georef = _make_2x2_georef(north_up_georef)
        r = raster(np.ones((2, 2), dtype=np.float32), georef, valid=np.ones((2, 2), dtype=np.bool_))
        s = repr(r)
        assert "Raster(" in s
        assert "shape=(2, 2)" in s
        assert "float32" in s

    def test_repr_includes_all_valid(self, north_up_georef):
        georef = _make_2x2_georef(north_up_georef)
        r = raster(np.ones((2, 2), dtype=np.float32), georef, valid=np.ones((2, 2), dtype=np.bool_))
        assert "valid=all" in repr(r)

    def test_repr_includes_units_and_name(self, north_up_georef):
        georef = _make_2x2_georef(north_up_georef)
        r = raster(np.ones((2, 2), dtype=np.float32), georef, valid=np.ones((2, 2), dtype=np.bool_), units="meters", name="elev")
        s = repr(r)
        assert "meters" in s
        assert "elev" in s

    def test_repr_includes_provenance(self, north_up_georef):
        georef = _make_2x2_georef(north_up_georef)
        r = raster(
            np.ones((2, 2), dtype=np.float32),
            georef,
            valid=np.ones((2, 2), dtype=np.bool_),
            validity_provenance="explicit-caller",
        )
        assert "explicit-caller" in repr(r)


# ---------------------------------------------------------------------------
# Assertion helper
# ---------------------------------------------------------------------------


class TestAssertRasterEqual:
    def test_matching_rasters_pass(self, north_up_georef):
        georef = _make_2x2_georef(north_up_georef)
        r1 = raster(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32), georef, valid=np.ones((2, 2), dtype=np.bool_))
        r2 = raster(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32), georef, valid=np.ones((2, 2), dtype=np.bool_))
        assert_raster_equal(r1, r2)

    def test_different_grids_raises(self, north_up_georef, shifted_georef):
        r1 = raster(np.ones((8, 10), dtype=np.float32), north_up_georef, valid=np.ones((8, 10), dtype=np.bool_))
        r2 = raster(np.ones((8, 10), dtype=np.float32), shifted_georef, valid=np.ones((8, 10), dtype=np.bool_))
        with pytest.raises(AssertionError, match="grids differ"):
            assert_raster_equal(r1, r2)

    def test_different_validity_raises(self, north_up_georef):
        georef = _make_2x2_georef(north_up_georef)
        valid1 = np.ones((2, 2), dtype=np.bool_)
        valid2 = np.ones((2, 2), dtype=np.bool_)
        valid2[0, 0] = False
        r1 = raster(np.ones((2, 2), dtype=np.float32), georef, valid=valid1)
        r2 = raster(np.ones((2, 2), dtype=np.float32), georef, valid=valid2)
        with pytest.raises(AssertionError, match="Validity masks differ"):
            assert_raster_equal(r1, r2)

    def test_different_values_raises(self, north_up_georef):
        georef = _make_2x2_georef(north_up_georef)
        r1 = raster(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32), georef, valid=np.ones((2, 2), dtype=np.bool_))
        r2 = raster(np.array([[1.0, 99.0], [3.0, 4.0]], dtype=np.float32), georef, valid=np.ones((2, 2), dtype=np.bool_))
        with pytest.raises(AssertionError, match="Raster values differ"):
            assert_raster_equal(r1, r2)


# ---------------------------------------------------------------------------
# Import boundary
# ---------------------------------------------------------------------------


class TestImportBoundary:
    def test_import_map_algebra_no_cuda(self):
        import importlib

        if "lunarscout.map_algebra" in sys.modules:
            del sys.modules["lunarscout.map_algebra"]
        before = set(sys.modules.keys())
        import lunarscout.map_algebra  # noqa: F811
        after = set(sys.modules.keys())
        new_modules = after - before
        cuda_imported = {m for m in new_modules if "cuda" in m.lower() and "numba" in m.lower()}
        assert not cuda_imported, f"CUDA imported: {cuda_imported}"

    def test_import_map_algebra_no_spiceypy(self):
        import importlib

        if "lunarscout.map_algebra" in sys.modules:
            del sys.modules["lunarscout.map_algebra"]
        before = set(sys.modules.keys())
        import lunarscout.map_algebra  # noqa: F811
        after = set(sys.modules.keys())
        new_modules = after - before
        assert "spiceypy" not in new_modules, f"SpiceyPy pulled in: {new_modules}"

    def test_import_map_algebra_no_numba(self):
        import importlib

        if "lunarscout.map_algebra" in sys.modules:
            del sys.modules["lunarscout.map_algebra"]
        before = set(sys.modules.keys())
        import lunarscout.map_algebra  # noqa: F811
        after = set(sys.modules.keys())
        new_modules = after - before
        assert "numba" not in new_modules, f"map_algebra pulled in numba: {new_modules}"

    def test_raster_factory_no_side_effects(self, north_up_georef):
        georef = _make_2x2_georef(north_up_georef)
        result = raster(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32), georef)
        assert result.shape == (2, 2)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_2x2_georef(template: GeoReference, nodata: int | float | None = None) -> GeoReference:
    return GeoReference(
        projection_wkt=template.projection_wkt,
        projection_proj4=template.projection_proj4,
        affine_transform=template.affine_transform,
        width=2,
        height=2,
        pixel_size_x=template.pixel_size_x,
        pixel_size_y=template.pixel_size_y,
        nodata=nodata,
    )


def _make_3x3_georef(template: GeoReference, nodata: int | float | None = None) -> GeoReference:
    return GeoReference(
        projection_wkt=template.projection_wkt,
        projection_proj4=template.projection_proj4,
        affine_transform=template.affine_transform,
        width=3,
        height=3,
        pixel_size_x=template.pixel_size_x,
        pixel_size_y=template.pixel_size_y,
        nodata=nodata,
    )


def _georef_8x10() -> GeoReference:
    return GeoReference(
        projection_wkt=(
            'PROJCS["Moon_South_Pole_Stereographic",'
            'GEOGCS["Moon 2000",'
            'DATUM["D_Moon_2000",'
            'SPHEROID["Moon_2000_IAU_IAG",1737400.0,0.0]],'
            'PRIMEM["Reference_Meridian",0],'
            'UNIT["degree",0.0174532925199433]],'
            'PROJECTION["Polar_Stereographic"],'
            'PARAMETER["latitude_of_origin",-90],'
            'PARAMETER["central_meridian",0],'
            'PARAMETER["scale_factor",1],'
            'PARAMETER["false_easting",0],'
            'PARAMETER["false_northing",0],'
            'UNIT["metre",1]]'
        ),
        projection_proj4=(
            "+proj=stere +lat_0=-90 +lon_0=0 +k=1 +x_0=0 +y_0=0 "
            "+R=1737400 +units=m +no_defs +type=crs"
        ),
        affine_transform=(1000.0, 20.0, 0.0, 2000.0, 0.0, -20.0),
        width=10,
        height=8,
        pixel_size_x=20.0,
        pixel_size_y=-20.0,
        nodata=None,
    )
