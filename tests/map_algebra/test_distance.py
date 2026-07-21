from __future__ import annotations

import numpy as np
import pytest
from pyproj import CRS

from lunarscout.errors import DistanceFieldError, MapAlgebraError
from lunarscout.georeference import GeoReference
from lunarscout.map_algebra import (
    distance_to,
    raster as ma_raster,
    signed_distance,
)
from tests.map_algebra.conftest import MOON_WKT, MOON_PROJ4


def _georef(
    h: int,
    w: int,
    px: float = 20.0,
    py: float = -20.0,
    rx: float = 0.0,
    ry: float = 0.0,
    wkt: str = MOON_WKT,
    proj4: str = MOON_PROJ4,
) -> GeoReference:
    return GeoReference(
        projection_wkt=wkt, projection_proj4=proj4,
        affine_transform=(1000.0, px, rx, 2000.0, ry, py),
        width=w, height=h, pixel_size_x=px, pixel_size_y=py, nodata=None,
    )


def _make(values, px=20.0, py=-20.0):
    g = _georef(values.shape[0], values.shape[1], px=px, py=py)
    return ma_raster(values, g)


class TestDistanceTo:
    def test_single_center_seed(self):
        v = np.zeros((5, 5), dtype=np.bool_)
        v[2, 2] = True
        r = _make(v)
        result = distance_to(r, metric="euclidean")
        assert abs(result.values[0, 0] - np.sqrt(8)) < 0.01
        assert result.values[2, 2] == 0.0

    def test_taxicab(self):
        v = np.zeros((5, 5), dtype=np.bool_)
        v[2, 2] = True
        r = _make(v)
        result = distance_to(r, metric="taxicab")
        assert result.values[0, 0] == 4.0
        assert result.values[2, 2] == 0.0

    def test_chessboard(self):
        v = np.zeros((5, 5), dtype=np.bool_)
        v[2, 2] = True
        r = _make(v)
        result = distance_to(r, metric="chessboard")
        assert result.values[0, 0] == 2.0

    def test_max_distance(self):
        v = np.zeros((7, 7), dtype=np.bool_)
        v[3, 3] = True
        r = _make(v)
        result = distance_to(r, max_distance=2.0)
        assert result.values[3, 3] == 0.0
        assert result.values[0, 0] == 2.0

    @pytest.mark.parametrize("max_distance", [-1.0, np.nan, np.inf])
    def test_invalid_max_distance_rejected(self, max_distance):
        v = np.zeros((3, 3), dtype=np.bool_)
        v[1, 1] = True
        r = _make(v)
        with pytest.raises(DistanceFieldError, match="finite"):
            distance_to(r, max_distance=max_distance)

    def test_physical_units(self):
        v = np.zeros((5, 5), dtype=np.bool_)
        v[2, 2] = True
        r = _make(v, px=10.0, py=-10.0)
        result = distance_to(r, units="physical")
        assert result.units == "metre"
        assert abs(result.values[0, 0] - np.sqrt(8) * 10.0) < 0.1

    def test_physical_max_distance_in_meters(self):
        v = np.zeros((7, 7), dtype=np.bool_)
        v[3, 3] = True
        r = _make(v, px=10.0, py=-10.0)
        result = distance_to(r, units="physical", max_distance=25.0)
        assert result.values[0, 0] <= 25.0

    def test_no_seeds_raises(self):
        v = np.zeros((3, 3), dtype=np.bool_)
        r = _make(v)
        with pytest.raises(DistanceFieldError, match="No valid seed"):
            distance_to(r)

    def test_non_boolean_raises(self):
        r = _make(np.ones((3, 3), dtype=np.float32))
        with pytest.raises(MapAlgebraError):
            distance_to(r)

    def test_invalid_input_pixels_preserved(self):
        v = np.zeros((5, 5), dtype=np.bool_)
        v[2, 2] = True
        valid = np.ones((5, 5), dtype=np.bool_)
        valid[0, 0] = False
        r = ma_raster(v, _georef(5, 5), valid=valid)
        result = distance_to(r)
        assert not result.valid[0, 0]
        assert result.valid[2, 2]

    def test_taxicab_physical_rejected(self):
        v = np.zeros((5, 5), dtype=np.bool_)
        v[2, 2] = True
        g = _georef(5, 5, px=20.0, py=-20.0)
        r = ma_raster(v, g)
        with pytest.raises(DistanceFieldError, match="Physical units"):
            distance_to(r, metric="taxicab", units="physical")

    def test_anisotropic_physical_uses_both_affine_basis_vectors(self):
        v = np.zeros((5, 5), dtype=np.bool_)
        v[2, 2] = True
        g = _georef(5, 5, px=20.0, py=-10.0)
        r = ma_raster(v, g)
        result = distance_to(r, units="physical")
        assert result.values[2, 1] == pytest.approx(20.0)
        assert result.values[1, 2] == pytest.approx(10.0)

    def test_rotated_physical_uses_both_affine_basis_vectors(self):
        v = np.zeros((3, 3), dtype=np.bool_)
        v[1, 1] = True
        g = _georef(3, 3, px=0.0, py=0.0, rx=-10.0, ry=10.0)
        result = distance_to(ma_raster(v, g), units="physical")
        assert result.values[0, 1] == pytest.approx(10.0)
        assert result.values[1, 0] == pytest.approx(10.0)
        assert result.values[0, 0] == pytest.approx(np.sqrt(200.0))

    def test_skewed_physical_uses_full_affine(self):
        v = np.zeros((3, 3), dtype=np.bool_)
        v[1, 1] = True
        g = _georef(3, 3, px=10.0, py=-10.0, rx=5.0, ry=0.0)
        result = distance_to(ma_raster(v, g), units="physical")
        assert result.values[1, 0] == pytest.approx(10.0)
        assert result.values[0, 1] == pytest.approx(np.sqrt(125.0))

    def test_geographic_physical_rejected(self):
        crs = CRS.from_wkt(MOON_WKT).geodetic_crs
        assert crs is not None
        v = np.zeros((3, 3), dtype=np.bool_)
        v[1, 1] = True
        g = _georef(
            3,
            3,
            px=1.0,
            py=-1.0,
            wkt=crs.to_wkt(),
            proj4="+proj=longlat +R=1737400 +no_defs +type=crs",
        )
        with pytest.raises(DistanceFieldError, match="geographic"):
            distance_to(ma_raster(v, g), units="physical")

    def test_projected_crs_unit_is_preserved(self):
        foot_wkt = MOON_WKT.replace(
            'UNIT["metre",1]]',
            'UNIT["US survey foot",0.3048006096012192]]',
        )
        v = np.zeros((3, 3), dtype=np.bool_)
        v[1, 1] = True
        g = _georef(
            3,
            3,
            px=10.0,
            py=-10.0,
            wkt=foot_wkt,
            proj4=MOON_PROJ4.replace("+units=m", "+units=us-ft"),
        )
        result = distance_to(ma_raster(v, g), units="physical")
        assert result.units == "US survey foot"
        assert result.values[0, 1] == pytest.approx(10.0)

    def test_unknown_units_rejected(self):
        v = np.zeros((3, 3), dtype=np.bool_)
        v[1, 1] = True
        r = _make(v)
        with pytest.raises(MapAlgebraError, match="Unknown distance"):
            distance_to(r, units="bogus")  # type: ignore[arg-type]

    def test_unknown_units_rejected_typed(self):
        v = np.zeros((3, 3), dtype=np.bool_)
        v[1, 1] = True
        r = _make(v)
        with pytest.raises(MapAlgebraError, match="Unknown distance"):
            signed_distance(r, units="bogus")  # type: ignore[arg-type]

    def test_invalid_output_compute(self):
        v = np.zeros((3, 3), dtype=np.bool_)
        v[1, 1] = True
        valid = np.ones((3, 3), dtype=np.bool_)
        valid[0, 0] = False
        r = ma_raster(v, _georef(3, 3), valid=valid)
        result = distance_to(r, invalid_output="compute")
        assert result.valid.all()
        assert result.values[0, 0] > 0

    def test_invalid_output_unknown_rejected(self):
        v = np.zeros((3, 3), dtype=np.bool_)
        v[1, 1] = True
        with pytest.raises(MapAlgebraError, match="invalid_output"):
            distance_to(_make(v), invalid_output="bogus")  # type: ignore[arg-type]


class TestSignedDistance:
    def test_basic(self):
        v = np.zeros((5, 5), dtype=np.bool_)
        v[2:, :] = True
        r = _make(v, px=1.0, py=-1.0)
        result = signed_distance(r)
        assert result.values[0, 0] < 0
        assert result.values[4, 0] > 0

    def test_center_true(self):
        v = np.zeros((5, 5), dtype=np.bool_)
        v[2, 2] = True
        r = _make(v, px=1.0, py=-1.0)
        result = signed_distance(r)
        assert result.values[0, 0] < 0
        assert result.values[2, 2] > 0

    def test_invalid_pixels_remain_invalid(self):
        v = np.zeros((5, 5), dtype=np.bool_)
        v[2, 2] = True
        valid = np.ones((5, 5), dtype=np.bool_)
        valid[0, 0] = False
        r = ma_raster(v, _georef(5, 5, px=1.0, py=-1.0), valid=valid)
        result = signed_distance(r)
        assert not result.valid[0, 0]
        assert result.valid[2, 2]

    def test_all_true_raises(self):
        v = np.ones((3, 3), dtype=np.bool_)
        r = _make(v)
        with pytest.raises(DistanceFieldError, match="No valid False"):
            signed_distance(r)

    def test_all_false_raises(self):
        v = np.zeros((3, 3), dtype=np.bool_)
        r = _make(v)
        with pytest.raises(DistanceFieldError, match="No valid True"):
            signed_distance(r)

    def test_max_distance(self):
        v = np.zeros((9, 9), dtype=np.bool_)
        v[4, 4] = True
        r = _make(v, px=1.0, py=-1.0)
        result = signed_distance(r, max_distance=3.0)
        assert abs(result.values[0, 0]) <= 3.0

    def test_physical_units(self):
        v = np.zeros((5, 5), dtype=np.bool_)
        v[2, 2] = True
        g = _georef(5, 5, px=20.0, py=-20.0)
        r = ma_raster(v, g)
        result = signed_distance(r, units="physical")
        assert result.units == "metre"
        assert result.values[2, 2] == pytest.approx(20.0)

    def test_invalid_output_compute_uses_payload_class(self):
        v = np.array([[False, True, False]], dtype=np.bool_)
        valid = np.array([[True, True, False]], dtype=np.bool_)
        r = ma_raster(v, _georef(1, 3, px=1.0, py=-1.0), valid=valid)
        result = signed_distance(r, invalid_output="compute")
        assert result.valid.all()
        assert result.values.tolist() == [[-1.0, 1.0, -1.0]]
