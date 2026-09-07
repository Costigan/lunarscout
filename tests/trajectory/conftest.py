from __future__ import annotations

from collections.abc import Callable

import pytest
from pyproj import CRS

from lunarscout import GeoReference


@pytest.fixture
def make_trajectory_georef() -> Callable[..., GeoReference]:
    def make(
        *,
        width: int = 5,
        height: int = 5,
        affine: tuple[float, float, float, float, float, float] = (
            0.0,
            10.0,
            0.0,
            0.0,
            0.0,
            -10.0,
        ),
        crs_input: str = "ESRI:103878",
    ) -> GeoReference:
        crs = CRS.from_user_input(crs_input)
        return GeoReference(
            projection_wkt=crs.to_wkt(),
            projection_proj4=crs.to_proj4(),
            affine_transform=affine,
            width=width,
            height=height,
            pixel_size_x=affine[1],
            pixel_size_y=affine[5],
            nodata=None,
        )

    return make
