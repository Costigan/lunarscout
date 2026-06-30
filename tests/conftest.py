from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import rasterio
from pyproj import CRS
from rasterio.transform import Affine

_NODATA_TAG = "LUNARSCOUT_NODATA_VALUE"


@pytest.fixture
def lunar_projection() -> tuple[str, str]:
    crs = CRS.from_user_input("ESRI:103878")
    return crs.to_wkt(), crs.to_proj4()


@pytest.fixture
def affine_transform() -> tuple[float, float, float, float, float, float]:
    return (1000.0, 20.0, 0.0, 2000.0, 0.0, -20.0)


@pytest.fixture
def make_geotiff(tmp_path: Path, lunar_projection, affine_transform):
    created: list[Path] = []

    def _make(
        name: str,
        arrays: list[np.ndarray],
        *,
        nodata: int | float | None = None,
        projection: bool = True,
        transform: bool = True,
    ) -> Path:
        path = tmp_path / name
        first = np.asarray(arrays[0])
        profile = {
            "driver": "GTiff",
            "width": int(first.shape[1]),
            "height": int(first.shape[0]),
            "count": len(arrays),
            "dtype": first.dtype,
        }
        if projection:
            profile["crs"] = lunar_projection[0]
        if transform:
            profile["transform"] = Affine.from_gdal(*affine_transform)
        if nodata is not None:
            profile["nodata"] = nodata
        with rasterio.open(path, "w", **profile) as dataset:
            for index, array in enumerate(arrays, start=1):
                dataset.write(np.asarray(array), index)
                if nodata is not None and np.issubdtype(first.dtype, np.integer):
                    dataset.update_tags(index, **{_NODATA_TAG: str(int(nodata))})
        created.append(path)
        return path

    yield _make

    for path in created:
        path.unlink(missing_ok=True)
