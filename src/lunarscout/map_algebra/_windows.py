from __future__ import annotations

from collections.abc import Iterator
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np

from ..errors import MapAlgebraError, MapAlgebraExpressionError
from ..georeference import GeoReference
from ..raster import Raster
from ._model import RasterExpression


def enumerate_windows(
    width: int,
    height: int,
    window_width: int,
    window_height: int,
) -> Iterator[tuple[int, int, int, int, int, int]]:
    """Yield ``(idx, x0, y0, width, height, n_columns)`` in row-major order.

    Enumeration itself uses constant memory regardless of raster dimensions.
    """
    n_cols = max(1, (width + window_width - 1) // window_width)
    n_rows = max(1, (height + window_height - 1) // window_height)
    idx = 0
    for row in range(n_rows):
        y0 = row * window_height
        for col in range(n_cols):
            x0 = col * window_width
            w = min(window_width, width - x0)
            h = min(window_height, height - y0)
            yield idx, x0, y0, w, h, n_cols
            idx += 1


def _derive_validity_from_dataset(
    ds: Any,
    band: int,
    y0: int,
    x0: int,
    h: int,
    w: int,
) -> np.ndarray[Any, Any]:
    """Derive a validity mask from dataset masks and nodata."""
    try:
        mask_data = ds.read_masks(
            band,
            window=((y0, y0 + h), (x0, x0 + w)),
        )
        if mask_data.ndim == 3 and mask_data.shape[0] == 1:
            mask_data = mask_data[0]
        return np.asarray(mask_data, dtype=np.bool_)
    except Exception as exc:
        raise MapAlgebraError(
            "Unable to read the source validity mask for the requested window.",
            code="map_algebra_source_mask_read_failed",
            details={"band": band, "x": x0, "y": y0, "width": w, "height": h},
        ) from exc



class SourceWindowCache:
    """Bounded cache of open source datasets and per-source window data.

    Datasets are opened lazily on first access and closed after explicit
    ``close()`` or via context-manager exit.
    """

    def __init__(self, max_datasets: int = 16, max_windows: int = 64) -> None:
        if max_datasets < 1 or max_windows < 1:
            raise ValueError("Cache bounds must be positive.")
        self._max_datasets = max_datasets
        self._max_windows = max_windows
        self._datasets: OrderedDict[str, Any] = OrderedDict()
        self._windows: OrderedDict[
            tuple[str, int, int, int, int, int],
            tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]],
        ] = OrderedDict()
        self._closed = False

    def _open_dataset(self, path: str) -> Any:
        import rasterio

        key = path
        if key in self._datasets:
            self._datasets.move_to_end(key)
            return self._datasets[key]
        if len(self._datasets) >= self._max_datasets:
            oldest = next(iter(self._datasets))
            self._datasets[oldest].close()
            del self._datasets[oldest]
        ds = rasterio.open(key)
        self._datasets[key] = ds
        return ds

    def read_values(
        self,
        node: RasterExpression,
        window_idx: int,
        x0: int,
        y0: int,
        w: int,
        h: int,
    ) -> np.ndarray[Any, Any]:
        cache_key = (node._node_id, window_idx, x0, y0, w, h)
        if cache_key in self._windows:
            self._windows.move_to_end(cache_key)
            return self._windows[cache_key][0].copy()

        op_id = node._operation_id
        if op_id == "source":
            values, valid = self._read_file_window(node, x0, y0, w, h)
            self._cache_window(cache_key, values, valid)
            return values.copy()
        if op_id == "constant":
            const: Raster = node._operands[0]
            values = const.values[y0 : y0 + h, x0 : x0 + w].copy()
            valid = const.valid[y0 : y0 + h, x0 : x0 + w].copy()
            self._cache_window(cache_key, values, valid)
            return values.copy()
        if op_id.startswith("coordinate."):
            values, valid = self._generate_coordinate_window(node, x0, y0, w, h)
            self._cache_window(cache_key, values, valid)
            return values.copy()
        raise MapAlgebraExpressionError(
            f"Cannot read values from non-source node: {op_id}",
            code="map_algebra_not_a_source",
            details={"operation_id": op_id},
        )

    def read_valid(
        self,
        node: RasterExpression,
        window_idx: int,
        x0: int,
        y0: int,
        w: int,
        h: int,
    ) -> np.ndarray[Any, Any]:
        cache_key = (node._node_id, window_idx, x0, y0, w, h)
        if cache_key in self._windows:
            self._windows.move_to_end(cache_key)
            return self._windows[cache_key][1].copy()

        values = self.read_values(node, window_idx, x0, y0, w, h)
        return self._windows[cache_key][1].copy()

    def _cache_window(
        self,
        key: tuple[str, int, int, int, int, int],
        values: np.ndarray[Any, Any],
        valid: np.ndarray[Any, Any],
    ) -> None:
        if len(self._windows) >= self._max_windows:
            self._windows.popitem(last=False)
        self._windows[key] = (values, valid)
        self._windows.move_to_end(key)

    def discard_window(self, window_idx: int) -> None:
        """Release decoded data for a completed output window."""
        for key in tuple(self._windows):
            if key[1] == window_idx:
                del self._windows[key]

    def _read_file_window(
        self,
        node: RasterExpression,
        x0: int,
        y0: int,
        w: int,
        h: int,
    ) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        params = node._params_dict
        path = Path(params["path"])
        band = int(params["band"])
        ds = self._open_dataset(str(path))
        data = ds.read(band, window=((y0, y0 + h), (x0, x0 + w)))
        if data.ndim == 3 and data.shape[0] == 1:
            data = data[0]
        valid = _derive_validity_from_dataset(ds, band, y0, x0, h, w)
        return data, valid

    def _generate_coordinate_window(
        self,
        node: RasterExpression,
        x0: int,
        y0: int,
        w: int,
        h: int,
    ) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        georef = node._inferred_grid
        if georef is None:
            raise MapAlgebraExpressionError(
                "Coordinate node has no grid.",
                code="map_algebra_internal_error",
            )
        op_id = node._operation_id
        anchor = str(node._params_dict.get("anchor", "center"))
        offset = 0.5 if anchor == "center" else 0.0

        if op_id == "coordinate.row_indices":
            values = np.broadcast_to(
                np.arange(y0, y0 + h, dtype=np.int64).reshape(-1, 1), (h, w)
            ).copy()
            valid = np.ones((h, w), dtype=np.bool_)
        elif op_id == "coordinate.column_indices":
            values = np.broadcast_to(
                np.arange(x0, x0 + w, dtype=np.int64).reshape(1, -1), (h, w)
            ).copy()
            valid = np.ones((h, w), dtype=np.bool_)
        else:
            rows, cols = np.indices((h, w), dtype=np.float64)
            rows = rows + y0
            cols = cols + x0
            affine = georef.affine_transform
            x_vals = affine[0] + (cols + offset) * affine[1] + (rows + offset) * affine[2]
            y_vals = affine[3] + (cols + offset) * affine[4] + (rows + offset) * affine[5]
            if op_id == "coordinate.projected_x":
                values = x_vals
            elif op_id == "coordinate.projected_y":
                values = y_vals
            elif op_id in ("coordinate.longitude", "coordinate.latitude"):
                values = _transform_to_geodetic(
                    georef, x_vals, y_vals,
                    longitude=(op_id == "coordinate.longitude"),
                )
            else:
                raise MapAlgebraError(
                    f"Unknown coordinate operation: {op_id}",
                    code="map_algebra_unknown_operation",
                    details={"operation_id": op_id},
                )
            valid = np.isfinite(values) if values.dtype.kind == "f" else np.ones((h, w), dtype=np.bool_)
        return values, valid

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._windows.clear()
        for ds in self._datasets.values():
            try:
                ds.close()
            except Exception:
                pass
        self._datasets.clear()

    def __enter__(self) -> SourceWindowCache:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    @property
    def dataset_count(self) -> int:
        return len(self._datasets)

    @property
    def window_count(self) -> int:
        return len(self._windows)

    @property
    def is_closed(self) -> bool:
        return self._closed


def _transform_to_geodetic(
    georef: GeoReference,
    x_values: np.ndarray[Any, Any],
    y_values: np.ndarray[Any, Any],
    *,
    longitude: bool,
) -> np.ndarray[Any, Any]:
    try:
        from pyproj import CRS, Transformer
    except ImportError as exc:
        raise MapAlgebraError(
            "pyproj is required for longitude and latitude generation.",
            code="map_algebra_pyproj_required",
        ) from exc

    try:
        source_crs = CRS.from_wkt(georef.projection_wkt)
        geodetic_crs = source_crs.geodetic_crs
        if geodetic_crs is None:
            raise ValueError("CRS has no geodetic CRS")
        transformer = Transformer.from_crs(source_crs, geodetic_crs, always_xy=True)
        lon_values, lat_values = transformer.transform(x_values, y_values)
    except Exception as exc:
        raise MapAlgebraError(
            "Coordinates cannot be transformed to the grid's geodetic CRS.",
            code="map_algebra_unknown_geodetic_crs",
            details={"error": str(exc)},
        ) from exc
    selected = lon_values if longitude else lat_values
    return np.asarray(selected, dtype=np.float64).reshape(x_values.shape)
