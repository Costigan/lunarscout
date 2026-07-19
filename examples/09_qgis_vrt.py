"""Inspect individual timestamp TIFFs and the derived multi-band VRT.

Inputs: deterministic file-backed temporal series under --workspace.
Outputs: none; prints paths and GDAL band metadata for QGIS inspection.
Resources: metadata and one raster-band read; a GPU is not required.
"""

from __future__ import annotations

import rasterio

from _example_support import ensure_synthetic_series, example_parser


def main() -> None:
    args = example_parser(__doc__).parse_args()
    series = ensure_synthetic_series(args.workspace)
    if series.vrt_path is None:
        raise RuntimeError("This example requires a series with a VRT.")
    with rasterio.open(series.vrt_path) as dataset:
        band_count = dataset.count
        descriptions = dataset.descriptions

    print(f"Open this VRT in QGIS: {series.vrt_path}")
    print(f"VRT bands: {band_count}")
    for band_index in (1, band_count):
        print(f"band {band_index}: {descriptions[band_index - 1]}")
    print(f"Open one timestamp directly: {series.layer_paths[0]}")


if __name__ == "__main__":
    main()
