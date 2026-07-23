"""Reject shape-only matches and explicitly align lunar rasters.

Inputs: none; all values and lunar grid metadata are deterministic.
Outputs: printed grid checks, aligned values, and coordinate rasters.
Resources: one tiny CPU/GDAL resampling operation; no files, SPICE, or GPU.
"""

from __future__ import annotations

import lunarscout as ls
import numpy as np

from _example_support import example_parser, synthetic_georef


ma = ls.map_algebra


def main() -> None:
    example_parser(__doc__).parse_args()
    reference_grid = synthetic_georef(
        width=4,
        height=3,
        origin_x=-20.0,
        origin_y=20.0,
        pixel_size=10.0,
        nodata=None,
    )
    shifted_grid = synthetic_georef(
        width=4,
        height=3,
        origin_x=-15.0,
        origin_y=20.0,
        pixel_size=10.0,
        nodata=None,
    )
    values = np.arange(12, dtype=np.float32).reshape(3, 4)
    reference = ma.raster(values, reference_grid, units="metres", name="reference")
    shifted = ma.raster(values + 100.0, shifted_grid, units="metres", name="shifted")

    print(f"Same array shape: {reference.shape == shifted.shape}")
    print(f"Same spatial grid: {ls.same_grid(reference.georef, shifted.georef)}")
    try:
        ls.require_same_grid(reference.georef, shifted.georef)
    except ls.GridMismatchError as error:
        print(f"require_same_grid rejected them: {error.code}")

    try:
        ma.add(reference, shifted)
    except ls.MapAlgebraGridError as error:
        print(f"Direct algebra rejected the shape-only match: {error.code}")

    aligned = ma.align(
        shifted,
        to=reference.georef,
        resampling="nearest",
        output_nodata=None,
    )
    ls.require_same_grid(reference.georef, aligned.georef)
    combined = reference + aligned
    print(f"Same grid after explicit alignment: {ls.same_grid(reference.georef, aligned.georef)}")
    print("Aligned values:")
    print(aligned.values)
    print("Combined values:")
    print(combined.values)
    print("Combined validity:")
    print(combined.valid)

    rows = ma.compute(ma.row_indices(reference.georef))
    columns = ma.compute(ma.column_indices(reference.georef))
    projected_x = ma.compute(ma.projected_x(reference.georef, anchor="center"))
    projected_y = ma.compute(ma.projected_y(reference.georef, anchor="center"))
    print("\nZero-based row coordinates:")
    print(rows.values)
    print("Zero-based column coordinates:")
    print(columns.values)
    print(f"Projected x units: {projected_x.units!r}")
    print(projected_x.values)
    print(f"Projected y units: {projected_y.units!r}")
    print(projected_y.values)
    print(
        "\nCoordinates come from the lunar grid CRS. Lunarscout does not "
        "silently introduce WGS84."
    )


if __name__ == "__main__":
    main()
