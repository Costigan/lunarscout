"""Map-algebra terrain operations, resampling, and windowed write example.

Demonstrates: creating slope and hillshade expressions, explicit resampling
onto another grid, combining with a local expression, writing in bounded
windows, and canonical validity handling.

Inputs:  deterministic DEM and file-backed illumination series under
         --workspace.
Outputs: terrain_resample/slope.tif, terrain_resample/hillshade.tif,
         terrain_resample/resampled_hillshade.tif,
         terrain_resample/combined_score.tif.
Resources: small eager rasters and bounded window writes.
"""

from __future__ import annotations

import lunarscout as ls
import numpy as np

from _example_support import ensure_synthetic_scenario, ensure_synthetic_series, example_parser

ma = ls.map_algebra


def main() -> None:
    args = example_parser(
        "Map-algebra terrain, resampling, and windowed write example."
    ).parse_args()
    scenario = ensure_synthetic_scenario(args.workspace)
    series = ensure_synthetic_series(args.workspace)
    dem, georef = ls.read_geotiff(scenario.dem_path())
    if georef is None:
        raise RuntimeError("The DEM must be georeferenced.")

    # ------------------------------------------------------------------
    # 1. Build terrain expressions from the DEM.
    # ------------------------------------------------------------------
    dem_raster = ma.from_existing(dem, georef, units="metres", name="elevation")

    # Raster operands compute eagerly. File-backed source operands create
    # terrain nodes that retain their one-pixel halo for bounded writes.
    slope_eager = ma.slope(dem_raster, units="degrees")
    hillshade_eager = ma.hillshade(dem_raster, azimuth=315.0, altitude=45.0)
    print(
        f"Slope:  {slope_eager.shape}, {slope_eager.dtype}, "
        f"valid={slope_eager.valid.sum()}/{slope_eager.valid.size}"
    )
    print(
        f"Hillshade: {hillshade_eager.shape}, {hillshade_eager.dtype}, "
        f"valid={hillshade_eager.valid.sum()}/{hillshade_eager.valid.size}"
    )

    dem_expr = ma.source(scenario.dem_path(), units="metres")
    slope_expr = ma.slope(dem_expr, units="degrees")
    hillshade_expr = ma.hillshade(dem_expr, azimuth=315.0, altitude=45.0)

    # ------------------------------------------------------------------
    # 2. Write terrain products in bounded windows.
    # ------------------------------------------------------------------
    out_dir = args.workspace / "terrain_resample"
    out_dir.mkdir(parents=True, exist_ok=True)

    slope_path = ma.write(
        out_dir / "slope.tif", slope_expr,
        window_width=128, window_height=128, overwrite=True,
    )
    hillshade_path = ma.write(
        out_dir / "hillshade.tif", hillshade_expr,
        window_width=128, window_height=128, overwrite=True,
    )
    print(f"Slope windowed write:     {slope_path}")
    print(f"Hillshade windowed write: {hillshade_path}")

    # Verify the windowed slope matches eager compute.
    slope_roundtrip = ma.read(slope_path)
    assert slope_roundtrip.array_equal(slope_eager)
    print("  slope equality check passed.")

    # Verify the windowed hillshade matches eager compute.
    hillshade_roundtrip = ma.read(hillshade_path)
    assert hillshade_roundtrip.array_equal(hillshade_eager)
    print("  hillshade equality check passed.")

    # ------------------------------------------------------------------
    # 3. Explicit cross-grid resampling.
    # ------------------------------------------------------------------
    from lunarscout.georeference import GeoReference

    dst_georef = GeoReference(
        projection_wkt=georef.projection_wkt,
        projection_proj4=georef.projection_proj4,
        affine_transform=(georef.affine_transform[0] + 3.0, 14.0, 0.0,
                          georef.affine_transform[3] - 5.0, 0.0, -14.0),
        width=georef.width * 2 - 1,
        height=georef.height * 2 - 1,
        pixel_size_x=14.0,
        pixel_size_y=-14.0,
        nodata=None,
    )

    # resample_to with an expression operand returns a resampling expression node.
    resampled_expr = ma.resample_to(hillshade_expr, dst_georef, resampling="nearest")
    resampled_path = ma.write(
        out_dir / "resampled_hillshade.tif",
        resampled_expr,
        window_width=128,
        window_height=128,
        overwrite=True,
    )
    print(f"Resampled hillshade: {resampled_path}")
    resampled = ma.read(resampled_path)
    assert resampled.valid.any()
    print(f"  shape={resampled.shape}, valid={resampled.valid.sum()}/{resampled.valid.size}")

    # ------------------------------------------------------------------
    # 4. Combine terrain with sunlight in a local expression.
    # ------------------------------------------------------------------
    mean_sun_bare, mean_sun_georef = ls.temporal_mean(series)
    sun = ma.from_existing(
        mean_sun_bare, mean_sun_georef.with_nodata(None), units="fraction", name="mean_sun",
    )
    ls.require_same_grid(slope_eager.georef, sun.georef)

    sun_expr = sun.expression()
    candidate = (slope_expr <= 8.0) & (sun_expr >= 0.60)
    slope_score = 1.0 - ma.normalize_minmax(
        slope_expr, minimum=0.0, maximum=8.0,
    )
    sun_score = ma.normalize_minmax(sun_expr, minimum=0.0, maximum=1.0)
    score = ma.where(candidate, 0.4 * sun_score + 0.6 * slope_score, ma.invalid)

    # Write the combined score in bounded windows.
    score_path = ma.write(
        out_dir / "combined_score.tif",
        score,
        window_width=128,
        window_height=128,
        invalid_value=-1.0,
        overwrite=True,
    )
    print(f"Combined score: {score_path}")

    # ------------------------------------------------------------------
    # 5. Demonstrate canonical validity.
    # ------------------------------------------------------------------
    score_check = ma.read(score_path)
    assert score_check.valid.any()
    assert not score_check.valid[np.isclose(score_check.values, -1.0)].any()
    print(f"  validity: {score_check.valid.sum()} valid, "
          f"{score_check.valid.size - score_check.valid.sum()} invalid")
    print("Canonical validity check passed (fill != validity).")


if __name__ == "__main__":
    main()
