# Lunarscout Use Cases

This document collects typical, end-user-facing uses of the Lunarscout
library. Each use case maps to one or more public API entry points and is
written so that a script or notebook author can recognize their problem,
find the relevant function, and copy a minimal starting example. The list is
organized by domain rather than by implementation; the same underlying
capabilities often serve several different workflows.

Lunarscout is a calculation library. It supplies the terrain, raster,
temporal, celestial, horizon, lighting, visibility, and mobility mathematics
and the durable file formats that carry them. It does not decide whether a
site is safe or suitable, and it does not run a web application, database, or
job service. The thresholds and parameters in these examples are
illustrative, not mission policy.

For exact parameter contracts, file formats, numeric and validity rules,
restart behavior, and error codes, see the [User Guide](USER_GUIDE.md). For
package boundaries, pipeline internals, and resource management, see the
[Architecture](ARCHITECTURE.md).

## Table of Contents

- [Terrain and raster analysis](#terrain-and-raster-analysis)
- [Region and candidate screening](#region-and-candidate-screening)
- [Temporal data](#temporal-data)
- [Celestial geometry (SPICE)](#celestial-geometry-spice)
- [Horizons and lighting products](#horizons-and-lighting-products)
- [Mission and mobility planning](#mission-and-mobility-planning)
- [Map algebra](#map-algebra)
- [Scenario and workflow organization](#scenario-and-workflow-organization)

## Terrain and Raster Analysis

Read and write single-band GeoTIFFs with explicit `GeoReference` metadata,
compute terrain slope, aspect, and hillshade, compare and align rasters
between grids, and convert between raster pixel, projected, and lunar
longitude/latitude coordinates.

- `ls.read_geotiff`, `ls.write_geotiff`
- `ls.slope`, `ls.aspect`, `ls.hillshade`
- `ls.same_grid`, `ls.require_same_grid`, `ls.align`
- `GeoReference` coordinate conversion

## Region and Candidate Screening

Label, measure, filter, and outline connected regions in raster masks, and
combine terrain and illumination thresholds into spatially coherent
landing-site candidates.

- `ls.label_regions`, `ls.region_sizes`, `ls.filter_regions_by_size`, `ls.find_borders`

## Temporal Data

Work with UTC time ranges, in-memory `(time, y, x)` cubes, and file-backed
timestamped GeoTIFF series, including streaming reductions that do not
materialize a full cube.

- `ls.times`, `ls.iter_times`, `ls.TemporalCube`
- `ls.temporal_mean`, `ls.temporal_min`, `ls.temporal_max`, `ls.temporal_std`
- `ls.write_temporal_cube`, `ls.open_temporal_cube`, `ls.TemporalGeoTiffSeriesWriter`

## Celestial Geometry (SPICE)

Compute Sun and Earth local-frame vector histories and azimuth/elevation
angles at a lunar surface point, and plot body positions and paths against
terrain horizons.

- `ls.body_vectors_ned`, `ls.body_azimuth_elevation`
- `ls.body_azimuth_elevation_over_horizon`
- `ls.plot_body_elevation`, `ls.plot_body_elevations`, `scenario.plot_horizon`

## Horizons and Lighting Products

Generate CUDA-accelerated terrain horizon tiles, read stored horizons, and
generate lightmaps, permanent-shadow maps, Sun/Earth terrain-relative
elevation, safe-haven, and landed mission-duration products.

- `ls.generate_horizons`, `scenario.generate_horizons`
- `ls.generate_lightmap`, `ls.generate_psr`
- `ls.generate_sun_elevation`, `ls.generate_earth_elevation`, `ls.generate_safe_havens`
- `ls.mission_duration_from_sunlight` and the other mission-duration products

## Mission and Mobility Planning

Plan static (Dijkstra/A*) and dynamic rover paths on projected raster grids,
optionally including slope/slip travel-time factors and battery
state-of-charge constraints with solar and Earth-elevation providers.

- `ls.trajectory.static_travel_time`, `ls.trajectory.static_path`
- `ls.trajectory.dynamic_path`, `ls.trajectory.soc_path`

## Map Algebra

Work with eager `Raster` values and lazy `RasterExpression` graphs that carry
explicit validity masks, units, dtypes, and numeric policies, plus temporal
map algebra over file-backed series.

- `ls.Raster`, `ls.map_algebra` (`ma`)
- `ma.source`, `ma.compute`, `ma.write`, `ma.explain`, `ma.plan`
- `ma.temporal_source`, `ma.temporal_mean`

## Scenario and Workflow Organization

Organize a lunar site through a filesystem-safe `Scenario` facade with
canonical paths, resumable and cancellable product jobs, and map-product
catalog handling.

- `ls.open_scenario`, `scenario.*` product conveniences
- `ls.load_map_product_catalog`, `ls.search_map_products`, `ls.download_map_product`

---

## Generate CUDA-Accelerated Terrain Horizon Tiles from Primary + Surrounding DEMs

A terrain horizon records, for every observer pixel in a DEM, the maximum
terrain elevation angle around that pixel as a function of azimuth. Horizon
tiles are the input to every downstream lighting and visibility product, so
they are usually the first expensive calculation in a site analysis.

Generation is deliberately CUDA-only: the production implementation requires
a supported NVIDIA GPU and the `cuda` installation profile, and it raises a
structured `CudaError` rather than falling back to a CPU generator. The first
DEM defines the output grid; later DEMs extend the surrounding terrain
coverage. The cumulative horizon from an earlier DEM participates in
hierarchy culling for later DEMs.

```python
import lunarscout as ls

horizons = ls.generate_horizons(
    "/data/site/horizons",
    [
        "/data/site/dem.tif",
        "/data/regional-dem.tif",
    ],
    observer_height_m=0.0,
    compress=True,
    verbose=True,
)

print(horizons)
```

The `Scenario` convenience places the canonical primary DEM first
automatically:

```python
import lunarscout as ls

scenario = ls.open_scenario("/data/site")

horizons = scenario.generate_horizons(
    surrounding_dems=["surrounding/regional.tif"],
    compress=True,
)
```

Each output tile is a fixed 128 by 128 pixel patch with 1,440 `float32`
azimuth samples per pixel at 0.25-degree spacing (sample 0 is north).
Structurally complete tiles are reused across runs by default; pass
`overwrite=True` to regenerate them. New tiles are staged and atomically
published, so an interrupted run is resumable and a failed overwrite
preserves the prior completed tile.

> **Requirements.** A supported NVIDIA device and driver plus the `cuda`
> installation profile. Reading stored horizons and running downstream
> products do not require CUDA.
