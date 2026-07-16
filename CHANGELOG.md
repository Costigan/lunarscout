# Changelog

## Versioning policy

Lunarscout uses Semantic Versioning. Before 1.0, public APIs are provisional and breaking changes may occur in minor releases. Intentional breaking changes must be recorded here. Patch releases should not intentionally break documented behavior. The 1.0.0 milestone is reserved for a documented stable API surface and standalone native runtime story.

## Unreleased

- Defined the private Python/NumPy horizon host/device data contract, including
  checked precision conversions, dense segment tensors, flattened pyramid
  storage, kernel/configuration validation, slope-buffer semantics, verified
  oracle loading, and Phase 1 artifact round-trip tests. CUDA and horizon
  algorithm implementation remain deferred to later evaluation phases.
- Established Phase 0 and Phase 1 evidence for evaluating a full Python/Numba
  horizon-generator port: reproducible C# production baselines and warm
  multi-patch benchmarks, GPU-memory sampling, independent reference-ray
  oracles, synthetic and real-terrain fixture manifests, immutable NPZ test
  artifacts, per-DEM and final horizon buffers, and selected CUDA hierarchy
  traversal traces. No horizon algorithm has been ported yet.
- Created standalone Lunarscout repository skeleton from Lunar Analyst packages/lunarscout.
- Replaced the internal architecture guide with a draft user guide covering purpose, installation, usage, maturity, architecture, examples, and roadmap stubs.
- Added a Lunarscout-specific `requirements.in` and removed inherited Lunar Analyst application dependencies.
- Updated package metadata so core installs include raster/geospatial, native bridge, and HDF5 dependencies.
- Verified the new local virtual environment with `pytest -q`: 195 passed, 1 skipped.
- Added map product catalog support, including product/region models, catalog loading, ordered text search, scenario-safe naming, download-directory resolution, file download helper, public exports, and tests.
- Added lunar map product support assets and utility scripts, including a south-pole overview GeoTIFF, a product catalog JSON file, catalog maintenance/download scripts, and Git LFS tracking for `data/product_overview.tif`.
- Added `ls.GenerateHorizons(...)` and `ls.native.GenerateHorizons(...)` as Python.NET wrappers around `QuadTreeHorizonGenerator.GenerateHorizonsForPatches`, with skip-existing patch filtering, compression selection, progress callbacks, cancellation checks, and tests.
- Added `scripts/run_generate_horizons.py` as an editable local runner for manually validating native horizon generation.
- Added `AGENTS.md` with project-specific guidance for future coding agents.
- Updated the native MaxRev GDAL package-version test to locate the extracted Lunarscout repository layout and scan `native/` projects.
- Added SPICE-backed lunar local-frame APIs for `LonLat`, inclusive datetime iteration, Sun/Earth NED vectors, azimuth/elevation histories, pandas DataFrames, and matplotlib elevation plots.
- Added `ls.spice` kernel helpers for furnishing, lazy default loading, reload/unload/clear state, NAIF kernel download/cache, generated meta-kernels, and SHA-256 verification from the checked-in default kernel manifest.
- Added default SPICE kernel manifests under `data/spice/` and package data, plus implementation tracking in `docs/spice-local-frame-api-plan.md`.
- Updated the user guide with SPICE kernel setup, local NED frame conventions, vector/angle APIs, DataFrame helpers, and plotting examples.
- Added `scripts/get_link_tree.py` for recursive same-site HTML link discovery.
- Updated the lunar map product GUI to display overview-map CRS and longitude/latitude mouse coordinates and copy either coordinate pair to the clipboard with keyboard shortcuts.
- Changed the canonical Scenario horizon directory from `lighting/horizons` to root-level `horizons`, with matching tests, examples, README, and guide updates.
- Added Scenario canonical path helpers for `root_path()`, `hillshade_path()`, `slope_path()`, `aspect_path()`, and `roughness_path()`.
- Added native GDAL-backed terrain product generation through `Scenario.create_hillshade()`, `create_slope()`, `create_aspect()`, and `create_roughness()`, backed by a new `moonlib.TerrainProducts` static helper.
- Added `Scenario.generate_horizons()` as a scenario-aware wrapper around native horizon generation, including `dem_paths` and `surrounding_dems` handling with scenario-relative DEM resolution.
- Added Python horizon tile access helpers on `Scenario`, including patch coordinate conversion, horizon file lookup, `.bin`/`.cbin` single-pixel horizon reads, one-file handle caching, and cache close support.
- Added Scenario longitude/latitude DEM pixel lookup and horizon plotting helpers, including centered azimuth windows and empty azimuth/elevation axes.
- Added Scenario Sun/Earth horizon plot overlays for center points, apparent limb markers, center paths, and upper/lower limb paths; Sun defaults to gold and Earth defaults to blue.
- Suppressed the noisy pyproj WKT-to-PROJ.4 information-loss warning during Lunarscout GeoTIFF metadata reads.
- Expanded `docs/USER_GUIDE.md` with Scenario path, native terrain, horizon generation/access, horizon plotting, and Sun/Earth overlay examples.
- Updated SPICE body geometry helpers and plotting helpers to accept `TimeRange` values returned by `ls.times(...)` directly.
- Changed body path limb rendering from upper/lower dashed lines to a translucent filled limb band, while preserving explicit Matplotlib style overrides.
- Added `ls.body_azimuth_elevation_over_horizon(...)` and `Scenario.body_azimuth_elevation_over_horizon(...)` for azimuth plus elevation over an interpolated 1440-sample horizon.
- Added `horizon=` support to `ls.plot_body_elevations(...)` and added `Scenario.plot_body_elevations(...)` with `over_horizon=True` to fetch scenario horizons automatically.
- Expanded `docs/USER_GUIDE.md` with a succinct function overview table covering root functions, Scenario methods, and file-backed temporal object methods.
