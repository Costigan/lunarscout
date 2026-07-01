# Changelog

## Versioning policy

Lunarscout uses Semantic Versioning. Before 1.0, public APIs are provisional and breaking changes may occur in minor releases. Intentional breaking changes must be recorded here. Patch releases should not intentionally break documented behavior. The 1.0.0 milestone is reserved for a documented stable API surface and standalone native runtime story.

## Unreleased

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
