# Changelog

## Versioning policy

Lunarscout uses Semantic Versioning. Before 1.0, public APIs are provisional and breaking changes may occur in minor releases. Intentional breaking changes must be recorded here. Patch releases should not intentionally break documented behavior. The 1.0.0 milestone is reserved for a documented stable API surface and standalone native runtime story.

## Unreleased

- Created standalone Lunarscout repository skeleton from Lunar Analyst packages/lunarscout.
- Replaced the internal architecture guide with a draft user guide covering purpose, installation, usage, maturity, architecture, examples, and roadmap stubs.
- Added a Lunarscout-specific `requirements.in` and removed inherited Lunar Analyst application dependencies.
- Updated package metadata so core installs include raster/geospatial, native bridge, and HDF5 dependencies.
- Verified the new local virtual environment with `pytest -q`: 195 passed, 1 skipped.
