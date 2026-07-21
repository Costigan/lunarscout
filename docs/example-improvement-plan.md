# Example Improvement Plan

[Implementation of this plan has been completed.]

## Progress

- [x] Delete stale `.pyc` files
- [x] Fix `examples/README.md` stale path
- [x] Write N3: incremental temporal series writer
- [x] Write N1: SPICE body vectors and azimuth/elevation
- [x] Renumber existing examples 00–10 to 01–10
- [x] Move `14_timeseries_two_file_prototype.py` out of numbered sequence
- [x] Design and generate synthetic 256×256 DEM with positive and negative features
- [x] Generate four horizon tiles from synthetic DEM (GPU required)
- [x] Upload `synthetic-horizon-data-v1.tar.gz` as GitHub Release asset
- [x] Add SHA-256 manifest to `examples/data/synthetic_horizon_manifest.json`
- [x] Implement `ensure_synthetic_horizon_scenario()` in `_example_support.py`
- [x] Write N2: body elevation and horizon plotting
- [x] Write N4: synthetic lightmap on CPU
- [x] Fix `16_generate_horizons.py` with arg parser
- [x] Update `examples/README.md` with new numbering, data table, prereqs

## Goals

- [x] Every user-guide-documented capability has a runnable example.
- [x] CPU-only examples work self-contained on synthetic data.
- [x] GPU examples clearly state prerequisites and fail with actionable messages.
- [x] Numbering is contiguous and grouped by domain.

## Current gaps

| Gap | Detail |
|-----|--------|
| No SPICE example | `body_vectors_ned`, `body_azimuth_elevation`, `body_azimuth_elevation_over_horizon`, `iter_times` undocumented by example |
| No plotting example | `plot_body_elevation`, `plot_body_elevations`, `plot_horizon`, `plot_body_position`, `plot_body_path`, `plot_zoomed_body_path` |
| No incremental writer example | `TemporalGeoTiffSeriesWriter` context manager |
| No CPU synthetic lighting example | Lightmap, PSR, or elevation on synthetic data (runnable without GPU or real DEM) |
| `16_generate_horizons.py` broken UX | Hardcoded paths, no arg parser |
| Numbering gaps | 07, 08, 11, 12, 13 missing; stale `.pyc` files remain |
| `examples/README.md` stale path | References old `lunarscout-numba-horizon` repo name |

## Proposed new examples

### N1. SPICE body vectors and azimuth/elevation

Pure CPU. Demonstrates:

- `ls.utc_datetime()`, `ls.times()`, `ls.iter_times()`
- `ls.body_vectors_ned()` and the DataFrame variant
- `ls.body_azimuth_elevation()` and the DataFrame variant
- Local NED frame convention (x = north, y = east, z = down)
- Prints sample vectors and angles as text; no plot dependency

Data: no DEM or scenario needed. Uses a single `LonLat` point.

### N2. Body elevation over time and horizon plotting

Pure CPU. Demonstrates:

- `ls.plot_body_elevation(point, "sun", times)` — single body
- `ls.plot_body_elevations(point, ["sun", "earth"], times)` — multiple bodies
- `scenario.plot_azimuth_elevation_axes()` — empty polar-style axis
- `scenario.plot_horizon(point)` — with `center_azimuth` parameter
- `scenario.plot_body_position(ax, point, "sun", time)` — center and limb styles
- `scenario.plot_body_path(ax, point, "sun", times)` — center, limbs, center_and_limbs
- `scenario.plot_zoomed_body_path(point, bodies, times)` — equal-scale zoomed view

Uses `ensure_synthetic_horizon_scenario()` to obtain the 256×256 DEM and
four pregenerated horizon tiles from the GitHub Release asset.  Saves plots
to `analysis/plots/` as PNG files.

### N3. Incremental temporal series writer

Pure CPU. Demonstrates:

- `TemporalGeoTiffSeriesWriter` as context manager
- Writing layers one at a time without building a `TemporalCube`
- `progress_callback` and `cancellation_requested` arguments
- Reading back individual layers from the completed series

Data: synthetic. Reuses `_example_support` DEM.

### N4. Synthetic lightmap on CPU

Pure CPU. Demonstrates:

- `ls.generate_lightmap()` with `backend="cpu"`
- Explicit Moon-ME vectors passed as `sun_vectors_m` (avoids SPICE import)
- `ls.read_geotiff()` to inspect output
- Timestamp bands, `uint8` encoding, validity masks

Uses `ensure_synthetic_horizon_scenario()` to obtain data.

## Proposed reorganization

Current  | Proposed | Title
-------- | -------- | -----
00 | 01 | GeoTIFF I/O and coordinate conversion
01 | 02 | Slope, aspect, and hillshade
02 | 03 | Connected-region labeling and filtering
03 | 04 | Explicit grid comparison and raster alignment
04 | 05 | In-memory temporal cubes and time-axis reductions
05 | 06 | File-backed timestamped GeoTIFF series
-- | **07** | **Incremental temporal series writer** (new N3)
06 | 08 | Streaming reductions over a file-backed series
09 | 09 | QGIS VRT inspection *(moves up one slot)*
10 | 10 | Landing-site screening *(moves up one slot)*
-- | **11** | **SPICE body vectors and azimuth/elevation** (new N1)
-- | **12** | **Body elevation and horizon plotting** (new N2)
-- | **13** | **Synthetic lightmap on CPU** (new N4)
15 | 14 | PSR on a real scenario
16 | 15 | Horizon generation
17 | 16 | Downstream products on a real scenario
14 | -- | *(removed from numbered sequence; kept as `benchmarks/` or similar)*

## Fixes to existing examples

### `16_generate_horizons.py`

- [x] Add `--primary-dem`, `--surrounding-dem`, `--output`, `--observer-height-m`,
  and `--overwrite` arguments.  Failure paths print actionable diagnostics
  (no GPU, missing DEM, incompatible driver).

### `14_timeseries_two_file_prototype.py`

- [x] Moved to `benchmarks/timeseries_two_file_prototype.py`. It is not a
  Lunarscout API example; it is a storage evaluation benchmark.

### `examples/README.md`

- [x] Fixed the stale `cd` path.
- [x] Added a data-requirements table listing which examples need a GPU, SPICE
  network access, a real scenario, or pregenerated synthetic horizons.
- [x] Mentioned the `_example_support.py` dependency and that examples are
  designed to be run from the examples directory.

### Stale `__pycache__`

- [x] Deleted `.pyc` files for removed scripts (07, 08, 11, 12, 13).

## Synthetic scenario data for lighting examples

Examples N2 (plotting) and N4 (synthetic lightmap) need a synthetic
256×256 DEM with both positive and negative terrain features (a bowl plus
ridges/cones) and four pregenerated 128×128 horizon tiles.  The DEM can be
generated on the fly, but horizon generation is CUDA-only, so the tiles
must be pre-built and distributed.

**Approach: GitHub Release asset with first-use download.**

- A single `.tar.gz` asset attached to a GitHub Release (e.g.
  `synthetic-horizon-data-v1`), containing four `.cbin` files plus a
  pre-generated `dem.tif` (bundling the DEM avoids any grid mismatch
  between generated DEM and pre-built horizons).
- Estimated size: 15–40 MB compressed.
- A function in `_example_support.py`, `ensure_synthetic_horizon_scenario(workspace)`,
  downloads on first use, verifies a checked-in SHA-256 manifest, unpacks
  into `$LUNARSCOUT_EXAMPLE_DATA_DIR` (default:
  `$XDG_DATA_HOME/lunarscout/examples/`), and copies into the workspace.
  Subsequent runs use the cached copy.  Network failure or checksum
  mismatch is non-fatal — the example exits with an actionable message.
- The checked-in manifest lives in `examples/data/synthetic_horizon_manifest.json`.
- This data is needed only by N2 and N4.  Other examples that use horizons
  (real-scenario PSR, downstream products) do not call this function; they
  use real data from the user's scenario directory.

### Manifest format

```json
{
  "version": 1,
  "asset_name": "synthetic-horizon-data-v1.tar.gz",
  "release_tag": "synthetic-horizon-data-v1",
  "files": {
    "dem.tif": {"sha256": "..."},
    "horizons/00000/horizon_00000_00000_000.cbin": {"sha256": "..."},
    "horizons/00000/horizon_00000_00128_000.cbin": {"sha256": "..."},
    "horizons/00128/horizon_00128_00000_000.cbin": {"sha256": "..."},
    "horizons/00128/horizon_00128_00128_000.cbin": {"sha256": "..."}
  }
}
```

### Tasks

- [ ] Design and generate the synthetic DEM with positive and negative
  features (256×256, south polar stereographic, bowl + ridges/cones).
- [ ] Generate the four horizon tiles with `ls.generate_horizons()` on a
  GPU machine.
- [ ] Upload the `.tar.gz` as a GitHub Release asset with tag
  `synthetic-horizon-data-v1`.
- [ ] Add the SHA-256 manifest to `examples/data/synthetic_horizon_manifest.json`.
- [ ] Implement `ensure_synthetic_horizon_scenario()` in `_example_support.py`.

## Future real-data examples

After the synthetic examples are working, add examples that download and
process real LOLA digital elevation models.  These are separate from the
synthetic-data path:

- [ ] **N5. Download and prepare a LOLA DEM** — download a real LOLA polar
  DEM tile, read with `ls.read_geotiff()`, compute slope/aspect/hillshade,
  and write a prepared scenario.
- [ ] **N6. Generate horizons and products from a LOLA DEM** — takes the
  prepared scenario, runs `ls.generate_horizons()` (CUDA required), then
  generates one or more downstream products.

These examples will use the same GitHub Release asset approach for LOLA DEM
data (larger files, potentially hundreds of MB), or consume user-provided
DEMs via command-line arguments.  Design separately once the synthetic
examples are complete.

## Order of implementation

- [x] 1. Delete stale `.pyc` files and fix `examples/README.md` path.
- [x] 2. Write new example N3 (incremental writer) — no new data dependency.
- [x] 3. Write new example N1 (SPICE vectors) — no new data dependency.
- [x] 4. Renumber existing examples 00–10 to 01–10 (and update internal references).
- [x] 5. Move 14 out of the numbered sequence.
- [x] 6. Design and generate the synthetic 256×256 DEM on a GPU machine.
- [x] 7. Generate the four horizon tiles and upload
  `synthetic-horizon-data-v1.tar.gz` as a GitHub Release asset.
- [x] 8. Add the SHA-256 manifest and implement `ensure_synthetic_horizon_scenario()`.
- [x] 9. Write N2 (plotting) and N4 (synthetic lightmap).
- [x] 10. Fix 16 (horizon generation) with arg parser.
- [x] 11. Update `examples/README.md` with new numbering, data requirements table,
  and prerequisite guidance.
