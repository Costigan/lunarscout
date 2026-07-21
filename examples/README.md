# Lunarscout Examples

These scripts demonstrate the public `lunarscout` API in increasing order of
scope. They are ordinary Python programs; no notebook is required.

## Setup

Use the repository virtual environment:

```bash
cd /e/projects/lunarscout
export PYTHONPATH="$PWD/src"
```

Deterministic examples create a synthetic scenario under
`/tmp/lunarscout_examples` by default. Select another location with
`--workspace` or `LUNARSCOUT_EXAMPLE_WORKSPACE`.

```bash
PYTHONPATH="$PWD/src" .venv/bin/python examples/01_geotiff_and_coordinates.py
```

All deterministic scripts are safe to run repeatedly; their named output
GeoTIFFs are overwritten. The timestamped fixture series is reused when it
is already complete.

Examples are designed to be run from the repository root.  They share a
common support module at `_example_support.py`.

## Data requirements

| Requirement | Examples |
|-------------|----------|
| None (fully synthetic) | 01–08, 10–11 |
| SPICE kernel download (first use) | 11–12, 14–16 |
| Synthetic horizon data (download on first use from GitHub Releases) | 12–13 |
| Real scenario with `dem.tif` and `horizons/` | 14–16 |
| NVIDIA GPU | 15 (horizon generation), 14/16 (for `--backend cuda`) |

## Script index

| Script | Demonstrates | GPU | Data |
|--------|-------------|-----|------|
| `01_geotiff_and_coordinates.py` | GeoTIFF I/O, metadata, scalar and array coordinate conversion | No | Synthetic |
| `02_terrain_products.py` | Slope, aspect, and hillshade | No | Synthetic |
| `03_region_filtering.py` | Connected-region labeling, size filtering, cleanup, border extraction | No | Synthetic |
| `04_alignment.py` | Grid comparison and explicit GDAL alignment | No | Synthetic |
| `05_temporal_cube.py` | UTC time range, `TemporalCube`, and time-axis reducers | No | Synthetic |
| `06_file_backed_series.py` | Timestamped TIFF series, layer lookup, metadata | No | Synthetic |
| `07_incremental_writer.py` | `TemporalGeoTiffSeriesWriter` — incrementally build a file-backed series without a `TemporalCube` | No | Synthetic |
| `08_streaming_reductions.py` | File-backed streaming temporal reducers | No | Synthetic |
| `09_qgis_vrt.py` | Individual TIFF and multi-band VRT inspection with rasterio | No | Synthetic |
| `10_landing_site_screening.py` | Combined terrain, illumination, and region screening | No | Synthetic |
| `11_spice_vectors.py` | SPICE Sun/Earth NED vectors, azimuth/elevation, DataFrame helpers | No | SPICE |
| `12_body_and_horizon_plots.py` | Body elevation plots, horizon plots, body path overlays, zoomed body paths | No | Horizon data + SPICE |
| `13_synthetic_lightmap.py` | CPU lightmap from explicit vectors on synthetic DEM with pregenerated horizons | No | Horizon data |
| `15_python_psr.py` | PSR generation on a real scenario | Optional (Cuda default) | Real scenario + SPICE |
| `16_generate_horizons.py` | Resumable CUDA horizon generation from one or more DEMs | **Required** | User-provided DEMs |
| `17_downstream_products.py` | Lightmap, PSR, Sun/Earth elevation, safe havens, and four mission-duration products | Optional | Real scenario + SPICE |
| `18_map_algebra_screening.py` | Map-algebra terrain-lighting screening with validity, scoring, and GeoTIFF output | No | Synthetic |
| `19_map_algebra_focal.py` | Map-algebra focal smoothing, morphology opening, and distance fields | No | Synthetic |
| `20_map_algebra_temporal.py` | Temporal map algebra: time-series reduction composed with spatial constraints | No | Synthetic |

## Synthetic horizon data

Examples 12 and 13 require a synthetic 256×256 DEM and four pregenerated
128×128 horizon tiles. This data is downloaded automatically on first use from
GitHub Releases, verified against checked-in SHA-256 checksums, and cached under
`$LUNARSCOUT_EXAMPLE_DATA_DIR` (default: `$XDG_DATA_HOME/lunarscout/examples/`,
fallback: `~/.local/share/lunarscout/examples/`).

- **First run:** ~37 MB download. No SPICE kernels needed.
- **Subsequent runs:** instant (cache hit).
- **Download failure:** the example exits with an actionable message.

To skip the download, set `LUNARSCOUT_EXAMPLE_DATA_DIR` to a directory that
already contains `synthetic_horizon_scenario/` with `dem.tif` and `horizons/`.

## Python/Numba PSR example

`15_python_psr.py` uses the public `ls.generate_psr()` facade. Run it with a
scenario containing `dem.tif` and `horizons/`:

```bash
PYTHONPATH="$PWD/src" .venv/bin/python examples/15_python_psr.py \
  --scenario /data/mons_mouton \
  --output analysis/mons-mouton-psr.tif \
  --backend auto
```

The defaults use `--backend cuda` and read `/e/lunar_analyst_scenarios/mons-mouton`.
The calculation uses exact six-hour Moon-ME Sun vectors from 1970-01-01 through
2044-01-01. Pass `--overwrite` only when replacing an already completed output
is intended. Its progress callback receives the fraction of fully processed
horizon tiles and prints approximately once per percentage point, including
elapsed minutes, estimated remaining minutes, and estimated local completion
time.

## Horizon generation example

`16_generate_horizons.py` generates CUDA horizon tiles from one or more DEMs.
Insert your DEM paths before running:

```bash
PYTHONPATH="$PWD/src" .venv/bin/python examples/16_generate_horizons.py \
  --primary-dem /data/dem.tif \
  --output /data/horizons
```

Add `--surrounding-dem /data/regional.tif` for additional terrain coverage.
Use `--overwrite` to regenerate existing tiles.

## Downstream product example

`17_downstream_products.py` accepts an existing scenario containing `dem.tif`
and `horizons/`. It defaults to one CPU lightmap:

```bash
PYTHONPATH="$PWD/src" .venv/bin/python examples/17_downstream_products.py \
  /data/mons_mouton --backend cpu --product lightmap
```

Pass `--product` repeatedly or use `--product all` to generate PSR, Sun/Earth
elevation, safe havens, and all four mission-duration variants. `--backend
auto` uses CUDA when available and otherwise falls back to CPU. `--backend
cuda` never falls back.

The safe-haven example uses one output band per calendar month. Each `float32`
value is the longest contiguous low-Sun duration, in hours. Mission-duration
products use one candidate-start interval and also write hours. These examples
apply no slope suitability, battery, thermal, or traverse policy.

## QGIS

Run `06_file_backed_series.py`, then `09_qgis_vrt.py`. The latter prints:

- an individual timestamp TIFF that QGIS can open directly; and
- `series.vrt`, where VRT band `n + 1` corresponds to Python layer index `n`.

The VRT carries timestamp band descriptions but does not automatically
configure the QGIS temporal controller.

A reported pixel size of `1, -1` is expected for the north-up affine transform.
Set the project CRS from the raster layer; QGIS cannot and should not transform
the lunar custom CRS to the default WGS 84 project CRS.

## Historical benchmark

`benchmarks/timeseries_two_file_prototype.py` is retained as storage-evaluation
evidence, not as a supported Lunarscout product. It requires `h5py` and
`hdf5plugin` (not installed with Lunarscout).
