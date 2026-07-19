# Lunarscout Examples

These scripts demonstrate the public `lunarscout` API in increasing order of
scope. They are ordinary Python programs rather than notebook-only fragments.

## Setup

Use the repository-managed environment and make the package source available:

```bash
cd /e/projects/lunarscout-numba-horizon
export PYTHONPATH="$PWD/src"
export LUNARSCOUT_EXAMPLE_PYTHON=/e/projects/lunarscout/.venv/bin/python
```

Deterministic examples create a synthetic scenario under
`/tmp/lunarscout_examples` by default. Select another location with
`--workspace` or `LUNARSCOUT_EXAMPLE_WORKSPACE`.

```bash
"$LUNARSCOUT_EXAMPLE_PYTHON" examples/00_geotiff_and_coordinates.py
```

All deterministic scripts are safe to run repeatedly; their named output
GeoTIFFs are overwritten. The timestamped fixture series is reused when it is
already complete.

## Script Index

| Script | Demonstrates | Requirement |
| --- | --- | --- |
| `00_geotiff_and_coordinates.py` | GeoTIFF I/O, metadata, scalar and array coordinate conversion | CPU |
| `01_terrain_products.py` | Slope, aspect, and hillshade | CPU |
| `02_region_filtering.py` | Labels, sizes, filtering, cleanup, and borders | CPU |
| `03_alignment.py` | Grid comparison and explicit GDAL alignment | CPU |
| `04_temporal_cube.py` | UTC time range, `TemporalCube`, and reducers | CPU |
| `05_file_backed_series.py` | Timestamped TIFF series, lookup, and metadata | CPU |
| `06_streaming_reductions.py` | File-backed streaming reductions | CPU |
| `09_qgis_vrt.py` | Individual TIFF and multi-band VRT inspection | CPU |
| `10_landing_site_screening.py` | Combined terrain, illumination, and region screening | CPU |
| `14_timeseries_two_file_prototype.py` | Historical BigTIFF/HDF5 storage benchmark | Manual HDF5 packages |
| `15_python_psr.py` | Python/Numba Mons Mouton permanent-shadow GeoTIFF generation | NVIDIA GPU |
| `16_generate_horizons.py` | Public resumable Python/Numba horizon generation | NVIDIA GPU |
| `17_downstream_products.py` | Public lightmap, PSR, Sun/Earth elevation, safe-haven, and four mission-duration products | CPU or NVIDIA GPU |

## Python/Numba PSR Example

`15_python_psr.py` uses the public `ls.generate_psr()` facade. Run it from the
worktree with the shared repository virtual environment:

```bash
cd /e/projects/lunarscout-numba-horizon
PYTHONPATH="$PWD/src" \
  /e/projects/lunarscout/.venv/bin/python examples/15_python_psr.py
```

The defaults read `/e/lunar_analyst_scenarios/mons-mouton` and write
`examples/mons-mouton-psr.tif`. The calculation uses exact six-hour Moon-ME Sun
vectors from 1970-01-01 through 2044-01-01 and the CUDA backend. It resumes a
compatible interrupted staging product. Pass `--overwrite` only when replacing
an already completed output is intended. Its progress callback receives the
fraction of fully processed horizon tiles and prints approximately once per
percentage point, including elapsed minutes, estimated remaining minutes, and
the estimated local completion time. A GUI or notebook can pass the same
fraction directly to its progress-bar widget.

## Python/Numba Horizon Example

Edit the DEM paths in `16_generate_horizons.py`, then run it in an environment
with a compatible NVIDIA GPU. The first DEM defines the output grid and each
following DEM extends terrain coverage. The example uses only the public
`ls.generate_horizons()` API, resumes structurally complete tiles, and exits
with the CUDA diagnostic reason when no compatible device is available.

## Downstream Product Example

`17_downstream_products.py` accepts an existing scenario containing `dem.tif`
and `horizons/`. It defaults to one CPU lightmap, so it works with the base
installation and does not probe CUDA:

```bash
python examples/17_downstream_products.py /data/mons_mouton \
  --backend cpu --product lightmap
```

Pass `--product` repeatedly or use `--product all` to generate PSR, separate
Sun- and Earth-center terrain-relative elevation products, safe havens, and all
four mission-duration variants. `--backend auto` uses CUDA when the
`lunarscout[cuda]` profile and a compatible device are available, otherwise it
falls back to CPU. `--backend cuda` never falls back.

The safe-haven example uses one output band per maximal Earth-outage interval.
Each `float32` value is the longest contiguous low-Sun duration during that
outage, in hours. Mission-duration products use one candidate-start interval
covering the requested time range and also write hours. These examples apply
no slope suitability, battery, thermal, or traverse policy.

## QGIS

Run `05_file_backed_series.py`, then `09_qgis_vrt.py`. The latter prints:

- an individual timestamp TIFF that QGIS can open directly; and
- `series.vrt`, where VRT band `n + 1` corresponds to Python layer index `n`.

The VRT carries timestamp band descriptions but does not automatically
configure the QGIS temporal controller.

A reported pixel size of `1, -1` is expected for the north-up affine transform.
Set the project CRS from the raster layer; QGIS cannot and should not transform
the lunar custom CRS to the default WGS 84 project CRS.

## Two-File Time-Series Prototype

`14_timeseries_two_file_prototype.py` creates a small synthetic lighting cube
and writes two synchronized products:

This is retained as historical storage-evaluation evidence, not as a supported
Lunarscout product or advertised package capability. Its `h5py` and
`hdf5plugin` requirements are not installed with Lunarscout. The accepted
public lightmap and mission-duration pipelines write tiled, compressed,
resumable BigTIFF products.

- `shadow_maps.tif`: BigTIFF with one band per time step and 128 x 128 tiled
  blocks for map-frame access.
- `light_curves_chunk*.h5`: HDF5 files with shape `(y, x, time)` and tested
  spatial chunk sizes for compressed light-curve access.

The default run is intentionally small for local smoke testing. Increase
`--width`, `--height`, `--time-count`, and `--hdf5-chunk-xy` to benchmark
representative local NVMe or CephFS behavior.
