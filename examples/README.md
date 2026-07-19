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

Non-native examples create a deterministic synthetic scenario under
`/tmp/lunarscout_examples` by default. Select another location with
`--workspace` or `LUNARSCOUT_EXAMPLE_WORKSPACE`.

```bash
"$LUNARSCOUT_EXAMPLE_PYTHON" examples/00_geotiff_and_coordinates.py
```

All deterministic scripts are safe to run repeatedly; their named output
GeoTIFFs are overwritten. The timestamped fixture series is reused when it is
already complete.

## Script Index

| Script | Demonstrates | Native runtime |
| --- | --- | --- |
| `00_geotiff_and_coordinates.py` | GeoTIFF I/O, metadata, scalar and array coordinate conversion | No |
| `01_terrain_products.py` | Slope, aspect, and hillshade | No |
| `02_region_filtering.py` | Labels, sizes, filtering, cleanup, and borders | No |
| `03_alignment.py` | Grid comparison and explicit GDAL alignment | No |
| `04_temporal_cube.py` | UTC time range, `TemporalCube`, and reducers | No |
| `05_file_backed_series.py` | Timestamped TIFF series, lookup, and metadata | No |
| `06_streaming_reductions.py` | File-backed streaming reductions | No |
| `07_native_sun_fraction.py` | Native solar fraction with explicit memory storage | Yes |
| `08_native_horizon_margins.py` | File-backed native Sun/Earth horizon margins | Yes |
| `09_qgis_vrt.py` | Individual TIFF and multi-band VRT inspection | No |
| `10_landing_site_screening.py` | Combined terrain, illumination, and region screening | No |
| `11_native_end_to_end_validation.py` | Native memory/file parity, lifecycle, integrity, and resource evidence | Yes |
| `12_native_performance_benchmark.py` | Representative native series generation, reads, reductions, and resource measurements | Yes |
| `13_native_psr.py` | Native permanent-shadow byte-mask generation | Yes |
| `14_timeseries_two_file_prototype.py` | Two-file BigTIFF/HDF5 time-series storage prototype and access benchmark | No |
| `15_python_psr.py` | Python/Numba Mons Mouton permanent-shadow GeoTIFF generation | NVIDIA GPU |
| `16_generate_horizons.py` | Public resumable Python/Numba horizon generation | NVIDIA GPU |

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

## Historical Managed-Runtime Examples

The following `*_native_*` scripts are retained only as migration evidence for
the superseded C#/Python.NET implementation. They are not part of the public
Python/Numba API, installed package, or `0.1.0rc1` acceptance. New code should
use the public horizon and product functions demonstrated above.

Native examples require a real scenario containing `dem.tif` and
`horizons`, plus the configured Python.NET, .NET, moonlib, CSPICE,
GDAL, and SPICE data runtime.

The current source-tree native runtime bundles a newer GDAL dependency set
than the host Python GDAL. Select that bundle before Python starts, and preload
the host `libxml2` to keep the host SpatiaLite dependency compatible:

```bash
export MOONLIB_RUNTIME_DIR="$PWD/native/new_horizon/moonlib/bin/Debug/net10.0/linux-x64"
export LD_LIBRARY_PATH="$MOONLIB_RUNTIME_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export LD_PRELOAD="/lib/x86_64-linux-gnu/libxml2.so.2${LD_PRELOAD:+:$LD_PRELOAD}"
export LUNARSCOUT_EXAMPLE_SCENARIO=/path/to/lunar_scenario
.venv/bin/python examples/07_native_sun_fraction.py \
  --start 2027-01-01T00:00:00Z \
  --stop 2027-01-01T02:00:00Z \
  --step-hours 1
```

The memory example prints its exact allocation estimate before native startup.
The file-backed example needs temporary uncompressed scratch space plus final
output space; Lunarscout checks both before starting native computation.

Use `--overwrite` only when replacing the named native outputs is intended.
The scripts do not register products, publish layers, or mutate `scenario.db`.
Production long-running native work remains assigned to the isolated compute
worker; these direct scripts are local API and native-boundary validation tools.

`13_native_psr.py` wraps `MoonlibBridge.GeneratePermanentShadowMap`. The
current native contract samples the Sun center every six hours from 1970-01-01
through 2044-01-01 at zero observer elevation. Output is a native-grid uint8
GeoTIFF: `255` means the Sun center never clears the local horizon and `0`
means it clears the horizon at least once. Generation uses a staged file;
failure or cancellation preserves any completed destination. GDAL validity
mask `255` means the PSR byte was calculated; mask `0` means its required
horizon tile was missing or unreadable. Complete output uses the virtual
all-valid mask and partial output stores an internal 1-bit mask.

`11_native_end_to_end_validation.py` creates a dedicated solar-fraction series,
cancels an overwrite after native streaming begins, verifies byte-for-byte
preservation and scratch/staging cleanup, restarts the operation, and writes
`analysis/native_validation_report.json`. It independently checks the manifest
digest, TIFF metadata, VRT bands, exact memory/file pixel equality, and exact
native uint8-to-float32 conversion. It does not access `scenario.db`.

## QGIS

Run `05_file_backed_series.py`, then `09_qgis_vrt.py`. The latter prints:

- an individual timestamp TIFF that QGIS can open directly; and
- `series.vrt`, where VRT band `n + 1` corresponds to Python layer index `n`.

The VRT carries timestamp band descriptions but does not automatically
configure the QGIS temporal controller.

Manual validation of the native solar-fraction TIFF and VRT completed with
QGIS 3.44.7-Solothurn. A reported pixel size of `1, -1` is expected for the
north-up affine transform. Set the project CRS from the raster layer; QGIS
cannot and should not transform the lunar custom CRS to the default WGS 84
project CRS.

## Representative Performance Benchmark

`12_native_performance_benchmark.py` defaults to 3,800 hourly, 512 x 512
float32 Sun-margin layers. It measures native generation, scratch assembly,
GeoTIFF publication, VRT/final validation, random reads, all four temporal
reducers, process RSS, scratch usage, output size, and file count. Its cold and
warm random-read comparison refers to Lunarscout's full-layer application
cache; it does not drop the host-wide Linux page cache.

The output and scratch paths are isolated under `analysis/`, and the script
does not access `scenario.db`, register products, or reproject data. Use
`--reuse-existing` to repeat read and reduction measurements without rerunning
native generation.

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
