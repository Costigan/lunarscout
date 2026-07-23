#!/usr/bin/env python3
"""Generate the Lunarscout interactive notebook curriculum.

Commit these notebooks without outputs so diffs stay readable and
outputs do not become stale.  Execute during CI or documentation
builds to produce the rendered versions.
"""

from pathlib import Path
import nbformat as nbf

OUT = Path(__file__).resolve().parent.parent / "examples" / "notebooks"

# -- Shared constants -------------------------------------------------------
_MOON_WKT = (
    'PROJCS["ESRI:103878",'
    'GEOGCS["Moon_2000",DATUM["D_Moon_2000",'
    'SPHEROID["Moon_2000_IAU_IAG",1737400,0]],'
    'PRIMEM["Reference_Meridian",0],UNIT["Degree",0.0174532925199433]],'
    'PROJECTION["Polar_Stereographic"],'
    'PARAMETER["latitude_of_origin",-90],'
    'PARAMETER["central_meridian",0],'
    'PARAMETER["scale_factor",1],'
    'PARAMETER["false_easting",0],'
    'PARAMETER["false_northing",0],UNIT["Meter",1]]'
)

_MOON_PROJ4 = (
    "+proj=stere +lat_0=-90 +lon_0=0 +k=1 +x_0=0 +y_0=0 "
    "+R=1737400 +units=m +no_defs"
)

_SETUP_PREAMBLE = """\
import sys, os
from pathlib import Path

def _repo_root():
    \"\"\"Find the Lunarscout repository root from the kernel working directory.\"\"\"
    for start in [Path.cwd()] + list(Path.cwd().parents):
        if (start / "src" / "lunarscout" / "__init__.py").exists():
            return start
    raise RuntimeError(
        "Cannot locate Lunarscout repository root. "
        "Launch Jupyter from the repository root directory."
    )

_REPO = _repo_root()
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "examples"))
"""

_SETUP_PREAMBLE_TINY = """\
import sys, os
from pathlib import Path

def _repo_root():
    for start in [Path.cwd()] + list(Path.cwd().parents):
        if (start / "src" / "lunarscout" / "__init__.py").exists():
            return start
    raise RuntimeError("Cannot locate Lunarscout repository root.")

_REPO = _repo_root()
sys.path.insert(0, str(_REPO / "src"))
"""


def nb():
    n = nbf.v4.new_notebook()
    n.metadata = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.11.0",
        },
    }
    return n


def md(source):
    return nbf.v4.new_markdown_cell(source.strip())


def code(source):
    return nbf.v4.new_code_cell(source.strip())


def _moon_georef(*, width=4, height=3, origin_x=-20.0, origin_y=20.0,
                 pixel_size=10.0, nodata=None):
    """Build a tiny moon south-polar stereographic GeoReference."""
    return (
        f'ls.GeoReference(\n'
        f'    projection_wkt="""{_MOON_WKT}""",\n'
        f'    projection_proj4="{_MOON_PROJ4}",\n'
        f'    affine_transform=({origin_x}, {pixel_size}, 0.0,'
        f' {origin_y}, 0.0, {-pixel_size}),\n'
        f'    width={width}, height={height},\n'
        f'    pixel_size_x={pixel_size}, pixel_size_y={-pixel_size},\n'
        f'    nodata={nodata!r},\n'
        f')'
    )


# ===================================================================
# Notebook builders
# ===================================================================

def notebook_01():
    n = nb()
    n.cells = [
        md("""\
# 01 -- Raster and Spatial Foundations

This notebook covers GeoTIFF I/O, terrain products, connected-region
analysis, and explicit grid alignment.  It draws from command-line examples
01 through 04.

**Run the individual scripts:** `01_geotiff_and_coordinates.py`,
`02_terrain_products.py`, `03_region_filtering.py`, `04_alignment.py`
"""),
        md("""\
## Setup

All examples use a deterministic 64x64 synthetic lunar DEM in a south-polar
stereographic projection.  Run this cell once before executing any of the
sections below.
"""),
        code(f"""{_SETUP_PREAMBLE}

import lunarscout as ls
import numpy as np

from _example_support import (
    ensure_synthetic_scenario,
    synthetic_georef,
    synthetic_dem,
)

WORKSPACE = Path("/tmp/lunarscout_notebook_01")
WORKSPACE.mkdir(parents=True, exist_ok=True)

ensure_synthetic_scenario(WORKSPACE)
SCENARIO = ls.open_scenario(WORKSPACE / "synthetic_scenario")

print(f"Workspace: {{WORKSPACE}}")
print(f"Scenario DEM: {{SCENARIO.dem_path()}}")
"""),
        md("""\
---
## 1. GeoTIFF Read, Inspect, and Write

Read the DEM, verify that it is georeferenced, inspect its metadata, and
write a copy.
"""),
        code("""\
dem, georef = ls.read_geotiff(SCENARIO.dem_path())

if georef is None:
    raise RuntimeError("The DEM must be georeferenced.")

print(f"  dtype:           {dem.dtype}")
print(f"  shape:           {dem.shape}")
print(f"  min / max:       {dem.min():.2f} / {dem.max():.2f}")
print(f"  nodata:          {georef.nodata}")
print(f"  projection WKT:  {georef.projection_wkt[:60]}...")
print(f"  width x height:  {georef.width} x {georef.height}")
print(f"  pixel_size:      {georef.pixel_size_x:.2f} x {georef.pixel_size_y:.2f}")
print(f"  affine:          {georef.affine_transform}")

# Convert a sample pixel to projected and geographic coordinates.
x, y = 20, 16
px, py = georef.pixel_to_projected(x, y)
lon, lat = georef.pixel_to_lonlat(x, y)
print(f"\\nPixel ({x},{y}) -> projected ({px:.2f}, {py:.2f}) m")
print(f"Pixel ({x},{y}) -> lon/lat   ({lon:.4f}, {lat:.4f}) deg")
"""),
        code("""\
# Write a copy to the scenario analysis directory.
out = SCENARIO.output_path("analysis/dem_copy.tif")
ls.write_geotiff(out, dem, georef, overwrite=True)
print(f"Wrote {out}")
"""),
        md("""\
**Try this:** pick different pixel coordinates and convert them.  How do
projected coordinates change as you move one pixel right?  One pixel down?

---
## 2. Terrain Products -- Slope, Aspect, Hillshade

Lunarscout provides GDAL-compatible slope, aspect, and hillshade operations.
Each returns both the output values and a georeference for the product grid.
"""),
        code("""\
slope_values, slope_georef = ls.slope(dem, georef, output_nodata=-9999.0)
aspect_values, aspect_georef = ls.aspect(dem, georef, output_nodata=-9999.0)
# Nodata pixels produce NaN intermediates that trigger a harmless
# NumPy cast warning; the function replaces them with output_nodata.
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message="invalid value encountered in cast")
    hillshade_values, shade_georef = ls.hillshade(dem, georef, output_nodata=0)

valid_slope = slope_values[slope_values != slope_georef.nodata]
print(f"Slope     -- min: {valid_slope.min():.1f} deg, max: {valid_slope.max():.1f} deg")
valid_aspect = aspect_values[aspect_values != aspect_georef.nodata]
print(f"Aspect    -- min: {valid_aspect.min():.1f} deg, max: {valid_aspect.max():.1f} deg")
print(f"Hillshade -- min: {hillshade_values.min():.1f}, max: {hillshade_values.max():.1f}")
"""),
        code("""\
_terrain = SCENARIO.output_path("analysis/terrain")
_terrain.mkdir(parents=True, exist_ok=True)
ls.write_geotiff(str(_terrain / "slope_deg.tif"), slope_values, slope_georef, overwrite=True)
ls.write_geotiff(str(_terrain / "aspect_deg.tif"), aspect_values, aspect_georef, overwrite=True)
ls.write_geotiff(str(_terrain / "hillshade.tif"), hillshade_values, shade_georef, overwrite=True)
print(f"Terrain products written under {_terrain}")
"""),
        md("""\
Hillshade is a visualisation aid; it does not represent physical
illumination.  Slope and aspect are quantitative terrain measurements.

**Try this:** change the hillshade azimuth and altitude keyword arguments
and re-run.
"""),
        md("""\
---
## 3. Connected Regions

Turn a slope-threshold into a candidate mask, then label, measure, filter,
and outline the connected regions.
"""),
        code("""\
# Candidate: slope <= 8 degrees  (illustrative, not a mission threshold)
candidate = slope_values <= 8.0

labels, labels_georef = ls.label_regions(candidate, georef)
sizes, sizes_georef = ls.region_sizes(candidate, georef)

print(f"Number of connected candidates: {len(np.unique(labels)) - 1}")
print(f"Largest region size:            {sizes.max()}")
"""),
        code("""\
# Keep regions with at least 80 cells, apply morphological opening.
large, large_georef = ls.filter_regions_by_size(
    candidate, georef,
    threshold=80, comparator=">=",
    cleanup="opening", iterations=1,
)

# Extract borders of the remaining regions.
borders, borders_georef = ls.find_borders(large, large_georef)

print(f"Regions kept after size filter: {large.sum()}")
print(f"Border cell count:              {borders.sum()}")
"""),
        code("""\
_regions = SCENARIO.output_path("analysis/regions")
_regions.mkdir(parents=True, exist_ok=True)

# Use a zero-nodata encoding for boolean masks.
mask_nodata_georef = large_georef.with_nodata(0)
ls.write_geotiff(str(_regions / "large_regions.tif"), large.astype("uint8"), mask_nodata_georef, overwrite=True)
ls.write_geotiff(str(_regions / "borders.tif"), borders.astype("uint8"), mask_nodata_georef, overwrite=True)
print(f"Region products written under {_regions}")
"""),
        md("""\
The slope threshold of 8 degrees and the region-size threshold of 80 pixels
are **teaching choices**, not landing-site recommendations.

**Try this:** change the size threshold to 30 or 200.  Change the
comparator to `"<"`.  Use `connectivity=4` for cardinal-neighbor regions.
"""),
        md("""\
---
## 4. Grid Comparison and Explicit Alignment

Two arrays with the same shape can still describe different geographic
locations.  Lunarscout never infers grid compatibility from shape alone.
"""),
        code("""\
# Create a shifted copy of the slope raster.
shifted_values = np.roll(slope_values, shift=1, axis=1)

# Build a georeference with the origin moved 5 metres east.
from dataclasses import replace
shifted_georef = replace(slope_georef,
    affine_transform=(
        slope_georef.affine_transform[0] + 5.0,
    ) + slope_georef.affine_transform[1:],
)

# Same shape, different grid.
print(f"Shapes match: {slope_values.shape == shifted_values.shape}")
print(f"Grids match:  {ls.same_grid(slope_georef, shifted_georef)}")
"""),
        code("""\
# require_same_grid raises a structured error when grids differ.
try:
    ls.require_same_grid(slope_georef, shifted_georef)
except ls.GridMismatchError as e:
    print(f"GridMismatchError: code={e.code}")
    print(f"  Details: {e.details}")
"""),
        code("""\
# Align the shifted raster onto the reference grid with bilinear resampling.
aligned, aligned_georef = ls.align(
    shifted_values, shifted_georef,
    to=slope_georef,
    resampling="bilinear",
    output_nodata=-9999.0,
)

ls.require_same_grid(aligned_georef, slope_georef)
print("After alignment: grids match")

_alignment = SCENARIO.output_path("analysis/alignment")
_alignment.mkdir(parents=True, exist_ok=True)
ls.write_geotiff(str(_alignment / "aligned.tif"), aligned, aligned_georef, overwrite=True)
print(f"Aligned raster written under {_alignment}")
"""),
        md("""\
Alignment is an explicit analysis step.  Choose nearest-neighbour for
categorical data and a continuous method (bilinear, cubic, lanczos) for
continuous measurements.

**Try this:** use `resampling="nearest"` and compare the aligned values.
How big are the differences?
"""),
    ]
    return n


# ---------------------------------------------------------------------------
def notebook_02():
    n = nb()
    n.cells = [
        md("""\
# 02 -- Temporal Workflows

This notebook covers in-memory temporal cubes, file-backed GeoTIFF series,
incremental writing with progress, streaming reductions, and a screening
workflow.  It draws from command-line examples 05 through 10.

**Run the individual scripts:** `05_temporal_cube.py`,
`06_file_backed_series.py`, `07_incremental_writer.py`,
`08_streaming_reductions.py`, `10_landing_site_screening.py`
"""),
        md("""\
## Setup
"""),
        code(f"""{_SETUP_PREAMBLE}

import lunarscout as ls
import numpy as np

from _example_support import (
    ensure_synthetic_scenario,
    synthetic_georef,
    synthetic_dem,
    synthetic_times,
    synthetic_temporal_cube,
    ensure_synthetic_series,
)

WORKSPACE = Path("/tmp/lunarscout_notebook_02")
WORKSPACE.mkdir(parents=True, exist_ok=True)

ensure_synthetic_scenario(WORKSPACE)
series = ensure_synthetic_series(WORKSPACE)
georef = synthetic_georef()
cube = synthetic_temporal_cube(georef)

SCENARIO = ls.open_scenario(WORKSPACE / "synthetic_scenario")

print(f"Workspace: {{WORKSPACE}}")
"""),
        md("""\
---
## 1. In-Memory Temporal Cube

A `TemporalCube` holds a complete `(time, y, x)` NumPy array with UTC time
coordinates.  Use it for modest arrays and exploratory work.
"""),
        code("""\
print(f"  shape:      {cube.values.shape}")
print(f"  dtype:      {cube.values.dtype}")
print(f"  time start: {cube.times[0]}")
print(f"  time end:   {cube.times[-1]}")
print(f"  time steps: {len(cube.times)}")
print(f"  memory:     {cube.values.nbytes / 1024:.1f} KiB")
"""),
        code("""\
# Reduce the time axis into per-pixel statistics.
mean, mean_georef = ls.temporal_mean(cube)
_min, _ = ls.temporal_min(cube)
_max, _ = ls.temporal_max(cube)
std, _ = ls.temporal_std(cube)

print(f"Per-pixel mean:   [{mean.min():.3f}, {mean.max():.3f}]")
print(f"Per-pixel std:    [{std.min():.3f}, {std.max():.3f}]")
"""),
        code("""\
_temporal = SCENARIO.output_path("analysis/temporal")
_temporal.mkdir(parents=True, exist_ok=True)
for name, arr in [("mean", mean), ("minimum", _min), ("maximum", _max), ("standard_deviation", std)]:
    ls.write_geotiff(str(_temporal / f"{name}.tif"), arr, mean_georef, overwrite=True)
print(f"Temporal reducers written under {_temporal}")
"""),
        md("""\
**Try this:** create a `TemporalCube` from a NumPy array of your own
choosing, then try the same four reducers.
"""),
        md("""\
---
## 2. File-Backed Temporal Series

Store large time series as one single-band GeoTIFF per timestamp, plus a
manifest and optional VRT.  The series has no `.values` property -- this
prevents accidentally materialising a large cube.
"""),
        code("""\
print(f"Series root:   {series.root}")
print(f"Signal name:   {series.signal_name}")
print(f"VRT path:      {series.vrt_path}")

# Read by zero-based layer index.
layer0, _ = series.read_layer(0)
print(f"\\nLayer 0 shape: {layer0.shape}, min={layer0.min():.3f}, max={layer0.max():.3f}")
"""),
        code("""\
# Read nearest to a specific UTC time.
from datetime import datetime, timezone
target = datetime(2027, 1, 1, 2, 30, tzinfo=timezone.utc)
nearest, _ = series.read_time(target, method="nearest")
print(f"Nearest to {target}: shape={nearest.shape}")
"""),
        md("""\
**Try this:** read the layer for a specific index and print its raster
statistics using NumPy.
"""),
        md("""\
---
## 3. Incremental Writer

Write directly through `TemporalGeoTiffSeriesWriter` without first building
a `TemporalCube`.  Memory stays proportional to one layer.
"""),
        code("""\
from datetime import datetime, timedelta, timezone

output_root = SCENARIO.output_path("analysis/incremental_sun")
print(f"Writing to {output_root}")

def _on_progress(progress):
    print(f"  wrote {progress.last_time}")

with ls.TemporalGeoTiffSeriesWriter(
    output_root,
    georef=georef,
    dtype=np.float32,
    signal_name="sun_fraction",
    units="fraction",
    progress_callback=_on_progress,
    overwrite=True,
) as w:
    start = datetime(2027, 1, 1, 0, 0, tzinfo=timezone.utc)
    for i in range(6):
        t = start + timedelta(hours=i)
        layer = np.sin(
            np.linspace(0, np.pi, georef.width * georef.height,
                        dtype=np.float32)
        ).reshape(georef.height, georef.width) * 0.5 + 0.5
        w.write_layer(t, layer)

incremental = w.result
print(f"\\nFinalised: {incremental.root}")
incremental.close()
"""),
        md("""\
**Try this:** replace the sine-wave layer with a function that depends on
the pixel column index.
"""),
        md("""\
---
## 4. Streaming Temporal Reductions

The same reducer names (`temporal_mean`, `temporal_min`, ...) accept both
`TemporalCube` and `TemporalGeoTiffSeries`.  For a series they stream
layers without loading the full cube.
"""),
        code("""\
s_mean, s_georef = ls.temporal_mean(series)
s_stdev, _ = ls.temporal_std(series)

print(f"Streamed mean range:   [{s_mean.min():.4f}, {s_mean.max():.4f}]")
print(f"Streamed std range:    [{s_stdev.min():.4f}, {s_stdev.max():.4f}]")
"""),
        md("""\
Compare with the in-memory reductions from section 1.  The results should
agree within floating-point rounding.
"""),
        md("""\
---
## 5. Landing-Site Screening

Combine slope and temporal mean illumination into a candidate mask, then
filter for regions of adequate size.  This mirrors `10_landing_site_screening.py`.
"""),
        code("""\
dem, dem_georef = ls.read_geotiff(SCENARIO.dem_path())
slope_val, slope_georef = ls.slope(dem, dem_georef, output_nodata=-9999.0)
illum, illum_georef = ls.temporal_mean(series)

# Grid compatibility check before combining.
ls.require_same_grid(slope_georef, illum_georef)

# Illustrative thresholds -- not landing-site recommendations.
SLOPE_MAX = 8.0          # degrees
ILLUM_MIN = 0.60         # fraction
REGION_MIN = 80          # pixels

candidate = (slope_val <= SLOPE_MAX) & (illum >= ILLUM_MIN)
print(f"Candidate cells before filtering:  {candidate.sum()}")

large, large_georef = ls.filter_regions_by_size(
    candidate, slope_georef,
    threshold=REGION_MIN, comparator=">=",
)
borders, _ = ls.find_borders(large, large_georef)
print(f"Candidate cells after size filter: {large.sum()}")
"""),
        code("""\
_screening = SCENARIO.output_path("analysis/screening")
_screening.mkdir(parents=True, exist_ok=True)
nodata_gref = large_georef.with_nodata(0)
ls.write_geotiff(str(_screening / "candidate_sites.tif"), large.astype("uint8"), nodata_gref, overwrite=True)
ls.write_geotiff(str(_screening / "candidate_borders.tif"), borders.astype("uint8"), nodata_gref, overwrite=True)
print(f"Screening outputs written under {_screening}")
"""),
        md("""\
**Try this:** vary `SLOPE_MAX`, `ILLUM_MIN`, and `REGION_MIN`.  How do the
candidate regions respond?  What happens when you set `ILLUM_MIN` to 1.0?
"""),
        md("""\
---
## Cleanup
"""),
        code("""\
series.close()
print("Series closed.")
"""),
    ]
    return n


# ---------------------------------------------------------------------------
def notebook_03():
    n = nb()
    n.cells = [
        md("""\
# 03 -- Celestial Geometry

This notebook covers Sun and Earth local-frame vectors, azimuth/elevation
angles, body path plots against terrain horizons, and a synthetic lightmap
generated from explicit vectors.  It draws from command-line examples 11
through 13.

**Requirements:** SPICE kernel download on first use (network access) and
the synthetic horizon bundle (downloaded automatically, ~37 MB).

**Run the individual scripts:** `11_spice_vectors.py`,
`12_body_and_horizon_plots.py`, `13_synthetic_lightmap.py`
"""),
        md("""\
## Setup
"""),
        code(f"""{_SETUP_PREAMBLE}

from datetime import timedelta

import lunarscout as ls
import numpy as np
import matplotlib.pyplot as plt

from _example_support import ensure_synthetic_horizon_scenario

WORKSPACE = Path("/tmp/lunarscout_notebook_03")
WORKSPACE.mkdir(parents=True, exist_ok=True)

# Choose a south-polar point (must be within the synthetic horizon DEM).
# lat=-89.99 is ~300m from the pole; the 256x256, 20m-pixel DEM covers ~5km.
POINT = ls.LonLat(longitude=0.0, latitude=-89.99)

times_short = list(ls.iter_times(
    "2027-01-01T00:00:00Z", "2027-01-01T06:00:00Z", timedelta(hours=2),
))
times_long = list(ls.iter_times(
    "2027-01-01T00:00:00Z", "2027-01-02T00:00:00Z", timedelta(hours=1),
))

print(f"South-polar point: {{POINT}}")
print(f"Short sample: {{len(times_short)}} times  (elevation plots)")
print(f"Long sample:  {{len(times_long)}} times   (horizon paths)")
"""),
        md("""\
---
## 1. SPICE Vectors and Angles

Vectors are returned in the local North-East-Down frame, in km.  Angles use
azimuth 0 = north, 90 = east; elevation +90 = straight up.

The first call downloads and caches default SPICE kernels.  Subsequent runs
reuse the cache.
"""),
        code("""\
vectors_ned = ls.body_vectors_ned(POINT, "sun", times_short)
angles = ls.body_azimuth_elevation(POINT, "sun", times_short)

print("Sun NED vectors (km) -- shape:", vectors_ned.shape)
for i, t in enumerate(times_short):
    v = vectors_ned[i]
    a = angles[i]
    print(f"  {t}: NED=({v[0]:+.1f}, {v[1]:+.1f}, {v[2]:+.1f})  "
          f"az={a[0]:.0f} deg  el={a[1]:.1f} deg")
"""),
        code("""\
# Earth geometry as DataFrames.
df = ls.body_azimuth_elevation_dataframe(POINT, "earth", times_short)
df
"""),
        md("""\
**Try this:** change the point to `LonLat(longitude=45.0, latitude=-85.0)`
and compare the Sun azimuth progression.
"""),
        md("""\
---
## 2. Body Elevation Over Time
"""),
        code("""\
fig, ax = ls.plot_body_elevation(POINT, "sun", times_long, grid=True)
fig.suptitle("Sun Elevation vs Time")
fig.tight_layout()
plt.show()
"""),
        code("""\
fig, ax = ls.plot_body_elevations(
    POINT, bodies=["sun", "earth"],
    times=times_long, grid=True,
)
fig.suptitle("Sun and Earth Elevation")
fig.tight_layout()
plt.show()
"""),
        md("""\
**Try this:** set `grid=False` and experiment with the Matplotlib axes
returned by the plot functions.
"""),
        md("""\
---
## 3. Terrain Horizon and Body Paths

This section requires the synthetic horizon scenario (downloaded
automatically).  The conceptual shift is from elevation relative to an
ideal horizontal plane to elevation relative to actual surrounding terrain.
"""),
        code("""\
scenario = ensure_synthetic_horizon_scenario(WORKSPACE)
if scenario is None:
    print("Skipping horizon section -- synthetic horizon data not available.")
    print("Run again with network access for the initial download.")
else:
    print(f"Horizon scenario root: {scenario.root_path()}")
"""),
        code("""\
if scenario is not None:
    fig, ax = scenario.plot_horizon(POINT, center_azimuth=0.0)
    fig.suptitle("Terrain Horizon (north-centred)")
    fig.tight_layout()
    plt.show()
"""),
        code("""\
if scenario is not None:
    fig, ax = scenario.plot_horizon(POINT, center_azimuth=90.0)
    scenario.plot_body_path(ax, POINT, body="sun", times=times_long,
                            style="center_and_limbs", label="Sun")
    scenario.plot_body_path(ax, POINT, body="earth", times=times_long,
                            style="limbs", label="Earth")
    ax.legend()
    fig.suptitle("Sun and Earth Paths over Horizon")
    fig.tight_layout()
    plt.show()
"""),
        code("""\
if scenario is not None:
    fig, ax = scenario.plot_zoomed_body_path(
        POINT, bodies=["sun", "earth"], times=times_long,
        observer_height_decimeters=0,
    )
    fig.suptitle("Zoomed Body Path over Horizon")
    fig.tight_layout()
    plt.show()
"""),
        md("""\
**Try this:** change `center_azimuth` to 180 or 270 and re-plot the
horizon.  Use `over_horizon=True` with `plot_body_elevations` to see
elevation above the terrain.
"""),
        md("""\
---
## 4. Synthetic Lightmap with Explicit Vectors

Generate a lightmap using explicit Moon-ME Sun vectors.  This avoids SPICE
kernel loading during the product call and runs on CPU.
"""),
        code("""\
if scenario is not None:
    from datetime import datetime, timezone

    output = WORKSPACE / "horizon_scenario"
    lightmap_path = output / "analysis" / "synthetic_lightmap.tif"
    lightmap_path.parent.mkdir(parents=True, exist_ok=True)

    # Hard-coded Sun vectors for reproducibility (km, Moon-ME).
    sun_distance_km = 147_010_225.48
    sun_vecs = np.array([
        [sun_distance_km, 0.0, 0.0],
        [sun_distance_km, 0.0, 0.0],
        [sun_distance_km, 0.0, 0.0],
        [sun_distance_km, 0.0, 0.0],
    ], dtype=np.float64)

    ls.generate_lightmap(
        str(output / "dem.tif"),
        str(output / "horizons"),
        str(lightmap_path),
        times=[
            datetime(2027, 1, 1, h, tzinfo=timezone.utc)
            for h in range(0, 8, 2)
        ],
        sun_vectors_m=sun_vecs,
        backend="cpu",
        overwrite=True,
    )
    print(f"Lightmap written to {lightmap_path}")

    # Read back band statistics.
    for b in range(4):
        arr, _ = ls.read_geotiff(lightmap_path, band=b + 1)
        masked = arr[arr != 0]
        print(f"  Band {b}: {len(masked)} valid pixels, "
              f"range [{masked.min()}, {masked.max()}]")
"""),
        md("""\
**Try this:** vary the Sun vector direction to simulate different
illumination geometries.  The lightmap encodes visible solar fraction as
`uint8` (0 = fully obscured, 255 = fully visible).
"""),
    ]
    return n


# ---------------------------------------------------------------------------
def notebook_04():
    n = nb()
    georef_expr = _moon_georef()
    n.cells = [
        md("""\
# 04 -- Map-Algebra Foundations

This notebook introduces the eager `Raster` value type: construction,
inspection, arithmetic, comparisons, Boolean logic, validity handling,
grid alignment, units, and numerical policies.  It draws from command-line
examples 18 through 21.

**Run the individual scripts:** `18_map_algebra_basics.py`,
`19_map_algebra_validity.py`, `20_map_algebra_grids.py`,
`21_map_algebra_numerics.py`
"""),
        md("""\
## Setup
"""),
        code(f"""{_SETUP_PREAMBLE_TINY}

import lunarscout as ls
import numpy as np

ma = ls.map_algebra

MOON_WKT = \"\"\"{_MOON_WKT}\"\"\"
MOON_PROJ4 = \"{_MOON_PROJ4}\"

DEMO = {georef_expr}

print(f"Demo grid: {{DEMO.width}}x{{DEMO.height}}, {{DEMO.pixel_size_x}}m pixels")
"""),
        md("""\
---
## 1. Building and Inspecting a Raster

A `Raster` stores ordinary NumPy values together with a validity mask,
georeference, optional units, and an optional name.
"""),
        code("""\
rng = np.random.default_rng(42)
values = rng.uniform(0, 15, (DEMO.height, DEMO.width)).astype(np.float32)
valid = np.ones((DEMO.height, DEMO.width), dtype=bool)
valid[1, 1] = False   # mark one cell invalid

slope = ma.raster(values, DEMO, valid=valid, units="deg", name="slope")

print(slope)
print(f"\\n  values dtype:  {slope.values.dtype}")
print(f"  values shape:   {slope.values.shape}")
print(f"  valid count:    {slope.valid.sum()}")
print(f"  units:          {slope.units}")
print(f"  name:           {slope.name}")
"""),
        code("""\
# Inspect values and validity side by side.
print("Values:           Validity:")
for y in range(DEMO.height):
    v_line = "  ".join(f"{slope.values[y, x]:5.1f}" for x in range(DEMO.width))
    m_line = "  ".join(f"{'V' if slope.valid[y, x] else 'X':>5}" for x in range(DEMO.width))
    print(f"{v_line}   {m_line}")
"""),
        md("""\
The cell at row 1, column 1 is invalid even though it has a numeric
payload.  Validity is independent of the payload value.

**Try this:** create a `Raster` with `valid` all `True` and print it.
Then create one where the validity mask does not match the value shape
and observe the error.
"""),
        md("""\
---
## 2. Local Algebra

Map-algebra operations return new rasters, preserve metadata, and intersect
operand validity.
"""),
        code("""\
elevation = ma.raster(
    np.array([[1100, 1050, 1005,  980],
              [1200, 1150,    0,  950],
              [1300, 1250, 1200, 1150]], dtype=np.float32),
    DEMO, units="m", name="elevation",
)

# Beware: adding metres and degrees raises MapAlgebraUnitError.
try:
    elevation + slope
except ls.MapAlgebraUnitError as e:
    print(f"m + deg -> MapAlgebraUnitError: code={e.code}")
"""),
        code("""\
# Compatible operations: arithmetic, comparisons, clip.
scaled = elevation / 1000.0          # -> dim-less
steep = slope >= 8.0                  # -> bool
clipped = ma.clip(slope, lower=2.0, upper=10.0)

print(f"scaled units:  {scaled.units}")
print(f"steep dtype:   {steep.values.dtype}")
print(f"clipped range: [{clipped.values.min():.1f}, {clipped.values.max():.1f}]")
"""),
        code("""\
# Boolean operations: & | ~  (not 'and'/'or')
sun = elevation >= 1050.0
shadow = ~sun

both = steep & ~shadow
either = steep | ~shadow

print(f"Steep & ~shadow:  {both.values.sum()} cells")
print(f"Steep | ~shadow:  {either.values.sum()} cells")
"""),
        md("""\
Python `and` / `or` are rejected.  Use `&`, `|`, and `~` for raster
Boolean logic.

**Try this:** combine three conditions with `&`.  Compute `slope > 10.0`
and `elevation < 1100.0` and intersect them.
"""),
        md("""\
---
## 3. Validity, `where`, and `coalesce`

This is the most important validity lesson.  The inputs below contain a
valid zero, an invalid zero payload, and an invalid non-zero payload that
looks plausible.
"""),
        code("""\
# Three values at three pixels.
values_A = np.array([[1.0, 0.0, 99.0]], dtype=np.float32)
valid_A =  np.array([[True, True, False]], dtype=bool)

values_B = np.array([[0.5, 0.5, 0.5]], dtype=np.float32)
valid_B =  np.array([[True, False, False]], dtype=bool)

g1x3 = ls.GeoReference(
    projection_wkt=MOON_WKT, projection_proj4=MOON_PROJ4,
    affine_transform=(0.0, 10.0, 0.0, 0.0, 0.0, -10.0),
    width=3, height=1, pixel_size_x=10.0, pixel_size_y=-10.0,
    nodata=None,
)
A = ma.raster(values_A, g1x3, valid=valid_A, name="A")
B = ma.raster(values_B, g1x3, valid=valid_B, name="B")

print("A: values", A.values, "valid", A.valid)
print("B: values", B.values, "valid", B.valid)
"""),
        code("""\
# Ordinary arithmetic intersects operand validity.
C = A + B
print("A + B:")
print("  values:", C.values, "valid:", C.valid)

# where: requires condition AND selected branch to be valid.
w = ma.where(A < 10.0, A, ma.invalid)
print("\\nwhere(A < 10, A, invalid):")
print("  values:", w.values, "valid:", w.valid)
"""),
        code("""\
# coalesce: takes the first valid value.
c = ma.coalesce(A, B)
print("coalesce(A, B):")
print("  values:", c.values, "valid:", c.valid)

c2 = ma.coalesce(B, A)
print("coalesce(B, A):")
print("  values:", c2.values, "valid:", c2.valid)
"""),
        code("""\
# fill_invalid converts missing cells to valid caller values.
filled = ma.fill_invalid(A, -999.0)
print("fill_invalid(A, -999):")
print("  values:", filled.values, "valid:", filled.valid)
"""),
        md("""\
Reversing `coalesce` order changes meaning.  Filling with zero also changes
meaning: the cells cease to be missing.  Do not use filling merely to make a
plot look complete.

**Try this:** create a raster where all cells are invalid and then
`coalesce` it with another.  What happens?
"""),
        md("""\
---
## 4. Explicit Grid Alignment

Two same-shaped `Raster` values on shifted grids are rejected.
"""),
        code(f"""\
g2 = ls.GeoReference(
    projection_wkt=MOON_WKT, projection_proj4=MOON_PROJ4,
    affine_transform=(-15.0, 10.0, 0.0, 20.0, 0.0, -10.0),
    width=4, height=3, pixel_size_x=10.0, pixel_size_y=-10.0,
    nodata=None,
)

r1 = ma.raster(values, DEMO, name="on_demo")
r2 = ma.raster(values, g2, name="on_shifted")

try:
    _ = r1 + r2
except ls.MapAlgebraGridError as e:
    print(f"MapAlgebraGridError: code={{e.code}}")
    print(f"  {{e.details}}")
"""),
        code("""\
# Align explicitly.
r2_aligned = ma.align(r2, to=DEMO, resampling="bilinear", output_nodata=-9999.0)
result = r1 + r2_aligned   # succeeds
print(f"After alignment: {result.values[0, :2]} ...")
"""),
        md("""\
**Try this:** use `ma.row_indices(DEMO)` and `ma.projected_x(DEMO)` to
build coordinate rasters.  Materialise them with `ma.compute()`.
"""),
        md("""\
---
## 5. Units and Numerical Policies

Units are tracked through operations.  Numerical policies control overflow,
casting, and non-finite handling.
"""),
        code(f"""\
# Use a 1-row, 3-column grid for these demos.
g1x3 = ls.GeoReference(
    projection_wkt=MOON_WKT, projection_proj4=MOON_PROJ4,
    affine_transform=(0.0, 10.0, 0.0, 0.0, 0.0, -10.0),
    width=3, height=1, pixel_size_x=10.0, pixel_size_y=-10.0,
    nodata=None,
)
e1 = ma.raster(np.array([[1.0, 2.0, 3.0]], dtype=np.float32),
               g1x3, units="m", name="a")
e2 = ma.raster(np.array([[10.0, 20.0, 30.0]], dtype=np.float32),
               g1x3, units="m", name="b")
e3 = ma.raster(np.array([[1.0, 2.0, 3.0]], dtype=np.float32),
               g1x3, units="deg", name="c")

print(f"m + m  = {{(e1 + e2).values}}")      # OK
try:
    e1 + e3
except ls.MapAlgebraUnitError as e:
    print(f"m + deg -> MapAlgebraUnitError: {{e.code}}")
"""),
        code("""\
# Trigonometric functions require angle units.
g1x4 = ls.GeoReference(
    projection_wkt=MOON_WKT, projection_proj4=MOON_PROJ4,
    affine_transform=(0.0, 10.0, 0.0, 0.0, 0.0, -10.0),
    width=4, height=1, pixel_size_x=10.0, pixel_size_y=-10.0,
    nodata=None,
)
angle = ma.raster(np.array([[0.0, 30.0, 45.0, 90.0]], dtype=np.float32),
                  g1x4, units="degrees", name="angle")
s = ma.sin(angle)
print(f"sin([0 30 45 90] degrees): {s.values}")
"""),
        code("""\
# Overflow policies on integer arithmetic.
g1x3 = ls.GeoReference(
    projection_wkt=MOON_WKT, projection_proj4=MOON_PROJ4,
    affine_transform=(0.0, 10.0, 0.0, 0.0, 0.0, -10.0),
    width=3, height=1, pixel_size_x=10.0, pixel_size_y=-10.0,
    nodata=None,
)
tiny = ma.raster(
    np.array([[-100, 50, 120]], dtype=np.int8),
    g1x3, name="int8_values",
)

# raise on overflow.
try:
    tiny + tiny
except (ls.MapAlgebraDTypeError, ls.MapAlgebraOperationError) as e:
    print(f"int8 overflow (raise): code={e.code}")

# promote to a wider dtype.
promoted = ma.add(tiny, tiny, overflow="promote")
print(f"int8 overflow (promote): "
      f"dtype={promoted.values.dtype}, values={promoted.values}")

# wrap (NumPy behaviour).
wrapped = ma.add(tiny, tiny, overflow="wrap")
print(f"int8 overflow (wrap):    "
      f"dtype={wrapped.values.dtype}, values={wrapped.values}")
"""),
        code("""\
# Non-finite policies.
v = ma.raster(np.array([[1.0, 0.0, -1.0, 2.0]], dtype=np.float32), g1x4)

# log of non-positive values.
log_inval = ma.log(v, numeric_errors="invalid")
log_keep  = ma.log(v, numeric_errors="keep")

print("log with numeric_errors='invalid':")
print("  values:", log_inval.values, "valid:", log_inval.valid)
print("log with numeric_errors='keep':")
print("  values:", log_keep.values, "valid:", log_keep.valid)
"""),
        md("""\
Under `"keep"`, a NaN payload is still a valid cell.  Under `"invalid"`,
the validity mask carries the failure.  Choose the policy that matches your
scientific intent.

**Try this:** experiment with `ma.cast()` using `casting="safe"` versus
`"unsafe"`.  What happens when you cast a negative float to `uint8`?
"""),
    ]
    return n


# ---------------------------------------------------------------------------
def notebook_05():
    n = nb()
    n.cells = [
        md("""\
# 05 -- Suitability and Neighbourhoods

Weighted suitability scoring, focal smoothing, morphological cleanup, and
distance fields.  This notebook draws from command-line examples 22 and 25.

**Run the individual scripts:** `22_map_algebra_suitability.py`,
`25_map_algebra_focal.py`
"""),
        md("""\
## Setup
"""),
        code(f"""{_SETUP_PREAMBLE}

import lunarscout as ls
import numpy as np

ma = ls.map_algebra

from _example_support import (
    ensure_synthetic_scenario, ensure_synthetic_series,
    synthetic_georef, synthetic_dem,
)

WORKSPACE = Path("/tmp/lunarscout_notebook_05")
WORKSPACE.mkdir(parents=True, exist_ok=True)

ensure_synthetic_scenario(WORKSPACE)
georef = synthetic_georef()
SCENARIO = ls.open_scenario(WORKSPACE / "synthetic_scenario")

# Load DEM and temporal data as eager rasters.
dem, dem_georef = ls.read_geotiff(SCENARIO.dem_path())
ls.require_same_grid(georef, dem_georef)

slope_vals, _ = ls.slope(dem, dem_georef, output_nodata=-9999.0)
slope = ma.raster(slope_vals, georef, units="deg", name="slope")

series = ensure_synthetic_series(WORKSPACE)
illum_mean, _ = ls.temporal_mean(series)
sun = ma.from_existing(illum_mean, georef, units="fraction", name="illumination_mean")

print(f"Slope raster:  {{slope}}")
print(f"Sun raster:    {{sun}}")
print(f"Workspace:     {{WORKSPACE}}")
"""),
        md("""\
---
## 1. Weighted Suitability

Construct hard constraints, normalise scores, and combine them with
explicit weights.  Use `where` to preserve scores only for candidate cells.

**Thresholds are illustrative, not mission recommendations.**
"""),
        code("""\
# --- Configuration (edit these) ---
SLOPE_MAX = 8.0          # degrees
SUN_MIN = 0.60           # mean fraction
WEIGHT_SUN = 0.4
WEIGHT_SLOPE = 0.6
# ---------------------------------

# Hard constraints.
accept_slope = slope <= SLOPE_MAX
accept_sun = sun >= SUN_MIN
candidate = accept_slope & accept_sun

print(f"Candidate cells: {candidate.values.sum()} / {candidate.values.size}")

# Normalised scores (0 = worst, 1 = best).
slope_norm = ma.normalize_minmax(slope, minimum=0.0, maximum=15.0)
sun_norm = ma.normalize_minmax(sun, minimum=0.0, maximum=1.0)

score = (WEIGHT_SUN * sun_norm) + (WEIGHT_SLOPE * (1.0 - slope_norm))
scored = ma.where(candidate, score, ma.invalid)
scored = scored.with_name("weighted_score")

# Print statistics for valid cells.
stats = ma.statistics(scored)
print(f"\\nScore statistics (valid cells only):")
print(f"  min:    {stats.min_val}")
print(f"  max:    {stats.max_val}")
print(f"  mean:   {stats.mean}")
print(f"  std:    {stats.std}")
print(f"  count:  {stats.count}")
print(f"  invalid: {stats.invalid_count}")
"""),
        code("""\
_suit = SCENARIO.output_path("analysis/screening")
_suit.mkdir(parents=True, exist_ok=True)

nodata_gref = georef.with_nodata(0)
candidate_out = _suit / "candidate.tif"
ls.write_geotiff(
    str(candidate_out),
    candidate.values.astype("uint8"),
    nodata_gref,
    overwrite=True,
)
print(f"Candidate mask: {candidate_out}")

score_out = _suit / "candidate_score.tif"
ma.write(
    str(score_out),
    scored.expression(),
    dtype=np.float32,
    invalid_value=-9999.0,
    overwrite=True,
)
print(f"Score raster:   {score_out}")
"""),
        md("""\
**Try this:** change the weights so sun counts more than slope.  Change
`SLOPE_MAX` to 10.0 and observe how the candidate area grows.  What happens
if you set `SUN_MIN` to 1.0?
"""),
        md("""\
---
## 2. Focal Smoothing

A 3x3 focal mean smooths the slope raster.  This demonstrates
neighbourhood operations.
"""),
        code("""\
smoothed = ma.focal_mean(slope, size=3, edge="nearest")
print(f"Original slope  range: [{slope.values.min():.1f}, {slope.values.max():.1f}]")
print(f"Smoothed slope  range: [{smoothed.values.min():.1f}, {smoothed.values.max():.1f}]")

STEEP = 8.0
steep = smoothed >= STEEP
print(f"\\nSteep cells (>= {STEEP} deg, smoothed): {steep.values.sum()}")
"""),
        md("""\
---
## 3. Morphological Opening

Remove small isolated steep features with morphological opening.
"""),
        code("""\
opened = ma.opening(steep, size=3)
print(f"Before opening: {steep.values.sum()} steep cells")
print(f"After opening:  {opened.values.sum()} steep cells")
"""),
        md("""\
---
## 4. Distance to a Feature

Compute the Euclidean distance from every cell to the nearest steep-zone
pixel.
"""),
        code("""\
dist = ma.distance_to(opened, metric="euclidean", units="pixels")
d_stats = ma.statistics(dist)
print(f"Distance statistics ({dist.units}):")
print(f"  min:   {d_stats.min_val}")
print(f"  max:   {d_stats.max_val}")
print(f"  mean:  {d_stats.mean}")
print(f"  count: {d_stats.count}")
"""),
        code("""\
_focal = SCENARIO.output_path("analysis/focal")
_focal.mkdir(parents=True, exist_ok=True)

ls.write_geotiff(str(_focal / "smoothed_slope.tif"), smoothed.values, georef, overwrite=True)
ls.write_geotiff(str(_focal / "opened_mask.tif"), opened.values.astype("uint8"), nodata_gref, overwrite=True)
ls.write_geotiff(str(_focal / "distance_to_steep.tif"), dist.values, georef, overwrite=True)
print(f"Focal products written under {_focal}")
"""),
        md("""\
**Try this:** compute distance in physical metres instead of pixels.
What additional information does Lunarscout need to support that?
"""),
        md("""\
---
## Cleanup
"""),
        code("""\
series.close()
"""),
    ]
    return n


# ---------------------------------------------------------------------------
def notebook_06():
    n = nb()
    n.cells = [
        md("""\
# 06 -- Lazy and Temporal Algebra

This notebook introduces `RasterExpression` (deferred calculation),
`ma.source()`, terrain expression nodes, `ma.explain()` / `ma.plan()`,
bounded-window writes with `ma.write()`, and temporal map algebra with
`ma.temporal_source()` and temporal reductions.  It draws from command-line
examples 27 and 31.

A `RasterExpression` describes a calculation without running it.  Use
`ma.explain()` and `ma.plan()` to review it, `ma.compute()` to
materialise it eagerly, or `ma.write()` for supported bounded execution.

**Run the individual scripts:** `27_map_algebra_terrain_resample.py`,
`31_map_algebra_temporal.py`
"""),
        md("""\
## Setup
"""),
        code(f"""{_SETUP_PREAMBLE}

import lunarscout as ls
import numpy as np

ma = ls.map_algebra

from _example_support import (
    ensure_synthetic_scenario, ensure_synthetic_series,
    synthetic_georef, synthetic_dem,
)

WORKSPACE = Path("/tmp/lunarscout_notebook_06")
WORKSPACE.mkdir(parents=True, exist_ok=True)

ensure_synthetic_scenario(WORKSPACE)
georef = synthetic_georef()
SCENARIO = ls.open_scenario(WORKSPACE / "synthetic_scenario")
series = ensure_synthetic_series(WORKSPACE)

DEM_PATH = SCENARIO.dem_path()
print(f"DEM:    {{DEM_PATH}}")
print(f"Grid:   {{georef.width}}x{{georef.height}}, {{georef.pixel_size_x}}m")
print(f"Series: {{series.root}}")
"""),
        md("""\
---
## 1. Expressions, Explain, and Plan

A `RasterExpression` is an immutable description of a calculation that has
not yet run.  You obtain one from `ma.source()`, `Raster.expression()`, or
registered operators.
"""),
        code("""\
# Eager reference: slope from the DEM.
dem_arr, _ = ls.read_geotiff(DEM_PATH)
eager_slope, _ = ls.slope(dem_arr, georef, output_nodata=-9999.0)

# Lazy: describe the same calculation without running it.
dem_expr = ma.source(DEM_PATH, units="m")
slope_expr = ma.slope(dem_expr)

print(type(slope_expr).__name__)
"""),
        code("""\
# Explain: a human-readable summary of the expression tree.
print(ma.explain(slope_expr))
"""),
        md("""\
**Explain** shows the operation tree without calculating any pixels.  Put
`ma.explain()` directly beside the expression it describes.
"""),
        code("""\
# Plan: validate the expression and inspect the output strategy.
import tempfile
with tempfile.TemporaryDirectory() as td:
    out = Path(td) / "slope.tif"
    plan = ma.plan(slope_expr, output=out)
    print(f"Output grid:  {plan['output_grid']['width']}x{plan['output_grid']['height']}")
    print(f"Output dtype: {plan['output_dtype']}")
    print(f"Window size:  {plan['planner']['window_width']}x{plan['planner']['window_height']}")
    print(f"Window count: {plan['planner']['total_windows']}")
    print(f"Node count:   {plan['node_count']}")
"""),
        md("""\
Planning is read-only.  It rejects unsupported operations before creating
any output file.

**Try this:** explain `ma.hillshade(dem_expr, azimuth=315.0, altitude=45.0)`.
What nodes do you see?
"""),
        md("""\
---
## 2. Bounded-Window Writes

`ma.write()` evaluates an expression in bounded windows and produces a
staged, resumable, atomically-published GeoTIFF.
"""),
        code("""\
_out_dir = WORKSPACE / "terrain_resample"
_out_dir.mkdir(parents=True, exist_ok=True)

# Compute and write slope through the expression path.
slope_out = _out_dir / "slope.tif"
ma.write(
    str(slope_out),
    slope_expr,
    dtype=np.float32,
    invalid_value=-9999.0,
    window_width=128,
    window_height=128,
    overwrite=True,
)
print(f"Wrote {slope_out}")

# Read back and compare with eager calculation.
read_slope, _ = ls.read_geotiff(slope_out)
# Strip nodata for valid-cell comparison.
eager_valid = eager_slope != -9999.0
same = np.allclose(eager_slope[eager_valid], read_slope[eager_valid], atol=1e-4)
print(f"Windowed result matches eager: {same}")
"""),
        code("""\
# Compose terrain and illumination: weighted score.
illum_mean, _ = ls.temporal_mean(series)
sun_raster = ma.from_existing(illum_mean, georef, units="fraction", name="illumination_mean")

hillshade_expr = ma.hillshade(dem_expr, azimuth=315.0, altitude=45.0)

# Normalise both to [0, 1].
hs_norm = ma.normalize_minmax(hillshade_expr, minimum=0.0, maximum=1.0)
sun_norm = ma.normalize_minmax(
    sun_raster.expression(), minimum=0.0, maximum=1.0,
)

W_SUN, W_TERRAIN = 0.4, 0.6
score_expr = (W_SUN * sun_norm) + (W_TERRAIN * hs_norm)

print(ma.explain(score_expr))
"""),
        code("""\
score_out = _out_dir / "combined_score.tif"
ma.write(
    str(score_out),
    score_expr,
    dtype=np.float32,
    invalid_value=-1.0,
    overwrite=True,
)
print(f"Wrote {score_out}")
"""),
        md("""\
The `invalid_value` (`-1.0`) identifies invalid payload bytes for
interchange.  The **dataset mask** remains the authoritative validity
representation.

**Try this:** inspect the slope plan again.  Then write hillshade in the
same way and verify the result against eager hillshade from `ls.hillshade()`.
"""),
        md("""\
---
## 3. Temporal Map Algebra

Wrap a file-backed `TemporalGeoTiffSeries` with `ma.temporal_source()`,
reduce it with `ma.temporal_mean()`, and combine the resulting spatial
expression with eager constraints.
"""),
        code("""\
temporal_expr = ma.temporal_source(series)
mean_expr = ma.temporal_mean(temporal_expr)

print(ma.explain(mean_expr))
"""),
        code("""\
# Temporal reductions are whole-raster nodes.  Use ma.compute() first.
mean_sun = ma.compute(mean_expr)

print(f"Mean sun: dtype={mean_sun.values.dtype}, "
      f"range=[{mean_sun.values.min():.4f}, {mean_sun.values.max():.4f}]")
"""),
        code("""\
# Combine with a spatial constraint.
dem_arr, _ = ls.read_geotiff(DEM_PATH)
slope_vals, _ = ls.slope(dem_arr, georef, output_nodata=-9999.0)
slope_raster = ma.raster(slope_vals, georef, units="deg", name="slope")

SLOPE_MAX = 8.0        # degrees (illustrative)
SUN_MIN = 0.40         # fraction (illustrative)

candidate = (slope_raster <= SLOPE_MAX) & (mean_sun >= SUN_MIN)
print(f"Candidate cells: {candidate.values.sum()} / {candidate.values.size}")
"""),
        code("""\
_temporal_out = SCENARIO.output_path("analysis/temporal")
_temporal_out.mkdir(parents=True, exist_ok=True)

ma.write(
    str(_temporal_out / "mean_sun.tif"),
    mean_sun.expression(),
    dtype=np.float32,
    invalid_value=-9999.0,
    overwrite=True,
)
ma.write(
    str(_temporal_out / "temporal_candidate.tif"),
    candidate.expression(),
    dtype=np.uint8,
    invalid_value=0,
    overwrite=True,
)
print(f"Temporal outputs written under {_temporal_out}")
"""),
        md("""\
The materialisation boundary is visible: the temporal source is
file-backed, the reduction streams layers, `ma.compute()` creates the
spatial result, and ordinary spatial map algebra handles the candidate mask.

Use a `TemporalRasterExpression` for temporal composition and a
`RasterExpression` after a time-reducing operation produces one spatial
field.

**Try this:** replace `temporal_mean` with `temporal_min` or
`temporal_max`.  How do the candidate counts change?
"""),
        md("""\
---
## Cleanup
"""),
        code("""\
series.close()
print("Series closed.")
"""),
    ]
    return n


# ===================================================================
NOTEBOOKS = {
    "01_raster_foundations.ipynb": notebook_01,
    "02_temporal_workflows.ipynb": notebook_02,
    "03_celestial_geometry.ipynb": notebook_03,
    "04_map_algebra_foundations.ipynb": notebook_04,
    "05_suitability_and_neighborhoods.ipynb": notebook_05,
    "06_lazy_and_temporal_algebra.ipynb": notebook_06,
}


def main():
    for filename, builder in NOTEBOOKS.items():
        path = OUT / filename
        n = builder()
        nbf.write(n, str(path))
        print(f"Wrote {path}  ({len(n.cells)} cells)")
    _strip_outputs()


def _strip_outputs():
    import subprocess, sys
    files = [str(p) for p in sorted(OUT.glob("*.ipynb"))]
    subprocess.run(
        [sys.executable, "-m", "nbstripout", *files],
        check=True,
    )


if __name__ == "__main__":
    main()
