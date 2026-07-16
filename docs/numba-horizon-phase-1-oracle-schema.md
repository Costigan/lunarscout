# Numba Horizon Phase 1 Reference-Ray Oracle Schema

**Schema version:** 1

**C# baseline commit:** `f3b21b5a7d510162783c8e6a1aa01ca2edc61277`

The Phase 1 oracle artifact combines deliberately simple C# reference-ray
results with selected production ILGPU intermediates. It is intended to
validate Python coordinate geometry, ray direction, terrain sampling, spherical
slope calculation, nested-DEM distance continuity, production ray-sample
placement, polynomial fitting, and max-pyramid construction before production
kernel parity is attempted.

The artifact pair is:

- `tests/data/numba_horizon/phase1_reference_rays.npz`
- `tests/data/numba_horizon/phase1_reference_rays.json`

The JSON file is the schema and manifest. It records source commit, conventions,
fixture definitions, analytical expectations, every array's dtype, shape, axes,
units, and SHA-256 over canonical C-order data. The NPZ contains only numeric
arrays and never requires pickling. ZIP member timestamps and permissions are
fixed so identical inputs produce a byte-identical NPZ.

## Captured Cases

The 27-case synthetic set contains:

- flat spherical terrain toward east;
- one east-side obstacle viewed toward east and west;
- one northeast obstacle viewed toward northeast and southwest, exercising both
  fitted pixel axes;
- an inner flat DEM alone;
- nested inner/outer DEMs with an obstacle in the outer DEM;
- flat terrain at 10 and 30 m/pixel;
- horizon-setting obstacles in every cardinal and intercardinal direction;
- near/lower and far/higher peaks, a ridge across adjacent quarter-degree
  bins, negative elevations with an elevated observer, and obstacles on both
  sides of the 500 m calculation threshold;
- nodata holes, borders, and an entirely nodata ray;
- boundary observers, partial 63 by 47 dimensions, and multi-DEM coverage at
  30 and 60 m/pixel.

A separate pyramid-only 7 by 5 fixture contains NaN, positive and negative
infinity, values at and below -20,000 m, a valid -19,999 m sample, and an
all-invalid factor-four block. It does not run through the reference ray
emulator.

These cases freeze independent reference behavior and the production CUDA
pyramid for every captured DEM. Two subpatch fixtures cover boundary
interpolation and materially nonzero grid convergence (about 0.0414 rad).

A bounded subpatch fixture captures all 16 halo-inclusive centers for a 16 by
16 corner tile with 8-pixel subpatches, 16 azimuth bins, and two DEMs. Its 512
segments retain the production `[azimuth][subpatch_center][DEM]` ordering. The
negative halo clamps onto the first legal center, so duplicate-center segment
records provide an exact internal consistency check. Grid-convergence center
and per-pixel gradients are captured but, consistent with the Phase 0 finding,
the current production subpatch kernel does not apply them.

A production buffer fixture runs one pixel through all 1,440 azimuth bins and
two DEM passes. The second DEM is 1,025 by 1,025 pixels and has a far synthetic
obstacle that forces hierarchy level-one descent. The artifact stores both
unmerged pass buffers, the elementwise maximum, the final degree conversion,
and a 925-step trace emitted by the CUDA kernel for east bin 360 on DEM pass 1.
The trace records parameter and true distance, hierarchy level, cell, pixel,
block maximum, sampled elevation and slope, advance, and action code.

Real-terrain fixtures are deliberately separate from this compact NPZ. NASA
LOLA acquisition/provenance and the gated four-DEM local production stack are
defined in `docs/numba-horizon-phase-1-real-terrain-fixtures.json`.

## Array Names

DEM arrays use:

```text
<case>__dem_<index>__elevation_m
<case>__dem_<index>__geo_transform
```

Per-DEM reference passes use:

```text
<case>__pass_<index>__slopes
<case>__pass_<index>__trace_distance_m
<case>__pass_<index>__trace_elevation_m
<case>__pass_<index>__trace_slope
<case>__pass_<index>__trace_pixel_x
<case>__pass_<index>__trace_pixel_y
<case>__pass_<index>__direction_me
```

Nominal-bin production ray fitting uses:

```text
<case>__ray_fit_pass_<index>__sample_distance_m
<case>__ray_fit_pass_<index>__sample_pixel_x
<case>__ray_fit_pass_<index>__sample_pixel_y
<case>__ray_fit_pass_<index>__sample_latitude_rad
<case>__ray_fit_pass_<index>__sample_longitude_rad
<case>__ray_fit_pass_<index>__sample_row
<case>__ray_fit_pass_<index>__sample_column
<case>__ray_fit_pass_<index>__sample_terrain_height_m
<case>__ray_fit_pass_<index>__observer_vector_moon_centered_m
<case>__ray_fit_pass_<index>__nominal_direction_moon_centered
<case>__ray_fit_pass_<index>__segment_values
```

`segment_values` is a `float32` vector whose ordered field names and per-field
units are stored in the matching JSON `ray_fit_passes` entry. `dem_id` remains
a JSON integer. This is a structured language-neutral contract, not a dump of
the C# struct's in-memory bytes. Samples and host geometry remain `float64`;
the segment vector captures the production conversion to `float32`.

The capture assembly can call existing internal sample and fit functions through
an explicit `InternalsVisibleTo` entry and a narrow diagnostic wrapper around
`FitRaySegment`. The wrapper delegates directly to the production fitter and is
not used by the production pipeline.

Production pyramid arrays use:

```text
<case>__dem_<index>__pyramid__level_<level>
<case>__dem_<index>__pyramid__level_metadata
<case>__dem_<index>__pyramid__level_cell_sizes
<case>__dem_<index>__pyramid__map_parameters
<case>__dem_<index>__pyramid__projection_parameters
```

Level 0 is stored separately by production. Levels above zero use offsets into
one concatenated mip buffer; the metadata array columns are level, offset,
width, and height. Dimensions use ceiling division by four until both reach
one. The capture runs `BuildOrLoadPyramid` on the selected CUDA accelerator,
then the validator independently reconstructs every level from the preceding
array with the documented finite-and-greater-than-minus-20,000 rule. Dedicated
invalid-value assertions verify the -32,000 m all-invalid-block sentinel and
the strict cutoff boundary.

The bounded subpatch fixture uses:

```text
<fixture>__subpatch_fixture__centers
<fixture>__subpatch_fixture__grid_convergence
<fixture>__subpatch_fixture__segment_values
<fixture>__subpatch_fixture__segment_dem_ids
```

`centers` records grid indices, requested halo centers, and clamped segment
centers. `segment_values` has axes `(azimuth, subpatch_center, dem,
segment_field)` and uses the same named `float32` segment fields as the ray-fit
artifact. `segment_dem_ids` independently checks the final DEM axis and flatten
order.

Production horizon-buffer arrays use:

```text
<fixture>__horizon_buffer_fixture__per_dem_slopes
<fixture>__horizon_buffer_fixture__final_slopes
<fixture>__horizon_buffer_fixture__final_degrees
<fixture>__horizon_buffer_fixture__grid_convergence
<fixture>__horizon_buffer_fixture__traversal_trace
```

The trace field order and action-code mapping live in the JSON fixture metadata.
Action codes distinguish descent, culling, nodata skip, out-of-bounds exit, and
level-zero sampling. The validator proves the maximum recorded sample slope is
the selected per-DEM CUDA output bin.

Raster axes are `(y, x)`. Reference-ray trace arrays share a one-dimensional
`sample` axis; the production traversal trace uses `(step, traversal_field)`.
Coordinates use X for pixel column and Y for pixel row. Azimuth is degrees
clockwise from true north. Slopes are dimensionless rise/run values; angles are
derived with `atan(slope)`.

## Regeneration and Validation

From the prototype worktree, using the repository virtual environment:

```bash
/e/projects/lunarscout/.venv/bin/python \
  scripts/numba_horizon/capture_phase1_reference_oracles.py \
  --baseline-commit f3b21b5a7d510162783c8e6a1aa01ca2edc61277

/e/projects/lunarscout/.venv/bin/python \
  scripts/numba_horizon/validate_phase1_reference_oracles.py
```

Regeneration is an explicit developer action. Ordinary tests should load the
immutable artifact and must not invoke moonlib or silently regenerate missing
data. Regeneration of the current schema requires a visible CUDA device because
it captures the production ILGPU pyramid path; loading and validating the
artifact does not initialize CUDA.
