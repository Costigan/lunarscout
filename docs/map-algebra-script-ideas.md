# Map-Algebra Exploratory Script Ideas

Status: proposed script portfolio for implementation alongside the broad
map-algebra plan

Related plan: `docs/map-algebra-implementation-plan.md`

## Purpose

This document proposes scripts that illustrate, exercise, and investigate the
map-algebra capabilities planned for Lunarscout. These are intended for
`scripts/`, where maintainers can keep diagnostic output, implementation
comparisons, resource measurements, and editable local-data paths. Polished,
stable introductions to the public API should still live in `examples/` as
required by `AGENTS.md`.

The scripts should serve three related purposes:

- make scientific and metadata semantics visible on small synthetic rasters;
- provide repeatable integration and performance evidence for eager and
  file-backed execution; and
- develop realistic lunar analysis recipes that can later be distilled into
  shorter public examples.

Names below are suggestions. Numeric prefixes group scripts into a useful
learning and validation sequence but need not share numbering with
`examples/`.

## Common conventions

Every script should follow these conventions where applicable:

- Default to a deterministic synthetic lunar projected grid so the script can
  run without a network connection, SPICE kernels, private mission data, or a
  GPU.
- Accept optional input and output paths through command-line arguments. Never
  overwrite an existing output unless `--overwrite` is explicit.
- State the map-algebra phase and public operations required by the script.
  Until those operations exist, keep the idea documented rather than adding a
  script that imports placeholders or private implementation details.
- Use `import lunarscout as ls` and `ma = ls.map_algebra` for scientific work.
  A validation script may use NumPy, SciPy, Rasterio, or GDAL as an independent
  reference, but should label that code as reference or inspection logic.
- Print grids, dtypes, units, valid/invalid counts, and execution mode when
  those facts affect interpretation. Do not present array shape alone as proof
  of grid compatibility.
- Make expected failures part of the output for scripts about validation. Show
  the structured exception type, stable code, and useful details.
- Use fixed random seeds and record all parameters needed to reproduce a run.
- Keep generated artifacts under a caller-selected directory, with a safe
  default under `/tmp`, and print the exact paths produced.
- Avoid terrestrial assumptions and datasets. Physical-distance demonstrations
  must use a projected lunar grid or explicitly demonstrate rejection of an
  angular grid.
- When a script compares eager and file-backed results, compare both payload
  and canonical validity, and report the numerical tolerance for floating
  values.

## Proposed scripts

### 01 - Eager local algebra laboratory

Suggested name: `scripts/map_algebra/01_eager_local_algebra.py`

Prerequisite: Phase B.

Construct several tiny `Raster` values containing floating-point, signed
integer, unsigned integer, and Boolean data. Walk through arithmetic,
comparisons, strict Boolean operations, `where`, `coalesce`, clipping, casting,
classification, normalization, and selected math functions.

The script should print values and validity side by side, with cases covering:

- raster/raster, scalar-left, and scalar-right operations;
- invalid cells in selected and unselected `where` branches;
- first-valid ordering in `coalesce`;
- division by zero and invalid logarithm or square-root domains under every
  `numeric_errors` policy;
- integer overflow under `raise`, `wrap`, and `promote`; and
- safe, same-kind, and unsafe casts.

This is the compact semantic reference script from which focused public
examples can later be extracted.

### 02 - Units, dtypes, and structured failures

Suggested name: `scripts/map_algebra/02_units_dtypes_and_errors.py`

Prerequisite: Phase B.

Create slope in degrees, illumination as a fraction, elevation in metres, and a
dimensionless classification. Demonstrate legal scalar thresholds and matching
unit arithmetic, followed by deliberately rejected combinations such as adding
degrees to metres, multiplying two unit-bearing rasters without declared output
units, applying trigonometry to an unknown unit, and unsafe integer casts.

For each rejected operation, print the structured exception code and details.
For accepted operations, print the inferred output dtype and units. Include
boundary values for every supported integer width, especially `uint64` values
beyond exact `float64` representation.

### 03 - Validity and GeoTIFF mask round trip

Suggested name: `scripts/map_algebra/03_validity_mask_roundtrip.py`

Prerequisites: Phases B and D.

Build a raster where zero is valid science data, a separate set of pixels is
invalid, and the invalid payload happens to contain plausible numbers. Apply a
small expression using `set_invalid`, `fill_invalid`, `where`, and `coalesce`,
then write integer and floating GeoTIFF variants.

Reopen the files through both Lunarscout and Rasterio/GDAL and report:

- native payload values;
- dataset and band mask values;
- declared nodata metadata;
- validity provenance; and
- whether valid zero remained visible while invalid pixels remained masked.

An optional `--write-qgis-style` output can produce a small artifact intended
for the separate QGIS inspection script below.

### 04 - Grid safety and explicit alignment

Suggested name: `scripts/map_algebra/04_grid_safety_and_alignment.py`

Prerequisites: Phase B for rejection demonstrations and Phase C or later for
expression resampling.

Generate rasters with equal shapes but shifted origins, differing CRS values,
anisotropic pixels, and rotated affine transforms. Show that direct algebra
rejects mismatched grids before numerical calculation. Then align or resample
explicitly and repeat the calculation.

Add coordinate rasters for row, column, projected x/y, longitude, and latitude.
Print selected pixel-center coordinates to make anchor, axis order, and unit
conventions inspectable. The script should explicitly demonstrate that no
WGS84 or north-up assumption is inserted.

### 05 - Classification and layer combination

Suggested name: `scripts/map_algebra/05_classification_and_stacks.py`

Prerequisite: complete Phase B classification and stack inventory.

Create synthetic slope, roughness, illumination, and hazard layers. Exercise
`reclassify_values`, `reclassify_ranges`, `digitize`, and `one_hot`, including
each unmatched-value policy. Combine layers using `sum_layers`, `mean_layers`,
`min_layers`, and `max_layers`.

The output should make dtype promotion, units, invalidity, and the difference
between an ordinary zero class and an invalid cell explicit. Include sparse,
negative, and large class identifiers where the operation supports them.

### 06 - Terrain and lighting candidate screening

Suggested name: `scripts/map_algebra/06_candidate_screening.py`

Prerequisite: Phase B; add file-backed mode after Phase D.

Implement the plan's central landing-site workflow using explicit slope,
roughness, sunlight, Earth visibility or elevation, and hazard inputs. Build:

- individual threshold masks;
- a combined candidate mask;
- normalized component scores;
- a caller-supplied weighted score; and
- a final score that is invalid outside the candidate mask.

Print the surviving pixel count after each criterion so a user can see which
policy removes each candidate. Support `--mode eager` initially and
`--mode file-backed` when expression execution is available. The two modes
should use the same scientific formula and compare their results.

### 07 - Expression explanation, identity, and planning

Suggested name: `scripts/map_algebra/07_expression_explain_and_plan.py`

Prerequisite: Phase C.

Build a moderately deep expression from file sources and one eager in-memory
constant. Without executing it, print:

- the concise human description;
- canonical JSON;
- scientific, restart, and execution-cache identities;
- inferred grid, dtype, units, validity rule, and halo;
- source identities and operation versions; and
- the dry-run execution plan and estimated resources.

Create a few controlled variants to show which changes alter scientific
identity, which affect restart compatibility, and which only change execution
cache identity. Assert that `explain()` and `plan()` create no output or staging
files.

### 08 - Windowed local parity and fusion

Suggested name: `scripts/map_algebra/08_windowed_local_parity.py`

Prerequisite: Phase C.

Generate a non-128-multiple raster with invalid patches crossing likely window
boundaries. Evaluate a multi-source, five-or-more-node local expression eagerly
and with several file-backed block/window sizes.

Compare payload, validity, dtype, units, and georeferencing. Print the planner's
fusion decisions and source-read counts so repeated sources and fused local
nodes are visible. Include integer/Boolean exact comparisons and floating-point
comparisons with stated tolerances.

### 09 - Durable write, cancellation, and resume

Suggested name: `scripts/map_algebra/09_durable_write_resume.py`

Prerequisite: Phase D.

Run a many-window expression in a child process, terminate it after a known
number of journaled windows, inspect the staging manifest, and resume. Compare
the published result with a clean uninterrupted run.

Separate subcommands should explore cancellation, restart mismatch,
`start_fresh`, overwrite protection, and failed replacement of an existing
complete output. The script must use only exact, resolved staging paths and
must report whether the previous output remained intact after each injected
failure.

### 10 - Focal edge and halo explorer

Suggested name: `scripts/map_algebra/10_focal_edges_and_halos.py`

Prerequisite: Phase E.

Create an impulse raster, a ramp, and a raster with an invalid island. Run focal
sum/mean/range/standard deviation, median or percentile, and convolution with
several footprints. Produce small textual matrices or optional images showing
the effects of:

- rectangular versus explicit footprints;
- `invalid`, `constant`, `nearest`, `reflect`, and `wrap` edge modes;
- `require_all`, `ignore_invalid`, and `propagate_center` validity policies;
  and
- `min_valid_count`.

Repeat selected cases window by window and compare them with an independent
whole-array SciPy reference, emphasizing cells on internal tile boundaries.

### 11 - Morphology and connected candidate regions

Suggested name: `scripts/map_algebra/11_morphology_and_regions.py`

Prerequisite: Phase E.

Start with a noisy Boolean candidate mask containing holes, narrow bridges,
isolated pixels, and invalid areas. Demonstrate dilation, erosion, opening,
closing, majority filtering, region labeling, border finding, and filtering by
region size with four- and eight-neighbor connectivity.

Report how each operation changes valid area, candidate area, number of
regions, and retained region sizes. Compare the Raster adapters with the
existing array-oriented region functions to document compatibility.

### 12 - Terrain-node window-boundary comparison

Suggested name: `scripts/map_algebra/12_terrain_expression_parity.py`

Prerequisite: Phase E terrain-node decision.

Evaluate slope, aspect, and hillshade on synthetic planar, conical, ridge, and
nodata-interrupted DEMs. Compare existing eager terrain functions with
map-algebra eager and file-backed nodes at multiple window sizes.

Concentrate diagnostics on outer edges, internal window seams, rotated grids,
anisotropic pixels, and `compute_edges`. If an operation remains eager-only,
the script should demonstrate the explicit rejection of a file-backed request
rather than materializing silently.

### 13 - Global streaming reductions

Suggested name: `scripts/map_algebra/13_global_reductions.py`

Prerequisite: Phase F.

Generate a reproducible raster distribution with invalid cells, outliers, and
known analytic statistics. Compare eager and streaming statistics, histograms,
unique counts, and exact/approximate percentiles.

Print accumulator dtype, pass count, approximation settings and error, peak
memory, and empty-valid-domain behavior. Exercise the `max_unique` safety bound
and show that reduction results are scalars or result objects rather than
one-cell georeferenced rasters.

### 14 - Zonal candidate summaries

Suggested name: `scripts/map_algebra/14_zonal_candidate_summary.py`

Prerequisite: Phase F.

Label candidate regions and summarize slope, illumination, and score by zone.
Include zone zero, negative and sparse zone IDs, a large 64-bit ID, an invalid
zone, an all-invalid value zone, and explicitly requested empty zones.

Display deterministic row ordering and per-column validity, serialize the
result to records, dictionaries, JSON, and CSV, then broadcast one selected
statistic back through `zonal_raster`. Verify that large integer IDs survive
every conversion exactly.

### 15 - Hazard-clearance distance fields

Suggested name: `scripts/map_algebra/15_hazard_clearance.py`

Prerequisite: Phase G.

Create synthetic rock, crater-rim, and exclusion masks, then calculate
Euclidean, taxicab, chessboard, and signed distances. Apply a caller-selected
clearance threshold to the candidate-screening result.

Use analytic single-seed and rectangular-mask cases before a larger synthetic
field. Compare square, anisotropic, and rotated projected grids, invalid output
policies, `max_distance`, no-seed/all-seed behavior, and SciPy where its
assumptions match. Also construct an angular grid and print the structured
rejection of an unconfigured physical-distance request.

### 16 - Temporal threshold and static-map composition

Suggested name: `scripts/map_algebra/16_temporal_threshold_summary.py`

Prerequisite: Phase H.

Create or open a timestamped illumination series, apply a layer-wise threshold,
and reduce it to count, fraction, duration, minimum, mean, maximum, and
variability rasters. Combine the resulting spatial expression with a static
slope or hazard raster without first writing an intermediate mean raster.

Show exact UTC coordinate validation, explicit temporal alignment failures,
static-raster broadcasting, invalid samples, time batching, bounded dataset
handles, and the distinction between sample counts and interval duration.

### 17 - QGIS mask inspection artifact

Suggested name: `scripts/map_algebra/17_qgis_mask_inspection.py`

Prerequisite: Phase D; optionally consume outputs from script 03 or 06.

Write a small, visually obvious GeoTIFF containing valid zeros, valid nonzero
classes, and invalid pixels. Print Rasterio/GDAL inspection commands and a
short QGIS checklist covering transparency, class rendering, CRS, extent,
dtype, nodata, and mask interpretation.

This script should not attempt GUI automation. Its purpose is to create a
stable artifact and record the expected observations for human inspection.

### 18 - Human-reviewed analysis proposal

Suggested name: `scripts/map_algebra/18_review_before_execute.py`

Prerequisites: Phases C and D.

Model the "assistant proposes, human reviews, library validates" workflow
without adding assistant logic to Lunarscout. Construct a candidate expression
from a fixed, auditable configuration; print `explain()` and `plan()`; require
an explicit command-line confirmation flag before calling `write()`; and print
the final provenance identity.

The script should also include rejected configurations for an unknown source,
implicit alignment, incompatible units, excessive graph depth, and unsafe
output encoding. Authorization remains entirely in the script, while
Lunarscout supplies validation and execution.

### 19 - Bounded-memory scaling benchmark

Suggested name: `scripts/map_algebra/benchmark_resource_scaling.py`

Prerequisite: the relevant file-backed operation phases.

Run representative local, focal, global, zonal, distance, and temporal
workloads at three or more raster sizes while holding window size and worker
count constant. Record machine-readable timing and resource results plus a
short human-readable interpretation.

Capture the benchmark metadata required by the implementation plan: input
dimensions, dtype, valid fraction, block size, compression, backend, worker
count, dependency versions, hardware, storage, cold/warm state, elapsed-stage
times, peak resident memory, and temporary/output disk usage. This is evidence
for scaling behavior, not a universal throughput promise.

## Shared support assets

The portfolio would benefit from a small support module under
`scripts/map_algebra/`, used only by these scripts. It could provide:

- deterministic projected lunar `GeoReference` fixtures, including north-up,
  rotated, anisotropic, shifted, and angular variants;
- synthetic DEM, slope-like, illumination, hazard, zone, and temporal-series
  generators;
- concise payload/validity printing;
- result comparison that reports payload, validity, grid, dtype, and unit
  differences separately;
- optional process peak-memory sampling and machine metadata capture; and
- output-directory creation and overwrite preflight.

The support module should not become public API, duplicate production
algorithms, hide important scientific parameters, or be imported by
`src/lunarscout`.

## Suggested implementation order

Implement scripts only after their required public surface is accepted:

1. Add scripts 01-06 as the eager semantic and workflow set.
2. Add scripts 07-09 when expression planning and durable output are complete.
3. Add scripts 10-12 with focal, morphology, and terrain expression support.
4. Add scripts 13-14 with global and zonal reductions.
5. Add script 15 after distance semantics are frozen.
6. Add script 16 after temporal adapters and reducers are accepted.
7. Add scripts 17-19 as release-candidate inspection, governance, and resource
   evidence.

Candidates for promotion into `examples/` should be short, public-API-only,
CPU-runnable where possible, and stable enough to maintain as user-facing
documentation. The candidate-screening, morphology, zonal-summary,
hazard-clearance, temporal-summary, and review-before-execute scripts map
directly to the example families required by Phase I of the implementation
plan.
