# Map-Algebra Example Ideas

Status: proposed user-facing example portfolio for the broad map-algebra API

Related plan: `docs/map-algebra-implementation-plan.md`

## Purpose

This document proposes examples for `examples/` that help users explore and
understand Lunarscout's planned map-algebra capabilities. The examples should
work as ordinary Python programs run directly with Python and should also be
easy to follow, copy into Jupyter, or adapt into marimo notebooks.

The portfolio should teach map algebra progressively:

1. construct an eager spatial raster and understand its metadata;
2. combine registered rasters without managing masks by hand;
3. build and inspect lazy file-backed expressions;
4. apply focal, regional, global, distance, and temporal operations; and
5. assemble those tools into auditable lunar mission-analysis workflows.

These examples are not tests or benchmarks. Small assertions may confirm facts
being demonstrated, but the emphasis should be on readable public API usage,
scientific interpretation, and useful outputs that invite experimentation.

## Example design principles

Each example should follow these principles where applicable:

- Use only public Lunarscout APIs for the analysis itself, normally through:

  ```python
  import lunarscout as ls

  ma = ls.map_algebra
  ```

- Run from the command line with the repository virtual environment and no
  interactive environment required:

  ```bash
  PYTHONPATH="$PWD/src" .venv/bin/python examples/18_map_algebra_basics.py
  ```

- Organize code into short conceptual sections with intermediate variables.
  A reader should be able to copy those sections into notebook cells without
  untangling command-line or filesystem machinery.
- Put execution in a `main()` function, while keeping analysis helpers small
  and independently callable. This works well for direct execution and gives
  notebook users reusable pieces to import.
- Use deterministic synthetic lunar rasters by default. An example may accept
  an optional real DEM or scenario path, but it must say what the user needs to
  provide and must not silently download terrestrial data.
- Prefer small synthetic inputs for eager examples so users can print arrays
  and immediately see validity behavior. Use larger generated or downloadable
  lunar inputs only where file-backed execution is the lesson.
- Print or plot the result at each meaningful stage. Show values, validity,
  units, grid information, and operation choices when they affect scientific
  interpretation.
- Preserve a clear distinction between invalid data and valid values such as
  zero, `False`, or class ID zero.
- Demonstrate structured failures when they teach an important contract, such
  as grid mismatch or incompatible units. Do not turn every example into an
  error catalogue.
- Explain whether an operation is eager, lazy, windowed, multi-pass, or
  materializing. Users should know when a line computes immediately and when it
  only describes future work.
- Default output to a safe example workspace and require `--overwrite` before
  replacing files.
- Keep GPU and SPICE requirements out of core map-algebra examples. If an
  example optionally consumes horizon, lighting, or body-vector products,
  explain how to use pregenerated inputs and when CUDA or SPICE was needed to
  create them.
- Avoid embedded mission policy. Thresholds, weights, invalid-area treatment,
  connectivity, distance units, and similar decisions should be visible user
  inputs.

## Suggested progression

The names and numbers below are provisional. They continue after the current
example sequence, but final numbering should be chosen when each example is
implemented.

### 18 - Raster and local-algebra basics

Suggested file: `examples/18_map_algebra_basics.py`

Introduce `Raster` using a small synthetic slope grid. Print its values,
validity mask, grid, units, shape, dtype, valid count, and memory size. Show
non-mutating helpers such as `with_name`, `with_units`, `with_validity`,
`filled`, `masked`, `copy`, and `readonly` where useful.

Then demonstrate a compact selection of eager local operations:

- arithmetic with scalars and another raster;
- comparisons that return Boolean rasters;
- strict Boolean combinations using `&`, `|`, and `~`;
- `clip`, `minimum`, and `maximum`;
- one domain-sensitive function such as `sqrt` or `log`; and
- `is_valid` and `is_invalid`.

Use printed two-dimensional arrays so users can predict each result. End with a
short reminder that Python `and` and `or` do not perform raster algebra and
that NumPy remains appropriate for non-spatial arrays.

Notebook exploration ideas:

- Change a threshold and rerun the candidate-mask cell.
- Change one input validity value and observe how it propagates.
- Compare `raster.values` arithmetic with registered map algebra.

### 19 - Validity, `where`, and `coalesce`

Suggested file: `examples/19_map_algebra_validity.py`

Focus on canonical validity without using a masked array as the internal data
model. Construct partially valid illumination and Earth-visibility rasters,
then demonstrate:

- strict validity intersection for ordinary operations;
- selected-branch validity in `where`;
- `ma.invalid` as an explicit invalid branch;
- first-valid selection with `coalesce`;
- changing validity with `set_invalid`;
- turning invalid cells into valid filled values with `fill_invalid`; and
- converting to a NumPy masked array only when needed by another tool.

Include valid zeros and invalid cells whose payloads contain plausible values.
The example should make it visually obvious that payload alone does not decide
validity.

Notebook exploration ideas:

- Reverse the order of `coalesce` inputs.
- Compare invalidity in the selected and unselected `where` branches.
- Fill invalid pixels with several values and inspect dtype constraints.

### 20 - Grid compatibility and explicit alignment

Suggested file: `examples/20_map_algebra_grids.py`

Create two same-shaped rasters that represent different places because one has
a shifted affine transform. Show `same_grid`, `require_same_grid`, and the
structured failure produced by direct algebra. Align one raster explicitly and
then combine it with the other.

Extend the example with projected x/y or row/column coordinate rasters. If the
longitude and latitude constructors are implemented, display a few pixel-center
coordinates while making clear that the input grid's geodetic CRS is used and
WGS84 is not assumed.

Notebook exploration ideas:

- Change the affine shift from a whole pixel to a partial pixel.
- Compare nearest and continuous-data resampling.
- Try an anisotropic or rotated synthetic grid.

### 21 - Units, dtypes, and numerical policies

Suggested file: `examples/21_map_algebra_numerics.py`

Use elevation in metres, slope in degrees, and illumination as a fraction to
explain Lunarscout's conservative unit rules. Demonstrate matching-unit
addition or comparison, scalar thresholds in the raster's units, explicit
output units for multiplication or division of unit-bearing rasters, and angle
handling in trigonometric functions.

Use small integer arrays to show dtype promotion and the difference between
`overflow="raise"`, `"wrap"`, and `"promote"`. Demonstrate safe and unsafe
casts, followed by `numeric_errors="invalid"`, `"keep"`, and `"raise"` on a
division or logarithm.

This example should favor a few carefully explained cases over a full dtype
matrix.

Notebook exploration ideas:

- Change an input dtype and inspect the inferred result dtype.
- Deliberately combine incompatible units and inspect the structured error.
- Compare numerical-error policies at the same pixel.

### 22 - Classification and weighted suitability

Suggested file: `examples/22_map_algebra_suitability.py`

Build a complete eager landing-site suitability calculation from synthetic
slope, roughness, illumination, Earth visibility, and hazard rasters. Show:

- threshold masks for hard constraints;
- `reclassify_values`, `reclassify_ranges`, or `digitize` for interpretable
  classes;
- normalization of continuous criteria;
- a caller-supplied weighted score;
- Boolean exclusion of hazard pixels; and
- `where(candidate, score, ma.invalid)` for the final result.

Print how many pixels survive each criterion and plot the inputs, candidate
mask, and score. Clearly label all thresholds and weights as illustrative
choices rather than Lunarscout recommendations.

This should become the main notebook-friendly map-algebra introduction because
it answers a realistic question while remaining small enough to understand in
one sitting.

Notebook exploration ideas:

- Expose thresholds and weights as variables in an early cell.
- Plot how candidate area changes with the slope threshold.
- Compare weighted sum, minimum-component, and hard-constraint approaches.

### 23 - Lazy expressions, explanation, and planning

Suggested file: `examples/23_map_algebra_expressions.py`

Repeat the scientific formula from the suitability example using
`ma.source()` instead of `ma.read()`. Explain that the Python expression builds
an immutable graph without opening every dataset or calculating the regional
output.

Show:

- `Raster` versus `RasterExpression`;
- a concise expression description;
- `ma.explain()` in human-readable and machine-readable form;
- `ma.plan()` as a dry run;
- inferred grid, dtype, units, validity behavior, and halo;
- repeated-source reuse and local-operation fusion; and
- explicit `compute()` versus bounded `write()`.

Keep the expression small enough that the printed explanation is readable.
Verify visibly that planning does not create the requested output file.

Notebook exploration ideas:

- Change one threshold and compare expression identities.
- Mix one eager `Raster` with file-backed sources.
- Compare `compute()` with `write()` and discuss memory implications.

### 24 - Writing and resuming a regional expression

Suggested file: `examples/24_map_algebra_file_backed.py`

Use a moderately sized synthetic or downloadable lunar raster to demonstrate
bounded file-backed execution. Build a multi-source local expression, print the
plan, write a tiled GeoTIFF, reopen it, and inspect its grid, dtype, units,
validity mask, and provenance.

Demonstrate ordinary user-facing output controls:

- output preflight;
- progress reporting;
- overwrite protection;
- cancellation with a clean incomplete state; and
- resuming a compatible interrupted calculation.

The example should explain windowed processing conceptually without exposing
private journal or staging implementation details.

Notebook exploration ideas:

- Display progress in a notebook-friendly callback.
- Change the output window or worker configuration and compare plans.
- Reopen and plot a small output window instead of loading the complete file.

### 25 - Focal statistics and neighborhood choices

Suggested file: `examples/25_map_algebra_focal.py`

Create a synthetic rough surface with an invalid patch. Compare focal mean,
range, standard deviation, median or percentile, and a simple convolution.
Plot the source and outputs using the same color scale where appropriate.

Use a few selected pixels to explain:

- rectangular windows and explicit footprints;
- edge modes;
- `require_all`, `ignore_invalid`, and `propagate_center` policies;
- `min_valid_count`; and
- the distinction between a pixel neighborhood and a physical-radius
  operation.

If file-backed focal execution is available, run the same formula in eager and
file-backed modes and state that halo handling prevents tile seams.

Notebook exploration ideas:

- Draw a custom cross-shaped footprint.
- Increase window size and observe smoothing and edge effects.
- Compare ignoring invalid neighbors with requiring all neighbors.

### 26 - Cleaning and measuring candidate regions

Suggested file: `examples/26_map_algebra_regions.py`

Start from the noisy candidate mask produced by the suitability workflow.
Demonstrate dilation, erosion, opening, closing, majority filtering, connected
region labeling, filtering by minimum size, and border extraction.

Plot each cleanup step and report the number and area of retained regions.
Show the difference between four- and eight-neighbor connectivity. Relate the
Raster adapters to Lunarscout's existing array-oriented region functions so
current users can see how the APIs fit together.

Notebook exploration ideas:

- Adjust the minimum region size interactively.
- Compare operation order, such as opening then closing versus the reverse.
- Highlight borders over the original suitability score.

### 27 - Terrain operations in map algebra

Suggested file: `examples/27_map_algebra_terrain.py`

Use a small synthetic lunar DEM to calculate slope, aspect, and hillshade as
registered map-algebra operations. Combine slope with a focal roughness-like
measure and a synthetic lighting raster to produce a terrain-screening result.

Explain one-pixel terrain halos, edge behavior, elevation units, anisotropic
pixel spacing, and how validity near DEM gaps propagates. If some terrain
operations remain eager-only, show the supported path and state the limitation
plainly.

Notebook exploration ideas:

- Change illumination azimuth/elevation for hillshade.
- Compare a planar DEM, bowl, ridge, and cone.
- Overlay the candidate mask on hillshade.

### 28 - Zonal summaries of candidate regions

Suggested file: `examples/28_map_algebra_zonal.py`

Label cleaned candidate regions and calculate zonal count, valid/invalid count,
mean slope, minimum illumination, maximum score, standard deviation, and a
selected percentile. Display the resulting `ZonalStatistics` as records and
write CSV and JSON versions.

Show that zone zero is ordinary unless configured otherwise, invalid zone cells
belong to no zone, and undefined statistics use per-column validity. Broadcast
one statistic back to a raster with `zonal_raster` and plot it.

Notebook exploration ideas:

- Sort or filter the returned immutable records for presentation.
- Rank regions using several zonal criteria.
- Request an empty zone ID and inspect count versus mean validity.

### 29 - Global summaries and distributions

Suggested file: `examples/29_map_algebra_global_stats.py`

Summarize a suitability or terrain raster using `statistics`, `histogram`,
`unique_counts`, and exact or approximate percentiles. Explain why reductions
return scalar or result values rather than a one-cell georeferenced raster.

Run the same summary eagerly on a small raster and by streaming a larger file.
Report count, invalid count, accumulator precision, percentile method, and any
explicit memory/accuracy choice.

Notebook exploration ideas:

- Plot a histogram with invalid pixels excluded.
- Compare exact and approximate percentiles.
- Use derived statistics as explicit inputs to normalization.

### 30 - Hazard proximity and clearance

Suggested file: `examples/30_map_algebra_distance.py`

Create or load a Boolean hazard mask and calculate distance to hazards in pixel
and projected physical units. Demonstrate Euclidean distance, pixel-unit
taxicab or chessboard distance, signed distance, and `max_distance` clipping.

Apply a user-selected clearance threshold to the suitability mask and plot the
distance field, excluded buffer, and surviving candidates. Use an anisotropic
projected lunar grid to explain how physical distance differs from pixel
distance. Include a short structured-error demonstration showing why physical
distance on an unconfigured angular grid is rejected.

Clearly state that straight-line proximity is not route planning, accumulated
cost, energy modelling, or traverse policy.

Notebook exploration ideas:

- Move or add hazard seeds.
- Sweep the clearance threshold and plot remaining candidate area.
- Compare signed distance with separate inside/outside masks.

### 31 - Temporal map algebra for illumination

Suggested file: `examples/31_map_algebra_temporal.py`

Create or open a small timestamped illumination series as a
`TemporalRasterExpression`. Apply a layer-wise threshold and calculate temporal
mean, minimum, maximum, standard deviation, exceedance count, and threshold
duration.

Combine the resulting spatial expression with static slope and hazard rasters
without writing an intermediate temporal reduction. Explain exact UTC
coordinate matching, static-raster broadcasting, time batching, invalid
samples, and the difference between sample counts and interval durations.

Notebook exploration ideas:

- Select a shorter time interval.
- Change the illumination threshold.
- Plot a temporal trace for one pixel beside the reduced spatial map.

### 32 - End-to-end lunar candidate exploration

Suggested file: `examples/32_map_algebra_candidate_explorer.py`

Combine the strongest ideas above into a single realistic workflow using a
synthetic scenario or documented downloadable lunar products:

1. open slope, roughness, illumination, Earth visibility, and hazard sources;
2. validate grids and units;
3. build hard constraints and a weighted score;
4. clean the candidate mask with morphology;
5. apply hazard clearance;
6. label candidate regions;
7. calculate zonal summaries;
8. explain and dry-run the expression;
9. write selected outputs; and
10. plot a compact decision dashboard.

All thresholds, weights, connectivity, footprint, clearance, and invalid-area
choices should be collected near the top of the file. The example should call
them illustrative analysis parameters, not mission recommendations.

For Jupyter, those parameters become a natural configuration cell. For marimo,
they can become sliders or selectors whose downstream plots update reactively.
The checked-in Python example should not require either notebook environment.

### 33 - Inspecting map-algebra output in QGIS

Suggested file: `examples/33_map_algebra_qgis.py`

Write a small set of GeoTIFF outputs designed for external inspection: a
continuous score, Boolean candidate mask encoded as `uint8`, labeled regions,
and a distance field. Ensure each contains valid zero values as well as invalid
pixels.

Print output locations and a short QGIS checklist covering CRS, extent,
transparency from the dataset mask, class rendering, units, and provenance
metadata. The script should create the artifacts but should not automate QGIS.

This closes the loop between in-memory analysis and ordinary geospatial desktop
inspection.

### 34 - Review an analysis before executing it

Suggested file: `examples/34_map_algebra_review_then_run.py`

Demonstrate the plan's "assistant proposes, human reviews, library validates"
pattern without putting assistant or authorization logic inside Lunarscout.
Construct a fixed candidate expression from explicit configuration, print
`explain()` and `plan()`, and require an explicit `--run` flag before writing
the result.

Show how validation catches an incompatible grid, unit mismatch, unavailable
source, or unsafe output encoding before execution. Finish by printing the
scientific provenance identity of the accepted output.

This example is useful even without an AI assistant: it teaches every user to
inspect a substantial calculation before committing time and storage to it.

## Supporting data and presentation

The map-algebra examples should share small deterministic data helpers through
the existing `examples/_example_support.py` or a focused companion module.
Useful synthetic assets include:

- a projected lunar grid with north-up, rotated, and anisotropic variants;
- a DEM containing a plane, bowl, ridge, cone, and nodata gap;
- slope, roughness, illumination, Earth-visibility, and hazard rasters derived
  from simple documented formulas;
- noisy candidate masks with holes and disconnected regions;
- zone rasters with ordinary zero, sparse IDs, and invalid cells; and
- a short UTC illumination series with known outages and invalid samples.

Helpers should reduce setup noise without hiding the operation being taught.
Generated inputs should be reproducible and small enough for CPU execution.
Examples that optionally use real lunar products should reuse the repository's
download-and-cache conventions and list data requirements in
`examples/README.md`.

Plots should use consistent conventions where practical:

- invalid pixels transparent or visibly hatched;
- Boolean masks shown with distinct false and true colors;
- continuous inputs and derived outputs labelled with units;
- shared color limits when comparing like quantities; and
- titles that include the scientifically important threshold, footprint, or
  distance choice.

Matplotlib should remain optional unless plotting is central to the example.
Every important result should also have a useful textual summary for direct
command-line users.

## Jupyter and marimo use

The checked-in source of truth should remain executable `.py` files under
`examples/`. They should be notebook-friendly without requiring duplicate
`.ipynb` files that are difficult to review.

To support Jupyter users:

- separate imports, data construction, analysis, and plotting into clear code
  blocks divided by comments;
- avoid relying on mutable global state established only by `main()`;
- return useful objects from helpers rather than only printing them; and
- include a note in each docstring identifying the most useful blocks to copy
  into cells.

To support marimo users:

- keep analysis functions pure where practical;
- collect user choices in explicit variables or a small immutable
  configuration value;
- avoid hidden filesystem side effects when a reactive cell reruns; and
- put file writing behind an explicit action or function call.

A later documentation task may add one Jupyter notebook and one marimo app that
compose these public helpers. Those should be generated or maintained
deliberately rather than becoming divergent copies of every example.

## Recommended implementation order

Examples should be added only after the public operations they teach are
accepted:

1. Add basics, validity, grids, numerics, and eager suitability after Phase B.
2. Add expression inspection and file-backed writing after Phases C and D.
3. Add focal, region, and terrain examples after Phase E.
4. Add zonal and global summaries after Phase F.
5. Add hazard clearance after Phase G.
6. Add temporal illumination after Phase H.
7. Finish with the end-to-end explorer, QGIS inspection, and review-before-run
   examples during Phase I.

The final public portfolio need not contain every proposed file. Closely
related ideas may be combined if the result stays approachable. At minimum,
the implemented set should cover the Phase I requirements for
terrain-lighting screening, weighted scoring, focal cleanup, zonal candidate
summaries, hazard clearance, large file-backed expressions, temporal threshold
summaries, QGIS mask inspection, and human review before execution.
