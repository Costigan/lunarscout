# Lunarscout Python Surface Design and Implementation Plan

Date: 2026-06-18

## Purpose

`lunarscout` will be the pip-installable Python library for lunar-oriented map algebra and early mission design calculations. It should help scientists and mission designers answer questions such as:

- Where are candidate sites that satisfy terrain, lighting, communication, and science constraints?
- How do candidate sites change as thresholds or time windows change?
- Which terrain and illumination products should be generated for early trade studies?
- Which calculations are lightweight Python/NumPy map algebra and which should use native `.NET`/C# acceleration through `pythonnet`?
- Which calculations should use Python, C#/.NET multithreaded code, ILGPU-synthesized CUDA kernels, Dask, or some combination of those implementation strategies?

The purpose of using `pythonnet` is not that every calculation should be written in C#. It is that Python is not always a good runtime for multi-core, heavily multithreaded compute, especially for long-running native-style terrain and illumination workloads. `pythonnet` lets a Python notebook/API surface call into C# implementations that can use .NET's threading/runtime model and existing `moonlib` validation. Some of that C# code also uses ILGPU, which can synthesize CUDA kernels. Dask can also distribute array computations and can target parallel/GPU-style execution, so implementation assignment should be a deliberate design choice rather than a default assumption that "native" always means C#.

This is not a fresh implementation. Lunar Analyst already has substantial pieces:

- `raster.calculate`: restricted single-expression map algebra.  The restrictions are intended to improve the ability of LLM's to specify map algebra calculations.
- `raster.transform`: restricted Python-like multi-step raster transform.  This is a second approach to the same thing.  (I'm not sure which came first.)
- notebook helper functions under `backend.notebook.notebook_helper`.
- lazy raster expression objects under `backend.jobs.raster_transform`.
- native C# temporal lighting and PSR-style map operations exposed through `MoonlibBridge` and `LightmapStreamingClient`.
- assistant/MCP tool surfaces and tests for map algebra workflows.

The work now is to label and consolidate the human-programmer Python interface as `lunarscout`, fill the gaps, make it pip-installable, create notebooks that demonstrate it, and then re-evaluate agent support once the human API is stable.

## Design First

The first milestone is not packaging mechanics. It is designing the Python surface a human programmer should want to use in notebooks and ordinary Python code.

The current concrete API draft is `docs/LUNARSCOUT_API_SKETCH.md`. The first
three notebook-first executable design specifications are under
`examples/`. These programs validate the implemented local API and remain
review artifacts for later managed surfaces; they do not
yet imply that the proposed package surface is implemented.

Where this broad plan still shows the earlier public lazy-`Raster` design,
`docs/LUNARSCOUT_API_SKETCH.md` and the tracked v0.1 first slice below are the
current design authority. The older sections will be revised as later slices
are planned; they are not implementation instructions for v0.1.

The design must answer:

- What does a scientist import?
- How do they open or define a scenario?
- How do they bind DEMs, rasters, horizons, and time ranges?
- How do they write readable map algebra?
- How do they save and inspect outputs?
- Which functions are local Python/NumPy and which are native-backed?
- How does the same calculation run locally, in a managed Lunar Analyst job, or through a future agent-authored workflow?

Implementation should follow those answers, not expose backend internals directly.

## Target User Experience

The desired import should be simple:

```python
import lunarscout as ls
```

The first-class workflow should look like this:

```python
import lunarscout as ls

scenario = ls.open_scenario("/data/mons_mouton")

dem = scenario.dem()
slope = ls.slope(dem)
candidate = ls.where((slope <= 8) & (dem > 0), 1, ls.nodata())

output = scenario.write_raster(
    candidate,
    "analysis/candidate_flat_sites.tif",
    publish=True,
    title="Candidate flat sites",
)
```

A temporal/native-backed workflow should look like this:

```python
import lunarscout as ls

scenario = ls.open_scenario("/data/mons_mouton")
times = ls.times(
    "2027-01-01T00:00:00Z",
    "2027-01-08T00:00:00Z",
    step_hours=2,
)

illumination = scenario.sun_fraction(times=times)
mean_illumination = ls.avg(illumination)
candidate = ls.where(mean_illumination >= 0.65, mean_illumination, ls.nodata())

scenario.write_raster(candidate, "analysis/mean_illumination_candidates.tif")
```

A remote managed-job workflow should feel like the same conceptual model:

```python
job = scenario.submit(
    candidate,
    output="analysis/mean_illumination_candidates.tif",
    publish=True,
)
job.wait()
```

The package should make local notebook use easy first, then lower the same expression model into governed Lunar Analyst job contracts where appropriate.

## Current Implementation Inventory

### Existing `raster.calculate`

Status: implemented and useful for simple one-step products.

What exists:

- Implemented through `ToolImplementations.raster_calculate`.
- Public tool name: `raster.calculate`.
- Generated API route: `/api/v1/jobs/raster-calculate`.
- Inputs can bind by scenario-relative path or `product_id`.
- Static rasters align to the scenario primary DEM grid.
- Temporal inputs can bind through legacy `signal` values:
  - `lighting_raster`
  - `earth_above_horizon`
  - `sun_above_horizon`
- Outputs are written as scenario-managed GeoTIFFs, registered as products/files, and optionally published as map layers.
- Selection-mask publishing supports transparent background behavior.

Supported functions and operators:

- arithmetic and comparisons
- elementwise boolean operators: `&`, `|`, `~`
- `where(...)`
- `nodata()`, `nan()`, `null()`
- `slope(...)`, `aspect(...)`, `hillshade(...)`
- temporal reducers: `min(...)`, `max(...)`, `avg(...)`, `std(...)`
- region helpers:
  - `label_regions(...)`
  - `region_sizes(...)`
  - `filter_regions_by_size(...)`
  - `find_borders(...)`

Design decision for `lunarscout`:

- Keep `raster.calculate` as a compatibility and assistant convenience path.
- Do not make its custom expression DSL the primary human Python API.
- Reuse its implementation where practical, but orient new examples around `lunarscout` expression objects and `raster.transform` lowering.

### Existing `raster.transform`

Status: implemented and closest to the desired human authoring model.

What exists:

- Implemented through `ToolImplementations.raster_transform`.
- Public tool name: `raster.transform`.
- Generated API route: `/api/v1/jobs/raster-transform`.
- Accepts restricted Python-like scripts.
- Supports single-expression and multi-statement scripts.
- Multi-statement scripts must assign final output to `result`.
- Supports static and temporal inputs.
- Plans execution strategy and estimates working set.
- Supports full-extent static, full-extent temporal, and tiled temporal execution paths.
- Outputs include lineage metadata, planner summary, used variables/functions/operators, product ID, file ID, and progress events.

Supported script features:

- intermediate assignments
- arithmetic, comparisons, broadcasting, and elementwise boolean operations
- `where(...)`
- `np.where(...)` through a sealed facade
- tolerated `import numpy as np` compatibility import
- `nodata()`, `nan()`, `null()`
- `slope(...)`, `aspect(...)`, `hillshade(...)`
- temporal reducers: `min(...)`, `max(...)`, `avg(...)`, `std(...)`

Temporal model:

- Preferred model uses a reserved `times` binding:
  - `inputs.times = {"kind": "times", "start_utc": ..., "stop_utc": ..., "step_hours": ...}`
  - temporal rasters bind through `temporal_source` and `times`.
- All internal temporal calculations use UTC. Time zone conversion is allowed only at API boundaries for parsing user input, formatting outputs, notebook display, reports, and other I/O.
- Supported temporal sources:
  - `sun_fraction`
  - `sun_over_horizon_deg`
  - `earth_over_horizon_deg`
  - `station_over_horizon_deg`
- Legacy top-level time fields and `signal` bindings still exist.

Design decision for `lunarscout`:

- Treat `raster.transform` as the canonical remote/governed execution target.
- Build the public Python API so expressions can lower into `raster.transform` requests.
- Avoid inventing a second remote execution contract.

### Existing Notebook and Local Python Helpers

Status: useful internal/local surface, not yet a polished public package.

Existing useful APIs:

- `scenario_dem()`
- `raster_file(path)`
- `slope_raster(src)`
- `aspect_raster(src)`
- `hillshade_raster(src, ...)`
- lazy `Raster` arithmetic/comparison/boolean operations
- `np.where(...)` support through `Raster.__array_function__`
- `write_output_raster(...)`
- `label_regions(...)`
- `region_sizes(...)`
- `filter_regions_by_size(...)`
- `find_borders(...)`
- `run_lightmap_streaming_raster_job(...)`
- `run_lightmap_signal_streaming_raster_job(...)`
- `run_lightmap_native_reduction_raster_job(...)`
- `bootstrap_native_and_register_gdal()`
- `create_moonlib_bridge()`

Design decision for `lunarscout`:

- Extract or wrap this surface behind product-oriented names.
- Keep backend-internal imports working for compatibility, but stop documenting them as the preferred user API.
- Move toward `lunarscout.Scenario`, `lunarscout.Raster`, `lunarscout.TimeRange`, and top-level operation functions.

### Existing C# Native Integration

Status: real and useful as a selective heavy-compute backend.

The Python surface is the general map algebra interface. The C# backend does not need to become a general map algebra engine for arbitrary expressions. Its purpose is to handle heavy lifting for operations where C#/.NET is the right tool: multi-core, multithreaded, long-running calculations; validated lunar geometry already present in `moonlib`; and ILGPU-generated CUDA kernels where that is the best fit. At the same time, Dask remains a credible implementation option for chunked/distributed array work, including work that may map to GPU execution. `lunarscout` should evaluate these choices operation by operation.

What exists:

- Production Python native access goes through `MoonlibBridge`.
- `LightmapStreamingClient` bridges Python to C# temporal lighting signal streaming.
- `raster.calculate` and `raster.transform` consume native temporal signals for sun/earth/horizon analyses.
- `MoonlibBridge.GeneratePermanentShadowMap(...)` exposes native C# PSR map operations.
- Existing C# code includes ILGPU-oriented paths for GPU-style acceleration.
- Native PSR examples exist in notebooks.
- Native typed jobs exist for PSR-style and lighting workflows.

Current boundaries and gaps:

- Native-backed operations are not yet presented consistently through the `lunarscout` Python vocabulary.
- Python terrain helpers such as `slope`, `aspect`, and `hillshade` are currently Python-side, not C# delegated.
- Native PSR generation is separate from the normal expression vocabulary.
- Native PSR validity uses GDAL's mask band: `255` is calculated, `0` is
  unknown because required horizon coverage was unavailable. Complete products
  use the virtual all-valid mask; partial products store an internal 1-bit mask.
- There is no operation registry that tells users which functions are Python-backed, native-backed, or planned.

Design decision for `lunarscout`:

- Expose native-backed functions intentionally and explicitly.
- Do not hide native/runtime requirements when they matter.
- Keep all native calls behind a small `lunarscout.native` boundary that uses `MoonlibBridge` and `pythonnet`.
- Avoid making ordinary import of `lunarscout` immediately initialize `pythonnet`.
- Treat C#/.NET, ILGPU/CUDA, Dask, and Python/NumPy/rasterio as implementation choices selected per operation based on correctness, performance, packaging burden, and user experience.
- Do not require C# mapops to become the general expression evaluator; `lunarscout` owns the general Python map algebra surface.

## Product Scope

`lunarscout` is for early lunar mission design and site screening. It is not a generic GIS package.

Primary workflows:

- terrain suitability maps
- slope/aspect/hillshade inspection
- threshold and constraint masks
- connected-region and candidate-area analysis
- illumination summaries over time windows
- Earth-visibility and communication screening
- PSR-style screening
- combining terrain, lighting, and communication constraints
- producing GeoTIFFs and notebook visualizations for trade studies

Out of scope for the first `lunarscout` package:

- unrestricted Python execution as a remote service
- full GIS desktop replacement
- full focal/zonal/global map algebra taxonomy
- arbitrary user-defined kernels
- a new independent scenario database model
- a new C# expression engine before the Python surface is designed

## Python Surface Design

### Package Shape

Initial package name:

```text
lunarscout
```

Target import:

```python
import lunarscout as ls
```

Proposed modules:

```text
lunarscout/
  __init__.py
  scenario.py
  raster.py
  ops.py
  temporal.py
  native.py
  io.py
  remote.py
  examples.py
  _compat/
```

Design intent:

- `lunarscout.__init__` exposes the common notebook-facing API.
- `lunarscout.raster` holds `Raster`, `RasterSource`, grid metadata, and expression graph primitives.
- `lunarscout.ops` holds map algebra functions.
- `lunarscout.temporal` holds `TimeRange`, temporal sources, and reducers.
- `lunarscout.native` owns `pythonnet`/`MoonlibBridge` bootstrap and native-backed functions.
- `lunarscout.remote` owns optional Lunar Analyst API/job submission.
- `_compat` wraps current backend implementations during migration.

### Core Objects

#### `Scenario`

Represents a scenario folder and optional Lunar Analyst backend identity.

Required behavior:

- open a scenario folder
- resolve primary DEM
- resolve scenario-relative raster paths safely
- optionally attach to a running Lunar Analyst backend
- write local GeoTIFF outputs
- optionally register/publish outputs through backend APIs
- submit managed jobs when backend connection is available

Sketch:

```python
scenario = ls.open_scenario("/data/mons_mouton")
dem = scenario.dem()
input_raster = scenario.raster("inputs/slope.tif")
scenario.write_raster(result, "analysis/output.tif")
scenario.submit(result, output="analysis/output.tif")
```

Design decision:

- The local filesystem scenario is the base abstraction.
- Backend/API attachment is optional, not required for import or simple local notebook use.

#### `Raster`

Represents either a concrete raster source or a lazy expression.

Required behavior:

- arithmetic and comparison operator overloads
- elementwise boolean composition
- NumPy-friendly display/materialization
- retains source identity and grid metadata when known
- can be evaluated locally
- can be lowered to a `raster.transform` request when possible

Sketch:

```python
dem = scenario.dem()
slope = ls.slope(dem)
candidate = (slope <= 8) & (dem > 0)
```

Design decision:

- `Raster` should be a lazy expression by default.
- `Raster.read()` or `Raster.materialize()` should be explicit because mission-scale rasters may be large.
- Expressions should remain serializable enough to lower into job requests.

#### `TimeRange`

Represents a temporal domain for lighting/visibility calculations.

Sketch:

```python
times = ls.times("2027-01-01T00:00:00Z", "2027-01-08T00:00:00Z", step_hours=2)
```

Design decision:

- `lunarscout` always stores and computes temporal domains in UTC.
- Naive datetimes and timezone-free strings default to UTC; callers may supply
  `source_timezone` to interpret them in another IANA timezone.
- Time zone translation should be provided for I/O and reporting only, for
  example formatting a UTC result table in a requested local time zone.
- Internally lower to the existing inclusive `times` binding model used by
  `raster.transform`.

#### `Job`

Represents a remote managed Lunar Analyst job.

Sketch:

```python
job = scenario.submit(result, output="analysis/candidates.tif")
job.wait()
job.result()
```

Design decision:

- Keep remote job support separate from local expression authoring.
- Local notebooks should still work without a running FastAPI service.

### Top-Level Operation Names

Initial public functions:

- `ls.where(condition, x, y)`
- `ls.nodata()`
- `ls.slope(raster)`
- `ls.aspect(raster)`
- `ls.hillshade(raster, azimuth=315, altitude=45)`
- `ls.label_regions(mask, cleanup="none", iterations=0)`
- `ls.region_sizes(mask, cleanup="none", iterations=0)`
- `ls.filter_regions_by_size(mask, threshold, comparator=">=", cleanup="none", iterations=0)`
- `ls.find_borders(mask)`
- `ls.avg(temporal_raster)`
- `ls.min(temporal_raster)`
- `ls.max(temporal_raster)`
- `ls.std(temporal_raster)`
- `ls.times(start, stop, step_hours)`

Native-backed or native-source functions:

- `scenario.sun_fraction(times=...)`
- `scenario.sun_over_horizon_deg(times=...)`
- `scenario.earth_over_horizon_deg(times=...)`
- `scenario.station_over_horizon_deg(station_name, times=...)`
- `scenario.psr(...)`

Design decision:

- Use obvious scientific names rather than backend signal names.
- Keep native-backed operations discoverable from `Scenario`, because they need scenario DEM, horizon directory, SPICE/native runtime, and time metadata.

### Operation Registry

`lunarscout` should include an operation registry that records implementation availability.

Example fields:

- public function name
- category: terrain, region, temporal, visibility, native
- local implementation: Python/NumPy, rasterio, scipy, none
- remote implementation: `raster.transform`, `raster.calculate`, native job, none
- native implementation: `MoonlibBridge`, `LightmapStreamingClient`, none
- parallel implementation options: Python/NumPy, Dask, C#/.NET threads, ILGPU/CUDA, none
- implementation rationale: why this operation is assigned to that implementation
- supported dimensions: 2D, `[time, y, x]`
- notes and constraints

Example:

```python
ls.describe_operation("sun_fraction")
```

Design decision:

- This registry should be user-visible.
- It should also drive docs and agent guidance later.

### Local vs Remote Execution

`lunarscout` should support two execution modes:

1. Local notebook execution
   - operates directly on files/arrays
   - uses Python/NumPy/rasterio
   - may use native `pythonnet` bridge for selected operations
   - good for experimentation

2. Managed Lunar Analyst execution
   - lowers expressions to `raster.transform` or native typed jobs
   - records lineage and progress
   - registers outputs in scenario state
   - supports cancellation
   - good for reproducible app/assistant workflows

Design decision:

- The same expression should be reusable in both modes when possible.
- If an expression cannot lower to remote execution, `lunarscout` should say why with a clear diagnostic.

## Packaging Design

### Project Layout

Preferred first step inside this repo:

```text
packages/lunarscout/
  pyproject.toml
  README.md
  src/lunarscout/
  tests/
  examples/
```

Alternative:

```text
lunarscout/
  pyproject.toml
  src/lunarscout/
```

Recommendation:

- Use `packages/lunarscout/` to make clear this is a package inside the larger Lunar Analyst monorepo.
- Keep `moonlayers_pkg/` separate.
- Do not bury `lunarscout` under `backend/`.

### Dependencies

Core dependencies:

- `numpy`
- `rasterio`
- `pydantic` or lightweight dataclasses for typed specs

Optional dependencies:

- `pythonnet` for native `.NET`/C# calculations
- local `moonlib` runtime payloads or package extras
- `httpx` for remote Lunar Analyst API calls
- `matplotlib` for examples
- `jupyter`/`marimo` for notebook examples
- `moonlayers` for interactive map display examples

Suggested extras:

```toml
[project.optional-dependencies]
native = ["pythonnet"]
remote = ["httpx"]
notebooks = ["jupyter", "marimo", "matplotlib"]
moonlayers = ["moonlayers"]
dev = ["pytest", "ruff", "mypy"]
```

Design decision:

- `pip install lunarscout` should not require `pythonnet` unless native functionality is requested.
- `pip install "lunarscout[native]"` should install the Python dependency for native interop, but the .NET runtime and native payload discovery still need explicit documentation.

### Native Runtime Strategy

Native-backed calculations need a clear packaging story.

The native runtime exists because some operations need reliable multi-core/multithreaded execution or existing validated C# lunar geometry code, not because the user-facing API should stop feeling like Python. ILGPU-backed C# kernels and Dask-backed Python array execution should both be considered when selecting implementations for expensive functions. The package should make those decisions explicit while keeping the notebook API stable.

Rules:

- Importing `lunarscout` must not initialize `pythonnet`.
- Native bootstrap must be lazy.
- Errors must explain missing `.NET`, missing `pythonnet`, missing `moonlib`, or missing CSPICE/GDAL payloads separately.
- Production native access should go through `MoonlibBridge`.
- Direct use of internal `moonlib` types should remain private/test-only.

Open packaging decisions:

- Whether `lunarscout[native]` depends on a separate wheel that carries `moonlib` native artifacts.
- Whether native artifacts remain repo-local and are discovered through environment variables for now.
- Whether to publish a pure-Python wheel first and document native acceleration as "available in Lunar Analyst checkout/runtime".

Recommended phased decision:

- Phase 1: pure-Python `lunarscout` package with optional native bridge that works inside this repo/runtime.
- Phase 2: define a separate distributable native payload strategy after the Python surface stabilizes.

### Versioning

Recommended initial version:

```text
0.1.0
```

Versioning rules:

- `0.x` can evolve quickly while notebooks are being designed.
- Public function names and notebook examples should be treated as sticky once documented.
- Backend-internal compatibility imports should not be part of semantic version guarantees.

## Notebook Example Plan

Create notebooks as first-class design artifacts. The notebooks should drive the surface design before the package is considered ready.

Recommended location:

```text
packages/lunarscout/examples/notebooks/
```

Also mirror selected examples into Lunar Analyst docs or notebook examples if useful.

### Notebook 1: Slope Threshold Site Screening

Goal:

- Show the simplest local map algebra workflow.

Demonstrates:

- opening a scenario
- loading DEM
- calculating slope
- thresholding slope
- writing a candidate mask
- optionally displaying in MoonLayers

Key API decisions tested:

- `open_scenario`
- `scenario.dem()`
- `slope`
- `where`
- `nodata`
- `scenario.write_raster()`

### Notebook 2: Terrain Plus Region Filtering

Goal:

- Turn a noisy terrain mask into candidate regions.

Demonstrates:

- connected component labeling
- filtering by minimum area/pixel count
- border extraction
- preserving transparent background

Key API decisions tested:

- `label_regions`
- `region_sizes`
- `filter_regions_by_size`
- `find_borders`
- output metadata for masks

### Notebook 3: Illumination Time Window Screening

Goal:

- Show native-backed temporal signal use.

Demonstrates:

- defining a time range
- loading/generating/using horizon-derived lighting signals
- computing average/minimum illumination
- thresholding candidates

Key API decisions tested:

- `times`
- `scenario.sun_fraction`
- `avg`
- native/runtime error messages
- local vs managed execution choice

### Notebook 4: Earth Visibility and Communication Screening

Goal:

- Combine Earth visibility constraints with terrain constraints.

Demonstrates:

- `earth_over_horizon_deg`
- thresholding communication access
- combining boolean masks
- producing a trade-study layer

Key API decisions tested:

- temporal source naming
- boolean expression readability
- output lineage

### Notebook 5: PSR-Style Native Mapops

Goal:

- Demonstrate an explicitly native C#-backed calculation.

Demonstrates:

- native bootstrap
- required horizon inputs
- native PSR generation
- output comparison with simpler temporal masks

Key API decisions tested:

- `scenario.psr`
- native operation diagnostics
- managed job fallback

### Notebook 6: Local vs Managed Job Execution

Goal:

- Show that the same conceptual calculation can run locally or as a governed job.

Demonstrates:

- local materialization
- remote submission to Lunar Analyst backend
- job progress
- cancellation or timeout behavior
- output registration and map publishing

Key API decisions tested:

- expression lowering
- `scenario.submit`
- job object ergonomics
- compatibility with existing `raster.transform`

### Notebook 7: Early Mission Design Trade Study

Goal:

- Combine terrain, illumination, Earth visibility, and region filtering into a complete early design workflow.

Demonstrates:

- scenario setup
- multiple constraints
- score map or candidate mask
- sensitivity to thresholds
- output products for human review

Key API decisions tested:

- the whole package as a coherent scientist-facing tool.

## Implementation Plan

### Phase 0: Preserve Current Status and Inventory

Goal:

- Keep the existing map algebra status visible while transitioning to the `lunarscout` plan.

Tasks:

- [x] Record current `raster.calculate`, `raster.transform`, notebook helper, assistant, and native integration status in this plan.
- [ ] Create a source inventory mapping current functions to proposed `lunarscout` public names.
- [ ] Identify which current tests can be reused directly for `lunarscout`.

Acceptance criteria:

- [ ] Every proposed public function has a current implementation, wrapper target, or explicit gap label.

### Phase 1: Design the Public API Before Moving Code

Goal:

- Freeze a small, coherent first public surface.

Tasks:

- [x] Draft `lunarscout` API reference with function signatures and examples.
- [x] Decide that local raster values use NumPy arrays plus `GeoReference`, not
  a public lazy `Raster` type.
- [x] Decide the initial public types: `GeoReference`, `Scenario`, `TimeRange`,
  `ComputationPlan`, and `Job`.
- [ ] Decide exact function names and aliases.
- [x] Separate eager local NumPy execution from explicit managed computation
  plans.
- [ ] Decide how errors are named and surfaced.
- [ ] Decide how native availability is reported.
- [ ] Decide which operations are in v0.1 and which are planned.

Acceptance criteria:

- [ ] The example notebooks can be sketched using only proposed public names.
- [ ] No backend-internal import is required in notebook examples.

Current design artifacts:

- [x] Draft `docs/LUNARSCOUT_API_SKETCH.md` for review.
- [x] Draft the first three target API examples without backend-internal
  imports.
- [ ] Resolve the review questions in the API sketch and freeze the v0.1
  surface.

### Tracked v0.1 First Slice: GeoTIFF I/O, Georeferencing, and UTC

Status: implemented and verified, except for direct installation into the
sandbox-mounted read-only `.venv/site-packages`; editable build and install
were verified using a temporary prefix.

Goal:

- Establish the smallest installable `lunarscout` package around NumPy,
  single-band GeoTIFF I/O, reusable georeferencing, and UTC construction.

#### Files Allowed to Change

Package and tests:

- `packages/lunarscout/pyproject.toml`
- `packages/lunarscout/README.md`
- `packages/lunarscout/src/lunarscout/__init__.py`
- `packages/lunarscout/src/lunarscout/errors.py`
- `packages/lunarscout/src/lunarscout/georeference.py`
- `packages/lunarscout/src/lunarscout/geotiff.py`
- `packages/lunarscout/src/lunarscout/temporal.py`
- `packages/lunarscout/tests/conftest.py`
- `packages/lunarscout/tests/test_georeference.py`
- `packages/lunarscout/tests/test_geotiff_io.py`
- `packages/lunarscout/tests/test_temporal.py`

Documentation and environment wiring, only if required to install or run this
slice:

- `docs/LUNARSCOUT_API_SKETCH.md`
- `docs/LUNARSCOUT_PYTHON_SURFACE_PLAN.md`

Changing backend API routes, scenario services, jobs, native C#, MoonLayers,
or the design examples requires a separately reviewed slice.

#### Locked API Decisions

- [x] Raster values are ordinary `numpy.ndarray` objects.
- [x] Raster scalar types are ordinary NumPy dtypes.
- [x] `read_geotiff(filename, band=1)` reads exactly one GDAL band using
  one-based band numbering.
- [x] The returned array retains the selected band's TIFF datatype.
- [x] `GeoReference.nodata` is the selected band's actual nodata value;
  missing nodata is `None`.
- [x] Complex raster datatypes are deferred.
- [x] `read_geotiff()` returns `(array, None)` when projection or affine
  georeferencing is unavailable instead of constructing a partial
  `GeoReference`.
- [x] Invalid band zero, an out-of-range band, an unreadable file, and an
  unsupported datatype have distinct error codes.
- [x] `GeoReference` contains projected CRS WKT and PROJ.4 strings, all six
  affine coefficients, width, height, signed X/Y pixel-size coefficients, and
  nodata.
- [x] Integer pixel coordinates refer to pixel centers by default.
- [x] Coordinate helpers support scalar coordinates and NumPy arrays while
  preserving the input broadcast shape.
- [x] Coordinate order is `(column, row)`, `(easting, northing)`, and
  `(longitude, latitude)`.
- [x] Geographic conversion derives the lunar geographic spatial reference
  from the projected OSR spatial reference and uses traditional GIS axis
  order.
- [x] `write_geotiff()` writes one band using exactly `array.dtype`.
- [x] No auxiliary metadata is copied in v0.1.
- [x] GeoTIFF defaults are 128 x 128 tiles and datatype-sensitive compression.
- [x] Integer compression defaults to `DEFLATE` with predictor 2; floating
  compression defaults to `DEFLATE` with predictor 3.
- [x] BigTIFF defaults to GDAL `IF_SAFER`.
- [x] Writes use a temporary file in the destination directory followed by an
  atomic replacement on supported Linux filesystems.
- [x] `overwrite=False` rejects existing output; `overwrite=True` atomically
  replaces it.
- [x] `utc_datetime(...)` returns a timezone-aware standard Python `datetime`
  with UTC `tzinfo` without requiring the caller to specify `tzinfo`.
- [x] Lunarscout interprets naive datetimes and timezone-free ISO strings as
  UTC; aware non-UTC datetimes are converted to UTC.
- [x] Temporal arrays will use UTC `numpy.datetime64`; bridge conversion to
  .NET `DateTimeKind.Utc` remains explicit and is outside this slice.

#### API to Implement

- [x] Export `GeoReference`, `read_geotiff`, `write_geotiff`, and
  `utc_datetime` from `lunarscout`.
- [x] Implement immutable `GeoReference` validation.
- [x] Implement scalar and vectorized `pixel_to_projected()`.
- [x] Implement scalar and vectorized `projected_to_pixel()` using the inverse
  of the full affine transform.
- [x] Implement scalar and vectorized `projected_to_lonlat()`.
- [x] Implement scalar and vectorized `lonlat_to_projected()`.
- [x] Implement composed `pixel_to_lonlat()` and `lonlat_to_pixel()` helpers.
- [x] Implement `contains_pixel()` for scalar and array inputs.
- [x] Implement rotation-aware `projected_bounds()` from all four raster
  corners.
- [x] Lazily create and cache OSR coordinate transformations without mutating
  public georeferencing state.
- [x] Implement `read_geotiff()` with deterministic dataset cleanup.
- [x] Implement `write_geotiff()` with shape, dtype, nodata, path, creation
  option, and overwrite validation.
- [x] Validate that a declared nodata value is representable by `array.dtype`
  before writing.
- [x] Implement temporary-file cleanup for successful and failed writes.
- [x] Define stable `LunarscoutError`, GeoTIFF, georeferencing, band, datatype,
  transform, and output-exists exceptions with `code` and `details`.
- [x] Implement `utc_datetime()` with the standard `datetime` constructor's
  date/time fields except `tzinfo`.
- [x] Keep import of `lunarscout` free of `pythonnet`, native bootstrap,
  FastAPI startup, or scenario database access.

#### Dependency Decisions and Tasks

- [x] Use the repo-managed `.venv/bin/python` for all implementation and test
  commands.
- [x] Confirm the supported repo GDAL/OSGeo binding imports successfully in the
  package test environment.
- [x] Decide and document whether GDAL is an external system prerequisite or
  package dependency; do not add an unvalidated PyPI `GDAL` pin merely to make
  packaging metadata self-contained.
- [x] Declare NumPy as a package dependency using the repository-compatible
  version range.
- [x] Verify editable installation does not alter or initialize the moonlib
  runtime.

#### Required Tests

- [x] Editable install smoke test: `import lunarscout as ls` (temporary prefix
  because sandbox `.venv/site-packages` is read-only).
- [x] Import test proving no `pythonnet`/moonlib bootstrap occurs.
- [x] Read each supported non-complex GDAL datatype fixture and assert the
  exact returned NumPy dtype.
- [x] Select a non-default band from a multiband test fixture.
- [x] Assert band zero and out-of-range bands produce distinct stable errors.
- [x] Assert declared integer, floating, and absent nodata values round-trip
  without substitution.
- [x] Assert a file lacking projection or affine georeferencing returns
  `(array, None)`.
- [x] Assert WKT, PROJ.4, affine coefficients, dimensions, and signed pixel
  sizes are populated from GDAL.
- [x] Test scalar pixel-center and pixel-corner conversions.
- [x] Test scalar projected-to-pixel round trips without implicit rounding.
- [x] Test vectorized and broadcast coordinate conversions.
- [x] Test projected/lunar-lon-lat round trips using ESRI:103878.
- [x] Test rotated-affine projected bounds and inverse transforms.
- [x] Write/read round trips for supported integer and floating dtypes.
- [x] Inspect output block size, compression, predictor, BigTIFF compatibility,
  projection, affine transform, and nodata.
- [x] Assert array/georeference shape mismatch is rejected.
- [x] Assert an unrepresentable nodata value is rejected before writing.
- [x] Assert `overwrite=False` preserves an existing destination.
- [x] Assert `overwrite=True` replaces an existing destination only after a
  complete successful write.
- [x] Inject a write failure and assert the destination is not partial and the
  temporary file is removed.
- [x] Test `utc_datetime()` defaults, optional time components, microseconds,
  and `fold`.

#### Acceptance Criteria

- [ ] `.venv/bin/python -m pip install -e packages/lunarscout` succeeds in the
  supported repository environment. The editable build/install succeeds with
  a temporary prefix, but the sandbox mounts `.venv/site-packages` read-only.
- [x] `import lunarscout as ls` succeeds without importing or initializing
  `pythonnet`.
- [x] `array, georef = ls.read_geotiff(path)` preserves native dtype, actual
  nodata, and complete georeferencing.
- [x] `read_geotiff(path)` returns `(array, None)` for a readable but
  non-georeferenced TIFF.
- [x] Scalar and vectorized coordinate round trips meet documented numerical
  tolerances.
- [x] `write_geotiff()` produces a single-band tiled GeoTIFF with the exact
  array dtype and agreed creation defaults.
- [x] `ls.utc_datetime(2027, 1, 1)` returns an aware UTC `datetime`.
- [x] All package tests pass independently of FastAPI and native service
  startup.

#### Explicitly Out of Scope

- No public lazy `Raster` or ndarray subclass.
- No complex raster datatype support.
- No multiband array convention or multiband output.
- No masking or automatic replacement of nodata with `NaN`.
- No pixel-value remapping or dtype-conversion helper.
- No `align()` or grid-equality implementation in this slice.
- No slope, aspect, hillshade, or region implementation in this slice.
- No scenario object or standard scenario paths in this slice.
- No product registration, publication, or `scenario.db` mutation.
- No temporal cube generation, streaming, or native bridge changes.
- No backend image-readout refactor yet; reuse of `GeoReference` there is a
  later vertical slice.
- No managed computation plan or remote job submission.

#### Risks and Rollback

- OSR axis-order defaults can swap longitude and latitude. Tests must enforce
  traditional GIS order and lunar CRS round trips.
- PROJ/GDAL runtime discovery differs between the normal OSGeo and moonlib
  process modes. This slice runs only in the normal OSGeo mode and must not
  initialize the native runtime.
- GeoTIFF nodata is exposed by GDAL as a numeric value and may not preserve the
  source metadata's textual spelling; the contract is value preservation, not
  string preservation.
- Atomic replacement is guaranteed only when the temporary and destination
  files share a filesystem; the implementation therefore creates them in the
  same directory.
- Rollback is additive: remove `packages/lunarscout/` and revert only the two
  Lunarscout design documents. No database, API, deployment, or native rollback
  is required.

### Phase 2: Create the Package Skeleton

Goal:

- Make `lunarscout` installable in editable mode.

Tasks:

- [x] Add `packages/lunarscout/pyproject.toml`.
- [x] Add `packages/lunarscout/src/lunarscout/__init__.py`.
- [x] Add the first-slice modules `errors.py`, `georeference.py`, `geotiff.py`,
  and `temporal.py`; add scenario/native/remote modules only in later slices.
- [x] Add package README with early mission design framing.
- [x] Add package tests and a package-local pytest configuration.
- [x] Add editable install guidance to the package README.

Acceptance criteria:

- [ ] `.venv/bin/python -m pip install -e packages/lunarscout` works.
  Editable build/install was verified with a temporary prefix because the
  sandbox mounts `.venv/site-packages` read-only.
- [x] `import lunarscout as ls` works without initializing `pythonnet`.
- [x] Package tests run independently of backend service startup.

### Phase 3: Wrap Existing Local Raster Functionality

#### Tracked second slice: GDAL-compatible terrain operations

Status: implemented and verified.

Files changed:

- `packages/lunarscout/src/lunarscout/terrain.py`
- `packages/lunarscout/src/lunarscout/georeference.py`
- `packages/lunarscout/src/lunarscout/errors.py`
- `packages/lunarscout/src/lunarscout/__init__.py`
- `packages/lunarscout/tests/test_terrain.py`
- `examples/01_terrain_products.py`
- Lunarscout API, package, and tracking documentation

Locked and completed:

- [x] Implement `slope()`, `aspect()`, and `hillshade()` as eager NumPy APIs.
- [x] Return `(array, new_georef)` uniformly.
- [x] Require caller-selected `output_nodata` and validate it against the
  GDAL result dtype.
- [x] Default slope to degrees and support `units="percent"`.
- [x] Default to GDAL `compute_edges=False` behavior.
- [x] Use GDAL Horn algorithms and preserve their neighborhood semantics.
- [x] Preserve GDAL `float32` slope/aspect and `uint8` hillshade output dtypes.
- [x] Remap GDAL's native output nodata without coercing the result dtype.
- [x] Preserve exact 64-bit source nodata when building the GDAL memory raster.
- [x] Add scalar plane, edge, nodata-neighborhood, units, dtype, argument, and
  georeference regression tests.
- [x] Rewrite the slope-threshold design example around NumPy and
  `GeoReference`.

Out of scope for this slice:

- Region operations
- Alignment and reprojection
- Scenario paths and product-generation functions

#### Tracked third slice: connected-region analysis

Status: implemented and verified.

Files changed:

- `packages/lunarscout/src/lunarscout/regions.py`
- `packages/lunarscout/src/lunarscout/errors.py`
- `packages/lunarscout/src/lunarscout/__init__.py`
- `packages/lunarscout/tests/test_regions.py`
- `examples/02_region_filtering.py`
- Lunarscout API, package, dependency, and tracking documentation

Locked and completed:

- [x] Keep eight-neighbor connectivity fixed for v0.1.
- [x] Preserve `>=` and `<=` region-size comparators.
- [x] Preserve cleanup modes `none`, `erosion`, and `opening`, defaulting to
  zero iterations.
- [x] Preserve cleanup-aware seed selection mapped back onto original region
  shapes.
- [x] Use `int32` for labels and region sizes.
- [x] Return `(array, georef_or_none)` for every region operation.
- [x] Resolve `nodata="auto"` from `GeoReference` when present and otherwise
  skip nodata processing.
- [x] Allow numeric nodata override and explicit `nodata=None` disablement.
- [x] Treat valid nonzero numeric pixels as true and Boolean pixels directly.
- [x] Exclude nodata pixels from connectivity and restore them in numeric
  outputs.
- [x] Return masked Boolean arrays only when nodata processing is active.
- [x] Keep SciPy imports lazy while declaring SciPy as a package dependency.
- [x] Add connectivity, comparator, cleanup, nodata, masking, dtype, shape, and
  validation regression tests.
- [x] Rewrite the region-filtering design example around NumPy and
  `GeoReference`.

Out of scope for this slice:

- Physical-area filtering
- Configurable four-neighbor connectivity
- Alignment and reprojection
- Scenario paths and product-generation functions

#### Tracked fourth slice: explicit grid comparison and alignment

Status: implemented and verified.

Files changed:

- `packages/lunarscout/src/lunarscout/alignment.py`
- `packages/lunarscout/src/lunarscout/errors.py`
- `packages/lunarscout/src/lunarscout/__init__.py`
- `packages/lunarscout/tests/test_alignment.py`
- Lunarscout API, package, and tracking documentation

Locked and completed:

- [x] Implement `same_grid()` and `require_same_grid()` without treating
  nodata as a grid property.
- [x] Compare width and height exactly and compare CRS semantically through
  GDAL/OSR.
- [x] Compare affine coefficients exactly by default; permit only an explicit
  non-negative absolute tolerance.
- [x] Raise structured `GridMismatchError` diagnostics identifying differing
  grid fields.
- [x] Implement array-only `align(source, source_georef, to=...)` against the
  exact destination CRS, affine transform, width, and height.
- [x] Preserve source dtype by default and perform dtype conversion only when
  `output_dtype` is explicit.
- [x] Default output nodata to source nodata; support explicit numeric
  replacement and explicit `None` disablement.
- [x] Reject nodata values not representable by the output dtype.
- [x] Expose every resampling algorithm available through the supported GDAL
  runtime using stable canonical names.
- [x] Return `(array, new_georef)` with the actual destination nodata.
- [x] Add grid, tolerance, diagnostics, resampling, shape, dtype, and nodata
  regression tests.

Out of scope for this slice:

- Filename convenience inputs
- Automatic alignment in NumPy, terrain, or region operations
- Alignment lineage persistence
- Scenario paths, state, registration, and managed jobs
- Complex and multiband rasters
- Native terrain implementations

#### Tracked fifth slice: filesystem-only scenario paths

Status: implemented and verified.

Files changed:

- `packages/lunarscout/src/lunarscout/scenario.py`
- `packages/lunarscout/src/lunarscout/errors.py`
- `packages/lunarscout/src/lunarscout/__init__.py`
- `packages/lunarscout/tests/test_scenario.py`
- `examples/01_terrain_products.py`
- `examples/02_region_filtering.py`
- Lunarscout API, package, and tracking documentation

Locked and completed:

- [x] Implement `open_scenario()` for existing scenario directories.
- [x] Keep `Scenario` filesystem-only and free of FastAPI, database, and
  native-runtime imports.
- [x] Use the active canonical primary DEM path `dem.tif`.
- [x] Use the canonical horizon directory `horizons`.
- [x] Resolve arbitrary input and output paths relative to the scenario root.
- [x] Reject absolute paths, parent traversal, and existing symlink escapes.
- [x] Return normalized paths without creating directories or files.
- [x] Reject a non-`None` state owner explicitly until the managed-state slice
  is implemented.
- [x] Add root, convention, normalization, traversal, symlink, output, and
  state-boundary regression tests.
- [x] Update local examples to use scenario path conventions.

Out of scope for this slice:

- Reading or mutating `scenario.db`
- Product registration and layer publication
- `LocalScenarioState` and inter-process writer coordination
- `RemoteScenarioState`, job submission, progress, and cancellation
- Scenario completeness validation or automatic directory creation
- Native product generation

#### Tracked sixth slice: native diagnostics and lazy boundary

Status: implemented and verified without initializing CLR in the test process.

Files changed:

- `packages/lunarscout/src/lunarscout/native.py`
- `packages/lunarscout/src/lunarscout/errors.py`
- `packages/lunarscout/src/lunarscout/__init__.py`
- `packages/lunarscout/pyproject.toml`
- `packages/lunarscout/tests/test_native.py`
- Lunarscout API, package, and tracking documentation

Locked and completed:

- [x] Expose `ls.native` without importing `pythonnet`, CLR, moonlib, or the
  backend bootstrap module during ordinary `import lunarscout`.
- [x] Implement `ls.native.status()` with separate Python.NET, .NET runtime,
  moonlib, CSPICE, and GDAL component reports.
- [x] Implement non-initializing `ls.native.is_available()`.
- [x] Implement explicit `ls.native.initialize(force=False, verify=True)` by
  delegating to the existing authoritative Lunar Analyst bootstrap.
- [x] Keep bridge construction private and restrict it to `MoonlibBridge`.
- [x] Add stable native exception classes and component-specific diagnostics.
- [x] Declare Python.NET only in the optional `native` package extra.
- [x] Test import isolation, discovery, explicit initialization, error
  translation, and the MoonlibBridge boundary using fakes.
- [x] Verify the real checkout can discover all native components without
  loading Python.NET.

Out of scope for this slice:

- Changes to `backend.worker.native_bootstrap`
- Actual CLR initialization during tests
- Direct use of moonlib types other than `MoonlibBridge`
- Temporal lightmap streaming and in-memory temporal cubes
- PSR or other native product generation
- Worker protocol, progress, cancellation, and FastAPI changes

#### Tracked seventh slice: temporal domain and named in-memory cube

Status: implemented and verified without native runtime initialization.

Files changed:

- `packages/lunarscout/src/lunarscout/temporal.py`
- `packages/lunarscout/src/lunarscout/errors.py`
- `packages/lunarscout/src/lunarscout/__init__.py`
- `packages/lunarscout/tests/test_temporal.py`
- `examples/04_temporal_cube.py`
- Lunarscout API and tracking documentation

Locked and completed:

- [x] Implement immutable `TimeRange` and `times()` with inclusive aligned
  stop behavior matching the worker contract.
- [x] Default naive datetimes and timezone-free strings to UTC.
- [x] Support explicit IANA `source_timezone` conversion for naive inputs.
- [x] Store time coordinates as read-only UTC-convention
  `numpy.datetime64[us]` values.
- [x] Implement frozen `TemporalCube(values, times, georef)` without copying
  its potentially large values array.
- [x] Validate `(time, y, x)` shape, georeference dimensions, time count,
  `NaT`, and strict coordinate ordering.
- [x] Expose shape, dtype, dimension, time-count, and memory-size properties.
- [x] Implement `temporal_mean`, `temporal_min`, `temporal_max`, and
  `temporal_std` as eager time-axis reducers.
- [x] Use the established `nodata="auto"`, numeric override, and explicit
  `None` conventions.
- [x] Update the illumination design example to consume `TemporalCube` and
  normal NumPy arrays.

Out of scope for this slice:

- Native population of a `TemporalCube`
- Streamed, chunked, memory-mapped, Dask, or file-backed temporal storage
- Allocation preflight for native generation
- Remote temporal lowering and managed jobs
- Temporal interpolation, resampling, and timezone-formatted reporting

#### Tracked eighth slice: file-backed timestamped GeoTIFF series

Status: core local storage implementation and unit verification complete.

Design authority:

- `docs/LUNARSCOUT_API_SKETCH.md`, section "File-Backed Timestamped GeoTIFF
  Series"

Locked design decisions:

- [x] Keep file-backed temporal data distinct from in-memory `TemporalCube` so
  accessing `.values` never unexpectedly materializes a multi-gigabyte cube.
- [x] Name the file-backed public type `TemporalGeoTiffSeries` and construct it
  with `open_temporal_cube()`.
- [x] Store one single-band, tiled, compressed GeoTIFF per UTC time sample.
- [x] Use UTC, lexically sortable timestamp filenames with fixed microsecond
  precision and no colon characters.
- [x] Treat the series directory as one logical temporal product even when it
  contains thousands of timestamp TIFFs.
- [x] Make `manifest.json` authoritative for format version, ordered times,
  layer paths, dtype, nodata, common grid, signal metadata, and provenance.
- [x] Use standard JSON with tagged encoding for non-finite nodata, canonical
  NumPy dtype text, both CRS forms, and per-layer index/time/path records.
- [x] Require relative, root-contained backing paths and strict common-grid
  validation across all layers.
- [x] Generate an optional, rebuildable `series.vrt` with one band per time in
  manifest order and timestamp band descriptions for QGIS/GDAL inspection.
- [x] Keep Python layer indexes zero-based and translate internally to GDAL's
  one-based VRT bands.
- [x] Read backing TIFFs directly for normal Python layer/time access; use the
  VRT primarily for QGIS and general GDAL interoperability.
- [x] Support exact, nearest, before, and after time lookup modes with
  structured failures instead of implicit clamping.
- [x] Use GDAL's block cache rather than duplicating decoded-tile caching.
- [x] Permit a bounded LRU of open datasets and an optional byte-budgeted LRU
  of complete read-only layers for repeated interactive access.
- [x] Make all caching optional and irrelevant to correctness.
- [x] Stream file-backed temporal reductions without allocating the full cube
  or a full-cube nodata mask.
- [x] Write through a staging/completion protocol so incomplete series are
  never accepted as complete and replacement preserves the prior good series.
- [x] Bind the completion record to the final manifest version and SHA-256
  digest so stale completion state is detectable.
- [x] Use no new storage dependency in the initial implementation.
- [ ] Benchmark representative data volume, including approximately 3,800
  layers and 3.8 GB compressed. The synthetic file-count benchmark below does
  not satisfy this data-volume benchmark.

Target API:

```python
series = ls.open_temporal_cube(
    path,
    layer_cache_bytes=...,
    max_open_datasets=...,
)
array, georef = series.read_layer(index)
array, georef = series.read_time(time, method="exact")
index = series.layer_for_time(time, method="nearest")
time = series.time_for_layer(index)

series = ls.write_temporal_cube(
    path,
    cube,
    signal_name="sun_fraction",
    units="fraction",
    provenance={"source": "moonlib"},
)

with ls.TemporalGeoTiffSeriesWriter(
    path,
    georef=georef,
    dtype=np.float32,
    progress_callback=on_progress,
    cancellation_requested=is_cancelled,
) as writer:
    for time, layer in generated_layers:
        writer.write_layer(time, layer)
```

Implementation tasks:

- [x] Define and version the JSON manifest schema.
- [x] Implement safe series open/validation and structured format errors.
- [x] Implement layer/time lookup and direct backing-TIFF reads.
- [x] Implement configurable open-dataset and full-layer LRU caches.
- [x] Generate relative-path VRTs with timestamp descriptions and metadata.
- [x] Implement staged writing, completion markers, overwrite, and recovery.
- [x] Implement an incremental staged writer so large producers can emit one
  `(time, array)` layer at a time without first allocating a `TemporalCube`.
- [x] Enforce exact per-layer dtype/grid and strict increasing-time validation.
- [x] Add structured per-layer progress callbacks and cooperative cancellation
  with staging cleanup and preservation of prior completed output.
- [x] Refactor `write_temporal_cube()` as a convenience wrapper over the
  incremental writer.
- [x] Implement streaming mean, minimum, maximum, and standard-deviation
  reducers with nodata handling.
- [x] Add an explicit native-generation storage selection and allocation
  preflight; never silently switch an in-memory request to file-backed output.
- [x] Choose the earlier sample for an exact `nearest` tie and test it.
- [ ] Define managed registration/export behavior for a logical series and its
  VRT backing assets in a later state-owner slice.
- [x] Add unit tests using small fixtures for round trips, VRT access, lookup,
  caching, integrity failures, overwrite recovery, and streaming reductions.
- [x] Add and run a gated 3,800-layer synthetic file-count benchmark. The
  1 x 1 layer run completed in 4.064 seconds on the development filesystem.
- [ ] Add a representative 3.8 GB compressed throughput benchmark using
  science-scale raster dimensions and content.

Implemented files:

- `packages/lunarscout/src/lunarscout/temporal_store.py`
- `packages/lunarscout/src/lunarscout/temporal.py`
- `packages/lunarscout/src/lunarscout/errors.py`
- `packages/lunarscout/src/lunarscout/__init__.py`
- `packages/lunarscout/tests/test_temporal_store.py`

At completion of this eighth slice, native generation, managed
registration/export, and representative data-volume benchmarking remained
follow-up work. Native generation is addressed by the ninth slice below;
managed state ownership and representative-volume benchmarking remain open.

#### Tracked ninth slice: native temporal generation storage selection

Status: implemented and verified with mocked streaming; native CLR integration
was compiled and covered by focused existing streaming tests.

Completed:

- [x] Require explicit `storage="memory"` or
  `storage="geotiff_series"`; never switch automatically.
- [x] Estimate the exact output allocation before native initialization and
  reject in-memory requests above the configured limit.
- [x] Require file-backed output paths and preflight both scratch and output
  disk space using an uncompressed upper estimate.
- [x] Assemble patch-major native V2 chunks into either a NumPy cube or a
  temporary disk memmap, then write timestamp layers through
  `TemporalGeoTiffSeriesWriter`.
- [x] Expose `Scenario.sun_fraction()`, `sun_over_horizon_deg()`, and
  `earth_over_horizon_deg()` using standard scenario DEM/horizon paths and
  root-contained outputs.
- [x] Convert native `sun_fraction_u8` to public float32 fractions in `[0, 1]`;
  preserve float32 degree units for Sun/Earth center horizon margins.
- [x] Reject malformed, out-of-order, overlapping, or incomplete native tile
  streams rather than publishing implicit zero regions.
- [x] Propagate structured progress and cooperative cancellation through
  streaming, scratch assembly, and series writing.
- [x] Route Lunarscout native streaming through `MoonlibBridge` forwarding
  methods instead of constructing another production moonlib bridge type.
- [x] Verify memory/file output, preflight rejection, bridge construction,
  safe Scenario output, coverage failure, and cancellation without CLR.

Implemented files:

- `packages/lunarscout/src/lunarscout/native_temporal.py`
- `packages/lunarscout/src/lunarscout/native.py`
- `packages/lunarscout/src/lunarscout/scenario.py`
- `packages/lunarscout/src/lunarscout/errors.py`
- `backend/worker/lightmap_streaming.py`
- `native/new_horizon/moonlib/MoonlibBridge.cs`
- Python and C# native temporal/bridge tests

Risks and rollback:

- Patch-major file-backed generation needs temporary uncompressed disk space;
  preflight requires room for both scratch and an uncompressed output upper
  estimate. A future time-major native stream could remove this scratch step.
- The MoonlibBridge forwarding methods are additive. Rollback consists of
  removing those forwarding methods, the Lunarscout native temporal adapter,
  and Scenario methods; existing typed jobs and streaming implementation
  remain otherwise unchanged.

Still out of scope:

- FastAPI product registration, publication, or scenario database mutation
- Automatic storage selection
- Partial-grid native products or configurable missing-patch fill behavior
- Station-specific communication signals
- Representative 3,800-layer/3.8 GB native throughput measurement

Explicitly out of scope for the initial implementation:

- Automatic QGIS temporal-controller configuration
- Publishing every timestamp TIFF as an independent scenario layer
- Treating the derived VRT as the authority for time metadata
- Silently loading the complete series into memory
- Zarr, HDF5, Dask, or a new storage dependency
- Sharded multiband GeoTIFFs

Goal:

- Expose current local notebook-helper behavior as eager NumPy operations that
  receive and return `GeoReference` where geospatial metadata is relevant.

Tasks:

- [x] Implement or wrap eager NumPy slope, aspect, and hillshade operations.
- [x] Preserve GDAL-compatible nodata and edge behavior with regression tests.
- [x] Wrap region utilities.
- [x] Use NumPy directly for arithmetic, boolean operations, selection, and
  reducers instead of duplicating them in `lunarscout`.
- [x] Keep CRS/grid alignment behavior explicit.

Acceptance criteria:

- [x] Notebook/example 1 can run using `import lunarscout as ls` when its sample
  scenario path is available.
- [x] Notebook/example 2 can run using `import lunarscout as ls` when its sample
  scenario path is available.
- [ ] Existing local raster transform runtime tests have `lunarscout` equivalents.

### Phase 4: Native Integration Boundary

Goal:

- Make native-backed operations usable without making `pythonnet` a surprise import-time dependency.

Tasks:

- [x] Implement `lunarscout.native.is_available()`.
- [x] Implement lazy native bootstrap with clear diagnostics.
- [ ] Wrap `LightmapStreamingClient` behind scenario methods for temporal sources.
- [x] Wrap `MoonlibBridge.GeneratePermanentShadowMap` behind `scenario.psr(...)`.
- [ ] Document native prerequisites.
- [x] Add tests with fake bridge/client objects.

Acceptance criteria:

- [ ] `import lunarscout` works without `pythonnet`.
- [ ] Calling native functions without native support gives actionable errors.
- [ ] Notebooks 3, 4, and 5 can be sketched or run in a native-enabled environment.

### Phase 5: Lowering to Managed Lunar Analyst Jobs

Goal:

- Reuse the same user expression model for governed remote execution.

Tasks:

- [ ] Add expression serialization/lowering from `lunarscout.Raster` to a `raster.transform` request.
- [ ] Add `Scenario.attach(base_url=...)` or `ls.connect(...)`.
- [ ] Add `Scenario.submit(...)`.
- [ ] Map local sources to scenario-relative paths or product IDs.
- [ ] Lower temporal sources to `times` and `temporal_source` bindings.
- [ ] Surface planner/preflight errors before job submission when possible.

Acceptance criteria:

- [ ] Notebook 6 works against a running Lunar Analyst backend.
- [ ] The generated request uses the existing `raster.transform` contract.
- [ ] No new remote map algebra contract is introduced.

### Phase 6: Example Notebooks as Acceptance Tests

Goal:

- Use notebooks to validate the actual user experience.

Tasks:

- [ ] Create the seven notebooks listed above.
- [ ] Keep notebooks small and runnable against sample data.
- [ ] Add smoke tests that execute notebook code cells where feasible.
- [ ] Capture expected outputs or assertions for generated rasters.
- [ ] Ensure examples say which operations are native-backed.

Acceptance criteria:

- [ ] A new user can follow the notebooks without importing backend internals.
- [ ] At least the non-native notebooks run in a plain Python environment.
- [ ] Native notebooks clearly skip or degrade when native payloads are unavailable.

### Phase 7: Documentation and Packaging Polish

Goal:

- Make the package usable outside this codebase.

Tasks:

- [ ] Write user guide: "Lunarscout for Early Lunar Mission Design".
- [ ] Write API reference.
- [ ] Document installation modes:
  - pure Python
  - notebooks
  - native
  - remote Lunar Analyst
- [ ] Document scenario folder expectations.
- [ ] Document native runtime requirements and troubleshooting.
- [ ] Add examples index.
- [ ] Decide publishing path for internal/private vs public PyPI.

Acceptance criteria:

- [ ] Package README is enough for first local workflow.
- [ ] Native setup failure modes are documented.
- [ ] Notebook examples link to API docs.

### Phase 8: Re-Evaluate Agent Support

Goal:

- Update assistant and MCP behavior after the human API is stable.

Tasks:

- [ ] Re-review RAG guidance in `docs/rag_corpus/guidance_map_algebra_scripts.txt`.
- [ ] Decide whether agents should author `lunarscout` notebooks, `raster.transform` jobs, or both.
- [ ] Update tool descriptions to match the `lunarscout` vocabulary.
- [ ] Add evals for notebook authoring with `lunarscout`.
- [ ] Add evals for selecting local notebook vs managed job execution.
- [ ] Add repair guidance based on the new public API errors.

Acceptance criteria:

- [ ] Agent support is evaluated against the final human API, not the old backend-internal names.
- [ ] Agent-authored examples use `import lunarscout as ls` where notebook code is requested.
- [ ] Tool calls still use governed `raster.transform`/native job contracts where remote execution is requested.

## Design Decisions to Make Explicit

### Is `lunarscout` a Standalone Library or Lunar Analyst SDK?

Recommendation:

- It should be both, but layered.
- Core local map algebra should run without a backend.
- Remote job submission should be optional.
- Native acceleration should be optional.

### Does `lunarscout` Replace `moonlayers`?

No.

- `lunarscout` is for calculations and early mission design products.
- `moonlayers` is for notebook map visualization.
- Examples may use both:

```python
import lunarscout as ls
from moonlayers import MoonMap
```

### Does `lunarscout` Replace `raster.calculate`?

Not immediately.

- `raster.calculate` remains a simple governed one-expression tool.
- `lunarscout` should prefer the richer `raster.transform` model for human Python.
- Over time, `raster.calculate` may become a compatibility path or an implementation detail for simple lowered expressions.

### Does `lunarscout` Use C# for All Calculations?

No.

- Many operations should remain Python/NumPy/rasterio.
- `lunarscout` owns the general map algebra surface; C# is an implementation backend for selected functions, not the user-facing algebra model.
- Native C# should be used where it materially improves performance, where robust multi-core/multithreaded execution matters, or where validated lunar geometry already exists in `moonlib`.
- ILGPU-backed C# kernels are an option for CUDA-style acceleration where existing code or performance evidence supports them.
- Dask is also a candidate for chunked/distributed array work and may overlap with some GPU/parallel use cases.
- The operation registry should make implementation choice and rationale visible.

### Should `lunarscout` Allow Arbitrary Python?

Local notebooks can always run arbitrary Python because notebooks are Python. But managed/remote execution should remain restricted and governed.

Design rule:

- Local authoring may be ergonomic Python.
- Remote lowering must preserve bounded, serializable, auditable execution.
- Do not turn remote `raster.transform` into unrestricted Python.

## Technical Risks

- Native packaging may be harder than the Python surface design.
- `pythonnet` runtime initialization can be process-global and hard to undo.
- Shipping native `.NET`, GDAL, CSPICE, and `moonlib` payloads may require separate wheels or explicit environment setup.
- A lazy expression API may drift from what `raster.transform` can serialize.
- If examples rely on backend scenario state too early, the standalone package will not feel standalone.
- If examples ignore backend job execution, the package will not integrate well with Lunar Analyst.

## Verification Strategy

Test categories:

- pure Python unit tests for expression graph construction
- local raster tests with tiny GeoTIFF fixtures
- CRS/alignment tests
- region utility tests
- temporal source request-construction tests
- native tests with fake bridge/client
- optional real native smoke tests
- remote lowering tests that assert generated `raster.transform` payloads
- notebook smoke tests

Minimum v0.1 acceptance:

- [ ] `import lunarscout as ls` works.
- [ ] local slope threshold notebook runs.
- [ ] local region filtering notebook runs.
- [ ] temporal source API can build requests without native import at package import time.
- [ ] native functions fail clearly when native runtime is unavailable.
- [ ] managed-job lowering produces valid `raster.transform` payloads.

## Bottom Line

`lunarscout` should not be a rewrite. It should be the named, pip-installable, human-facing Python layer over map algebra and early lunar mission design capabilities that already exist in Lunar Analyst.

The first priority is designing a coherent API and validating it with notebooks. Implementation should then extract, wrap, and stabilize existing code behind that API. Only after that should agent support be re-evaluated, because agents should learn the same Python surface that human mission designers use.
