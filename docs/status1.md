# Map-Algebra Public Layer Implementation Status

Date: 2026-07-21

## Scope

This pass implemented the public/metadata/documentation layer for halo-aware
terrain expression execution and explicit cross-grid resampling.  The underlying
core (`_spatial.py`, `_planner.py`, `_windowed.py`, `_windows.py`) was
previously implemented and verified.  This work added public wrappers, registry
enrichment, focused tests, documentation, an example script, plan reconciliation,
and a changelog entry. Codex then performed the final semantic/integration
review and repaired the public safety contract, example, tests, and checkbox
claims described below.

## Files Changed

| File | Status | Description |
|------|--------|-------------|
| `src/lunarscout/map_algebra/__init__.py` | Modified | Public terrain wrappers (`slope`, `aspect`, `hillshade`), resampling/alignment wrappers (`resample_to`, `align`), categorical safety helpers, and import of `make_terrain_expression` / `make_resample_expression` |
| `src/lunarscout/map_algebra/_registry.py` | Modified | Enriched `OperationSpec` entries for `terrain.slope`, `terrain.aspect`, `terrain.hillshade`, and `alignment.resample_to` with parameter types/defaults, output-dtype-rule, output-units-rule, validity-rule, cost-class, and examples |
| `src/lunarscout/map_algebra/_spatial.py` | Modified during review | Added structured parameter conversion, destination-nodata validation, GDAL-compatible Boolean mode execution, and correct zero-threshold coverage behavior |
| `docs/USER_GUIDE.md` | Modified | Added Terrain Operations section and Resampling and Alignment section to the Map Algebra chapter; removed stale "deferred" language from the grids section |
| `docs/ARCHITECTURE.md` | Modified | Expanded section 20.3 with recursive per-node window requests, one-pixel terrain expansion and crop-once, cumulative halo propagation, destination-to-source window mapping, exact nearest-neighbour 64-bit integer sampling, and deferred-capability list.  Added `_spatial.py` to module inventory |
| `docs/map-algebra-implementation-plan.md` | Modified | Reconciled the status header, milestone summary, terrain/alignment/resampling completion, and explicit partial status for general focal halos, backend coverage, and extreme declared nodata metadata |
| `CHANGELOG.md` | Modified | Added "Critical-path slice 2" entry under Unreleased |
| `examples/21_map_algebra_terrain_resample.py` | Added | Self-contained example: slope and hillshade expressions, explicit resampling, combined scoring, windowed writes, and canonical validity verification |
| `examples/README.md` | Modified | Added example 21 to the script index |
| `tests/map_algebra/test_public_terrain.py` | Added | 44 public terrain tests |
| `tests/map_algebra/test_public_resample.py` | Added | 66 public resampling and registry tests |
| `tests/test_examples.py` | Modified during review | Runs example 21 in the deterministic smoke sequence and checks its outputs |

## Public Signatures

```python
# Terrain
ma.slope(raster, *,
    output_nodata=np.nan, units="degrees",
    compute_edges=False, scale=1.0,
) -> Raster | RasterExpression

ma.aspect(raster, *,
    output_nodata=np.nan, compute_edges=False,
) -> Raster | RasterExpression

ma.hillshade(raster, *,
    output_nodata=0, azimuth=315.0, altitude=45.0,
    compute_edges=False, scale=1.0, z_factor=1.0,
) -> Raster | RasterExpression

# Resampling / Alignment
ma.resample_to(raster, grid, *,
    resampling="nearest", output_dtype=None,
    validity_coverage_threshold=None,
    categorical=None, allow_unsafe=False,
) -> Raster | RasterExpression

ma.align(raster, *, to, resampling="nearest",
    output_nodata="auto", output_dtype=None,
    validity_coverage_threshold=None,
    categorical=None, allow_unsafe=False,
) -> Raster
```

### Dispatch Rules

- **`Raster` operand** — constructs a terrain or resampling expression node,
  calls `ma.compute()` on it, and returns a materialized `Raster`.
- **`RasterExpression` operand** — returns an expression node without evaluation.
- **Other types** — raises `TypeError` directing the caller to the root-level
  array API (`ls.slope`, `ls.aspect`, `ls.hillshade`) or `ma.resample_to()`
  for expression operands.

## Safety Rules

Resampling applies the following categorical-versus-continuous safety rules:

| Condition | Behaviour |
|-----------|-----------|
| Integer or Boolean source dtype (default `categorical=None`) | Only `nearest` and `mode` allowed.  Interpolating or aggregating raises `AlignmentError`. |
| `mode` on explicitly continuous data (`categorical=False`) | Raises `AlignmentError` unless `allow_unsafe=True`. |
| Boolean source with any non-categorical resampling | Raises `AlignmentError` unless `allow_unsafe=True`. |
| `allow_unsafe=True` | Explicitly bypasses categorical, cast, and integer-interpolation safety rejections, but not unsupported dtypes or invalid algorithms. |
| `categorical=True` / `categorical=False` | Overrides the auto-inference from dtype kind. |
| Explicit `output_dtype` | Must be a safe source conversion unless `allow_unsafe=True`; categorical inference still uses the source dtype. |
| Continuous interpolation into integer output | Rejected unless `allow_unsafe=True` because it can round or truncate. |
| `ma.align(..., output_nodata=...)` | `"auto"` preserves source nodata metadata; numeric sets it; `None` disables it without changing canonical validity. |

Validity resampling:

- **Default** — nearest-neighbour categorical validity.
- **`validity_coverage_threshold`** — when supplied as a float between 0 and 1,
  each output pixel requires at least that fraction of valid source coverage
  (uses `average` resampling on the validity mask).
- **Exact 64-bit nearest** — `nearest` resampling uses a custom NumPy
  implementation that directly indexes the source array with integer coordinates,
  preserving exact `int64`/`uint64` payloads beyond the 53-bit mantissa
  precision of IEEE 754 `float64`.

## Registry Metadata

All four new operation IDs (`terrain.slope`, `terrain.aspect`, `terrain.hillshade`,
`alignment.resample_to`) have enriched `OperationSpec` entries including:

- Parameter names, descriptions, and defaults
- `output_dtype_rule` (e.g. `float32`, `uint8`, `source or explicit`)
- `output_units_rule` (e.g. `degrees or percent parameter`, `degrees`, `None`)
- `validity_rule` (canonical gradient validity, nearest-neighbour categorical, etc.)
- `cost_class` (`neighborhood` or `resampling`)
- Concise examples

Registry tests confirm:
- No duplicate operation IDs
- All versions are positive integers
- `list_operations(category="terrain")` and `list_operations(category="alignment")`
  expose the operations correctly
- `list_operations(execution_mode="file_backed")` claims match
  `WINDOWED_OPERATION_IDS`
- `describe_operation()` returns correct parameter, dtype, and units metadata

## Test Coverage (110 new public tests)

### Terrain tests (`test_public_terrain.py` — 44 tests)

| Class | Count | Coverage |
|-------|-------|----------|
| `TestPublicTerrainEager` | 20 | Eager `Raster` returns `Raster`; slope degrees, percent, scale; aspect flat-cell invalidity; hillshade azimuth, altitude, scale, z-factor; `compute_edges` true/false; output-nodata collision; valid-zero hillshade; dtypes and units; structured error rejection for invalid nodata, invalid units, invalid azimuth/altitude; unsupported operand type; identity change per parameter; results match base `lunarscout.terrain` API |
| `TestPublicTerrainExpression` | 12 | Expression return types; `compute()` parity; `write()` (windowed) parity against `compute()`; all three operations expression parity; canonical identity changes with parameters; invalid cells across window boundaries; numeric nodata collision remains valid; dtypes and units on expressions |
| `TestTerrainAgainstBase` | 3 | Slope, aspect, and hillshade results match the established `lunarscout.terrain` base API for representative inputs |
| `TestTerrainErrors` | 4 | Non-elevation (complex) dtype rejection; invalid slope units; invalid hillshade azimuth/altitude |
| `TestTerrainCanonicalIdentity` | 5 | Scientific identity changes for slope `units`, slope `scale`, aspect `compute_edges`, hillshade `azimuth`, hillshade `altitude` |

### Resampling tests (`test_public_resample.py` — 66 tests)

| Class | Count | Coverage |
|-------|-------|----------|
| `TestPublicResampleEager` | 21 | Eager dispatch; reviewed algorithms; alignment and output-nodata contracts; validity; dtype/nodata validation; and zero-threshold coverage |
| `TestPublicResampleExpression` | 20 | Expression construction; compute/public-writer parity; coverage; rotated/differing grids and CRS; exact 64-bit values; masks; identity; and explicit grid reconciliation |
| `TestResamplingSafety` | 12 | Source-based categorical inference; Boolean mode/overrides; categorical and continuous rejection; integer-interpolation and unsafe-cast rejection |
| `TestRegistryFiltering` | 13 | Category/execution filtering, metadata, duplicate/version checks, executor agreement, and public-signature parameter agreement |

### Execution modes exercised

- Eager `Raster` versus expression `compute()` parity
- Expression `compute()` versus multi-window `write()` parity
- Non-divisible window sizes (3×3, 4×3, 5×4, 3×64, etc.) to exercise internal seams
- Terrain public results checked against the established `lunarscout.terrain` base

## Implementation-Plan Reconciliation

### Completed or substantially completed

- Registration and public availability of `slope`, `aspect`, `hillshade`
- Whole-array versus tiled terrain parity across internal boundaries
- Explicit `ma.resample_to` expression nodes
- Eager `ma.align` `Raster` adapter
- Source-coordinate halo calculation for terrain (one-pixel)
- Exact crop-once terrain execution
- Cross-grid window planning for reviewed resampling modes (`nearest`, `bilinear`, `cubic`, `lanczos`, `average`)
- Categorical-versus-continuous resampling safety rules
- Validity resampling (default nearest, optional coverage threshold)
- Exact 64-bit integer nearest-neighbour resampling
- Registry metadata enrichment for all four new operation IDs
- Public tests, documentation, and example script

### Partial

- General focal statistics, convolution, and morphology windowed execution
  (eager implementations exist; expression nodes exist; window kernels do not)
- Local fusion across consecutive nodes
- Completed-window journal and resume
- Cancellation and progress hooks
- Global, zonal, and distance bounded execution
- Region adapters and cross-window labelling reconciliation
- Temporal spatial-window and time-batch mapping
- Comprehensive resource-scaling benchmark evidence
- Exact extreme-`int64`/`uint64` declared GeoTIFF nodata metadata remains
  partial because Rasterio/GDAL cannot represent every such sentinel exactly;
  exact nearest value payloads and explicit validity masks are supported.

### Intentionally skipped

- TestPyPI publication (deferred to a later milestone for real PyPI)

## Test Totals

- Map-algebra tests: **765 passed**
- Full ordinary CPU suite: **1223 passed, 17 skipped**
- `git diff --check`: clean (no whitespace issues)

Skipped tests (17) are GPU-only (require `LUNARSCOUT_REQUIRE_NUMBA_CUDA=1`)
or SPICE-gated; none were introduced by this pass.

## Example Script

`examples/21_map_algebra_terrain_resample.py` demonstrates:

1. Creating eager terrain values from a DEM `Raster` and true file-backed
   slope and hillshade expressions from `ma.source()`
2. Writing terrain products in bounded windows via `ma.write()`
3. Verifying windowed output matches eager `compute()` via `Raster.array_equal()`
4. Constructing a shifted, higher-resolution destination `GeoReference`
5. Explicitly resampling hillshade onto the new grid with `ma.resample_to()`
6. Combining slope and sunlight in a local expression with validity handling
7. Writing the combined score in bounded windows
8. Reading back and verifying canonical validity (fill value does not equal validity)

The example uses the same deterministic synthetic scenario as examples 18–20,
requires no GPU, and is safe to run repeatedly.

## Architecture: Key Design Points

### Recursive per-node window requests

Each operation node in the DAG requests the window it needs from its operands.
Leaf nodes serve data from the bounded `SourceWindowCache`; intermediate nodes
compute on the windows returned by their operands.  A per-window memo table
prevents redundant computation for shared sub-expressions.

### One-pixel terrain halo and crop-once

Terrain nodes declare `halo=1`.  During windowed execution, each terrain node
expands its request by one pixel on each edge, evaluates the full terrain kernel
through the same existing scientific implementation used by eager compute, then
crops back to the exact output window.  This cycle occurs once per terrain node.

### Cumulative halo

The planner walks the DAG in reverse topological order and propagates each
node's halo requirement upward.  Two nested terrain nodes require a cumulative
halo of two pixels from the source.  `ExecutionPlan.maximum_halo` reports the
largest cumulative source halo.

### Resampling breaks the halo chain

`alignment.resample_to` nodes map into a different pixel coordinate system.
Their interpolation support is accounted for by the destination-to-source
window mapper rather than expressed as an output-grid halo.

### Destination-window to source-window mapping

An output window is mapped conservatively into the source pixel grid by
forming its spatial envelope, transforming those bounds through the CRS with
21-point edge densification, adding interpolation support pixels, and clipping
to the source extent.

### No commit performed

All changes remain uncommitted after final semantic and integration review.

## Final Review Corrections

The final review found and fixed issues that the initial completion report did
not detect:

- categorical inference had incorrectly used `output_dtype` instead of the
  source dtype;
- continuous interpolation into integer output and other unsafe casts were
  not rejected as documented;
- `ma.align()` omitted its output-nodata contract;
- a validity threshold of zero could mark pixels outside source coverage valid;
- Boolean `mode` attempted to pass a Boolean destination buffer to GDAL;
- the example converted eager terrain results into constant expressions, so it
  did not exercise halo-aware file-backed terrain, and its final score was a
  `Raster` incorrectly passed to `ma.write()`;
- the new example was absent from executable example smoke coverage; and
- several broad general-focal/planner checkboxes were checked based only on
  terrain-specific completion.

These corrections are included in the final test totals above.
