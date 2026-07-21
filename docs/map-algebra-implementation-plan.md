# Broad Map-Algebra API Implementation Plan

Status: proposed plan for review; Phase A–C implemented, Phases D–I and
remaining inventory items are not yet implemented.

Target: `0.2.0rc1`

Last updated: 2026-07-20

This plan defines a broad, reusable map-algebra surface for Lunarscout. It is
intended to be detailed enough for an implementation agent to work through one
checked milestone at a time without relying on the superseded managed-runtime
plans. `docs/ARCHITECTURE.md`, `docs/USER_GUIDE.md`, `docs/PLAN1.md`, and
`AGENTS.md` remain authoritative where this plan is silent.

A checked item means implementation and the stated verification evidence both
exist. Draft code alone is not sufficient.

## 1. Outcome and boundaries

The outcome is an array-oriented map-algebra API that is pleasant in notebooks
and also safe for regional rasters that must be processed window by window. It
must preserve Lunarscout's existing rules for explicit grids, validity, bounded
resources, lazy optional capabilities, structured errors, and durable output.

Map algebra treats rasters as spatially registered fields and combines them
with a small number of operation families:

- **local operations** calculate each output pixel from the corresponding
  input pixel or pixels, such as `slope <= 8` or a weighted score;
- **focal operations** use a neighborhood around each output pixel, such as a
  local mean, dilation, or roughness measure;
- **zonal operations** group pixels by an explicitly supplied zone raster and
  summarize each group;
- **global operations** reduce a complete raster to statistics or use
  whole-raster information; and
- **distance operations** measure proximity to explicitly supplied seed
  pixels.

The word "map" is important. Two arrays with the same shape are not
necessarily maps of the same place. Lunarscout therefore carries the grid,
validity, units, and numerical rules through an operation rather than treating
map algebra as unqualified NumPy arithmetic. NumPy remains the appropriate
tool for arrays that have no spatial meaning.

The primary lunar mission workflows are:

- combining slope, roughness, lighting, Earth visibility, thermal proxies,
  science targets, hazards, and operational constraints;
- building threshold, suitability, exclusion, and weighted-score rasters;
- measuring and cleaning candidate regions;
- computing clearance and proximity fields around hazards or resources;
- summarizing values by candidate region or another explicitly supplied zone
  raster; and
- reducing large rasters or temporal products without loading an entire region
  or time cube unnecessarily.

The implementation must not assume that terrestrial datasets or terrestrial
semantics are available. In particular:

- [ ] Require every raster input explicitly; do not download or silently
  consult Earth basemaps, SRTM, land cover, roads, coastlines, hydrology, a
  geoid, magnetic models, or weather datasets.
- [ ] Treat "Earth visibility" as a lunar celestial-geometry product, not as an
  Earth-surface raster dependency.
- [ ] Do not assume WGS84, mean sea level, north-up grids, square pixels, or an
  Earth radius.
- [ ] Use the input CRS and affine transform. Reject operations whose requested
  physical interpretation cannot be derived safely from them.
- [ ] Keep generic numerical operations planetary-neutral. Put operations that
  require a lunar datum, radius, gravity, or other body model in a clearly
  named terrain or lunar-science API with explicit parameters.
- [ ] Do not fold mission policy into scientific operations. Thresholds,
  weights, invalid-area treatment, and suitability rules must be caller inputs.

This release does not include route finding, battery simulation, thermal
simulation, traverse policy, crater recognition, rock detection, hydrologic
flow modeling, remote execution, an unrestricted expression-language service,
or automatic discovery of external datasets. Straight-line distance fields are
in scope; least-cost path planning remains a later product family.

### 1.1 Why add `Raster` and `RasterExpression`

`Raster` is the eager, already-materialized value. It keeps ordinary NumPy
values together with the spatial and scientific metadata needed to combine
them safely. It is useful when the complete inputs fit comfortably in memory
and immediate feedback is desirable.

`RasterExpression` is an immutable description of a calculation that has not
yet run. For example, a source node for `slope.tif`, a comparison node for
`<= 8`, and a Boolean combination node form a small directed acyclic graph.
The expression carries inferred grid, dtype, units, validity behavior, halo,
and operation versions, but it carries no open datasets or computed regional
arrays.

This extra abstraction is warranted because a plain Python expression over
eager arrays cannot provide the required regional behavior. Lunarscout needs
to inspect the entire calculation before execution so it can:

- reject mismatched grids, units, unsafe casts, and unsupported operations
  before modifying output;
- read only the source windows and focal halos needed for each output tile;
- bound memory independently of total raster size;
- fuse compatible local work and avoid unnecessary intermediate GeoTIFFs;
- identify the scientific calculation for provenance and restart checks;
- resume a staged product without executing arbitrary user callbacks; and
- present a human- and machine-inspectable plan before a long calculation.

The expression graph is not intended to replace NumPy, become a general Python
compiler, or serve as a remote execution language in `0.2`. Its job is narrower:
describe registered raster operations well enough for safe eager
materialization or bounded file-backed execution.

## 2. Architectural decisions to approve before implementation

These decisions remove ambiguities that would otherwise cause incompatible
implementations.

### 2.1 Public package shape

- [x] Add public module `lunarscout.map_algebra`, normally imported as:

  ```python
  import lunarscout as ls

  ma = ls.map_algebra
  ```

- [x] Export the module itself from the package root, plus the common value
  types ``Raster`` and ``RasterExpression``. Do not export dozens of individual
  algebra operations from ``lunarscout.__init__``.
- [ ] Keep ``TemporalRaster`` and ``TemporalRasterExpression`` under
  ``lunarscout.map_algebra`` for ``0.2`` until temporal usage shows they belong in
  the already curated package root.
- [x] Keep existing tuple-returning APIs compatible. ``read_geotiff()``,
  ``slope()``, ``align()``, and region functions do not change return type.
- [x] Add adapters that deliberately cross between existing APIs and the new
  types; do not implicitly wrap every existing result.

### 2.2 Two explicit execution modes

The API has two modes with visibly different entry points:

1. **Eager mode** accepts `Raster`, NumPy arrays through an explicit adapter,
   or scalars, and returns `Raster`. It may hold a complete raster in memory.
2. **File-backed mode** starts with `source(path, ...)`, constructs a lazy
   `RasterExpression`, and is executed only by `compute()` or `write()`. It
   reads bounded windows and never silently materializes a regional raster.

Both modes use the same operation specifications, dtype inference, unit rules,
and validity rules. They are separate execution strategies, not two scientific
definitions of an operation.

- [x] Never accept a path in an eager operation.
- [x] Never make a function return ``Raster`` for one input type and a completed
  output path for another.
- [x] Make materialization explicit: ``ma.compute(expression)`` returns a
  ``Raster``; ``ma.write(path, expression, ...)`` returns a ``Path``
  (``ma.write`` deferred to Phase D).
- [x] Permit ``Raster.expression()`` to create a constant in-memory expression,
  but document that it retains the complete raster.
- [x] If every raster operand is a ``Raster``, execute eagerly and return
  ``Raster``. If any operand is a ``RasterExpression``, convert eager ``Raster``
  operands to explicit in-memory constant nodes and return
  ``RasterExpression``. A path is never converted implicitly.
- [x] Do not implement a string parser or ``eval``. Python operators build a
  sealed expression graph from registered operations.

### 2.3 Public value model

Implement an eager value type conceptually equivalent to:

```python
@dataclass(frozen=True, slots=True, eq=False)
class Raster:
    values: NDArray[Any]          # exactly two-dimensional
    georef: GeoReference
    valid: NDArray[np.bool_]      # True means scientifically valid
    units: str | None = None
    name: str | None = None
```

`eq=False` is deliberate because dataclass-generated object equality is not a
useful raster equality: NumPy comparison returns arrays, whole-raster exact
comparison can be expensive, invalid payloads may differ without changing
scientific values, and floating-point tolerance is operation-dependent. The
map-algebra implementation supplies a custom cell-by-cell `__eq__`/`__ne__`
that returns a Boolean `Raster`; it does not define object/value equality.
`Raster` and `RasterExpression` are explicitly unhashable, and whole-value
comparison uses named helpers.

- [x] Validate shape, dtype, grid dimensions, validity shape, and read-only
  metadata in ``Raster.__post_init__``.
- [x] Permit real numeric and Boolean values; reject object, string, datetime,
  and complex dtypes for ``0.2``.
- [x] Store ``values`` and ``valid`` as ordinary NumPy arrays. Do not make
  ``np.ma.MaskedArray`` the internal truth.
- [x] Make the validity array canonical after ingestion. A nodata payload is an
  encoding detail, not a value that every operation repeatedly compares.
- [x] Copy neither array by default, document that freezing the dataclass does
  not make the arrays immutable, and provide ``copy()`` and ``readonly()`` helpers.
- [x] Expose ``shape``, ``dtype``, ``height``, ``width``, ``nbytes``, ``all_valid``, and
  ``invalid_count`` properties.
- [x] Implement ``filled(value)`` and ``masked()`` as explicit conversions.
- [x] Implement ``with_name()``, ``with_units()``, and ``with_validity()`` as
  non-mutating helpers.
- [x] Implement ``same_grid(other)``, ``same_metadata(other)``,
  ``array_equal(other, *, equal_invalid_payload=False)``, and
  ``allclose(other, *, rtol, atol, equal_nan=False)`` with documented validity
  semantics. Add a separate test assertion helper with useful mismatch detail.
- [x] Do not use ``Raster`` equality or hashing for expression deduplication;
  use node identity or an explicit source/content identity.
- [x] Set ``__hash__ = None`` explicitly; ``dataclass(frozen=True, eq=False)`` alone
  would otherwise retain identity hashing.
- [x] Treat ``Raster.georef.nodata`` as source/encoding metadata only. In-memory
  validity must not depend on repeatedly comparing values with it; output
  adapters create a ``GeoReference`` carrying the selected output nodata.

Constructors and adapters:

```python
ma.raster(values, georef, *, valid=None, nodata="auto", units=None, name=None)
ma.from_masked_array(values, georef, *, units=None, name=None)
ma.read(path, *, band=1, units=None) -> Raster
ma.from_existing(values, georef, *, units=None, name=None) -> Raster
ma.to_existing(raster, *, nodata) -> tuple[NDArray[Any], GeoReference]
```

- [x] ``valid=None, nodata="auto"`` derives validity from a supplied masked-array
  mask and ``georef.nodata``; floating NaN matching a NaN nodata is invalid.
- [x] Do not treat every NaN or infinity as invalid unless requested with
  ``nonfinite="invalid"``. Default ingestion follows mask/nodata metadata.
- [x] Require a caller-supplied validity mask to have exactly the raster shape.
  Do not allow spatial broadcasting of validity masks.
- [x] ``ma.read()`` must combine the selected GDAL band mask, dataset mask, and
  declared nodata into canonical validity and retain the native band values.
- [ ] ``ma.write()`` must write both deterministic payload and a GDAL validity
  mask; a payload such as zero must remain usable as valid science data.

`Raster` always means a spatial raster with a real `GeoReference`.
Non-georeferenced two-dimensional arrays remain ordinary NumPy arrays and may
use NumPy algebra. Do not invent an identity or sentinel grid: doing so would
allow unrelated arrays to pass the same-grid check and undermine the central
spatial safety contract. A separate non-spatial value type may be considered
later only for a demonstrated Lunarscout workflow.

### 2.4 Expression model

Implement a public immutable ``RasterExpression`` with no public constructor.
Users obtain expressions from ``ma.source()``, ``Raster.expression()``, coordinate
constructors, or registered operators.

- [x] Expression nodes contain an operation identifier, immutable normalized
  parameters, operands, inferred dtype, grid, units, required halo, and a
  versioned semantic identifier.
- [x] Do not store arbitrary callbacks, lambdas, open datasets, CUDA objects,
  or mutable arrays in serializable expression nodes.
- [x] Allow Python arithmetic, comparison, and bitwise Boolean operators.
- [x] Reject Python ``and``, ``or``, chained comparisons, and truth testing with an
  actionable error explaining use of ``&``, ``|``, and parentheses.
- [x] Implement a stable JSON representation for provenance and restart
  identity. It is not a remote-execution contract in ``0.2``.
- [x] Hash source identity from resolved path, band, file size, modification
  time, grid, dtype, nodata, and mask flags. Provide an optional strong file
  digest for workflows that require it.
- [x] Keep operation identifiers and semantic versions independent of Python
  function names so aliases do not break restart metadata.

#### Canonical expression representation

The canonical representation must be deterministic across supported Python
versions and independent of object addresses, dictionary insertion order,
temporary paths, worker counts, JIT artifacts, and backend tuning. Define and
publish a versioned schema containing:

```text
schema_version
root_node_id
nodes[]
  node_id
  operation_id
  semantic_version
  normalized_parameters
  operand_node_ids and typed scalars
  inferred grid, dtype, units, validity rule, and halo
sources[]
  source_node_id
  normalized source descriptor
  source scientific identity
```

- [ ] Serialize nodes in deterministic topological order and dictionaries with
  sorted keys, fixed separators, UTF-8 encoding, and no insignificant
  whitespace.
- [ ] Encode scalars with an explicit type. Preserve arbitrary-size integers
  as decimal text and floating values with an exact hexadecimal form; do not
  emit JSON NaN or Infinity tokens.
- [ ] Normalize enums, paths, CRS text, affine values, dtype strings,
  footprints, percentile lists, and other structured parameters in one
  versioned helper.
- [ ] Reject parameters that cannot be represented canonically rather than
  falling back to `repr()` or pickle.
- [ ] Make parsing untrusted expression JSON explicitly out of scope for
  `0.2`; serialization is for inspection, provenance, and identity of graphs
  constructed through the Python API.

Use three identities because they answer different questions:

- **Scientific identity** hashes the canonical operation graph, semantic
  versions, normalized scientific parameters, grids, units, validity rules,
  and source scientific identities. It answers "what calculation and inputs
  does this claim to represent?"
- **Restart identity** additionally binds output dtype, nodata/invalid
  encoding, mask behavior, band layout, source stat or strong identities, and
  storage algorithm versions. It answers "may this staged output be resumed?"
- **Execution-cache identity** additionally binds Lunarscout version, selected
  backend, kernel implementation version, dependency/runtime versions, and
  relevant tuning. It answers "may compiled or tuned execution artifacts be
  reused?"

- [ ] Keep Numba/CUDA JIT artifacts and device properties out of scientific
  and restart identities; bind them only to execution-cache identity.
- [ ] A non-semantic implementation refactor must not change scientific or
  restart identity. A change that can alter scientific values or validity must
  bump the operation semantic version; a change to staged-storage
  compatibility must bump the storage/restart version; an implementation-only
  kernel change must invalidate execution-cache identity as needed.
- [ ] Version identity algorithms independently and add golden canonical-JSON
  and digest fixtures so accidental changes are detected during review.
- [ ] Provide `expression.describe()` and `expression.to_canonical_json()`;
  `describe()` is concise and human-oriented, while canonical JSON is complete
  and machine-oriented.

### 2.5 Grid and scalar rules

- [x] Every non-scalar raster operand in an operation must use the same grid by
  ``require_same_grid()``. Never accept shape equality as compatibility.
- [x] Never align implicitly. Users must call eager ``ma.align()`` or expression
  ``ma.resample_to()`` explicitly.
- [x] Scalars may broadcast over a raster. A length-one or one-dimensional
  array is not a scalar and is rejected.
- [x] A raster operation needs at least one raster operand so its output grid is
  unambiguous.
- [ ] Preserve rotated and anisotropic affine transforms for local operations.
- [ ] Neighborhood and distance operations must calculate halo and physical
  spacing from both affine basis vectors, not merely ``abs(pixel_size_x)``.

### 2.6 Validity rules

Use these defaults consistently in eager and file-backed execution:

- [x] Unary operations preserve input validity, then invalidate newly
  undefined results according to the operation's documented finite/domain
  policy.
- [x] Ordinary multi-raster operations use the intersection of raster operand
  validity masks.
- [x] Scalars are always valid unless they are explicitly represented as the
  ``ma.invalid`` sentinel.
- [x] Comparisons at invalid pixels are invalid, not false.
- [x] Boolean ``and``, ``or``, and ``xor`` use strict validity intersection rather
  than three-valued short-circuit semantics.
- [x] ``where(condition, x, y)`` is valid where the condition is valid and the
  selected branch is valid. Invalidity in the unselected branch does not
  invalidate the result.
- [x] ``coalesce(a, b, ...)`` selects the first valid value per pixel and is
  invalid only where no operand is valid.
- [x] ``fill_invalid(raster, value)`` makes filled pixels valid; ``set_invalid``
  changes validity without relying on a payload.
- [ ] Division by zero, invalid logarithm/square-root domains, and newly
  generated NaN/inf follow a public `numeric_errors=` option with values
  `"invalid"` (default), `"keep"`, and `"raise"`.
- [ ] File outputs always fill invalid cells deterministically before writing
  and write the GDAL mask. Validate that the chosen fill/nodata is exactly
  representable by the output dtype.

Validity ingestion must also retain provenance. Rasterio normally returns a
mask array even when the dataset has no stored mask; that array may be derived
from nodata or may be an all-valid fallback. The implementation must inspect
Rasterio mask flags rather than assume that every returned mask was explicitly
stored.

- [x] Record whether source validity came from a per-band/internal mask,
  dataset mask, alpha, nodata-derived mask, caller mask, or all-valid fallback.
- [x] When no authoritative explicit mask exists, derive validity explicitly
  from declared nodata, including exact integer and NaN handling.
- [x] Conservatively intersect independent explicit validity sources and
  document conflict behavior; never let a valid payload such as zero imply
  invalidity merely because zero is a common fill value.
- [x] Include validity provenance in source descriptions and manifests, and
  test nodata-only, explicit-mask-only, alpha, all-valid, and conflicting
  cases.

### 2.7 Dtype rules

- [ ] Centralize dtype inference in one helper used by eager and expression
  modes. *(dispatched per-operation, not centralized yet)*
- [x] Use documented NumPy 2.x promotion (``np.result_type``) for ordinary
  arithmetic, minimum/maximum, and ``where``, with explicit exceptions below.
- [x] Comparisons and Boolean operations return ``bool`` in memory.
- [x] True division returns at least ``float32``; use ``float64`` if an operand is
  ``float64`` or safe scalar inference requires it.
- [ ] Integer reductions use an accumulator dtype that prevents ordinary small
  raster overflow; document exact sum/count/mean/std output rules.
- [x] Integer arithmetic does not silently saturate. ``overflow="wrap"`` follows
  NumPy, ``overflow="raise"`` is the public default for checked eager integer
  operations, and ``overflow="promote"`` promotes to a safe supported dtype.
- [x] ``cast()`` supports ``casting="safe"``, ``"same_kind"``, and ``"unsafe"`` and an
  explicit overflow policy.
- [ ] Boolean GeoTIFF output requires an explicit integer encoding, defaulting
  to ``uint8`` values 0 and 1 with a separate validity mask.

### 2.8 Unit rules

Units are conservative metadata in `0.2`; do not add a unit-conversion
dependency until actual lunar workflows justify one.

- [x] Store units as an optional trimmed string and preserve ``None`` as unknown.
- [x] Add/subtract/comparison require exact unit equality when both raster
  operands have units. A numeric scalar threshold or offset is interpreted in
  the raster operand's units. If two raster operands are used and only one has
  units, raise unless ``allow_unknown_units=True``.
- [x] Multiplication/division of two unit-bearing rasters require explicit
  ``output_units``; scalar multiplication/division preserves raster units.
- [ ] Powers require a dimensionless scalar exponent and explicit output units
  for a unit-bearing raster unless exponent is one.
- [x] Trigonometric operations require ``degrees`` or ``radians``; inverse
  trigonometric operations declare their output angle unit.
- [x] ``clip``, reclassification, reductions, and comparisons document whether
  they preserve, replace, or remove units.
- [x] No operation infers meters merely because a coordinate number is large.

### 2.9 Qualities for future LLM-assisted analysis

A future Lunar Analyst assistant may help a human formulate an analysis, but
the human must be able to inspect the proposed calculation and the library must
not rely on the model to remember hidden numerical or geospatial rules. The
most useful library quality is therefore a small, composable, self-describing,
deterministic API rather than special "AI" behavior.

- [ ] Provide machine-readable operation metadata from the sealed registry:
  identifier, summary, operand kinds, parameters with types/defaults/ranges,
  units, output dtype rule, validity rule, execution modes, cost class, and
  examples.
- [ ] Provide `ma.describe_operation(id)` and `ma.list_operations(...)` using
  the same metadata that drives validation and documentation, so descriptions
  cannot silently diverge from code.
- [ ] Provide `ma.explain(expression)` with an ordered plain-language account
  of sources, explicit alignments, thresholds, units, validity choices,
  reductions, output encoding, and scientific algorithm versions.
- [ ] State that `explain()` is an audit aid, not evidence that thresholds or
  policy choices are scientifically appropriate. Human or application review
  remains required.
- [ ] Provide `ma.plan(expression, *, output=None)` as a read-only dry run. It
  validates the graph and reports output grid/dtype/units, source identities,
  passes, halos, estimated peak memory, temporary disk, output size, backend
  availability, and unsupported nodes without calculating or writing pixels.
- [ ] Make threshold inclusivity, angle units, connectivity, edge behavior,
  invalid policy, resampling, and approximate algorithms explicit parameters
  rather than context-dependent defaults where a silent choice could change a
  mission conclusion.
- [ ] Return structured errors with stable codes and repair-oriented details
  such as acceptable values, differing grid fields, and the operation/argument
  that failed. Do not include speculative instructions in the library error.
- [ ] Record canonical expression JSON, an analyst-facing explanation, source
  identities, library version, and output contract in durable provenance so a
  human can audit what an assisting model proposed.
- [ ] Provide deterministic `repr`/description ordering and compact examples
  that avoid aliases. Equivalent supported expressions should normalize to the
  same scientific identity where semantics truly match.
- [ ] Add adversarial tests for plausible model mistakes: omitted parentheses
  around comparisons, Python `and`/`or`, shape-only matches, degrees versus
  radians, fraction versus encoded byte values, implicit Earth/WGS84
  assumptions, unsafe nodata zero, and accidental eager materialization.
- [ ] Keep approval, filesystem allowlists, external job submission, and tool
  authorization in the calling application. Lunarscout validates and explains
  calculations but does not decide that an assistant is authorized to execute
  them.
- [ ] Do not accept generated JSON or text as executable input in `0.2`. A
  future governed parser must have a separately reviewed schema, size/depth
  limits, source policy, and security model.

## 3. Proposed public API inventory

Exact spelling may change during API review, but implementation must not begin
until this inventory is accepted and examples read naturally.

### 3.1 Local cell-by-cell operations

Support both functions and appropriate `Raster`/`RasterExpression` operators:

- [x] Arithmetic: ``add``, ``subtract``, ``multiply``, ``divide``, ``floor_divide``,
  ``remainder``, ``power``, ``negative``, ``positive``, ``absolute``.
- [ ] Pairwise/stack combination: ``minimum``, ``maximum``, ``sum_layers``,
  ``mean_layers``, ``min_layers``, ``max_layers``.
- [x] Comparisons: ``equal``, ``not_equal``, ``less``, ``less_equal``, ``greater``,
  ``greater_equal``, ``isclose``.
- [x] Boolean: ``logical_not``, ``logical_and``, ``logical_or``, ``logical_xor``;
  require Boolean operands rather than treating all nonzero numbers as true.
- [x] Conditional/validity: ``where``, ``coalesce``, ``is_valid``, ``is_invalid``,
  ``set_invalid``, ``fill_invalid``.
- [x] Range and conversion: ``clip``, ``cast``, ``round``, ``floor``, ``ceil``, ``trunc``.
- [x] Math: ``sqrt``, ``square``, ``exp``, ``log``, ``log10``, ``sin``, ``cos``, ``tan``,
  ``arcsin``, ``arccos``, ``arctan``, ``arctan2``, ``degrees``, ``radians``, ``hypot``.
- [ ] Classification: ``reclassify_values``, ``reclassify_ranges``, ``digitize``,
  and ``one_hot``. Require explicit default behavior for unmatched valid cells:
  preserve, set a value, or invalidate.
- [ ] Normalize: ``normalize_minmax`` and ``standardize``, using supplied statistics
  or explicit two-pass execution. Never hide a global pre-pass.

Example acceptance target:

```python
slope = ma.read("slope.tif", units="degrees")
sun = ma.read("mean_sun.tif", units="fraction")
ma.require_same_grid(slope, sun)

candidate = (slope <= 8.0) & (sun >= 0.60)
slope_score = 1.0 - ma.normalize_minmax(slope, minimum=0.0, maximum=8.0)
sun_score = ma.normalize_minmax(sun, minimum=0.0, maximum=1.0)
score = ma.where(candidate, 0.4 * sun_score + 0.6 * slope_score, ma.invalid)
ma.write("candidate_score.tif", score, dtype="float32", nodata=np.nan)
```

The corresponding file-backed expression differs at the source boundary, not
in its scientific formula:

```python
slope = ma.source("slope.tif", units="degrees")
sun = ma.source("mean_sun.tif", units="fraction")

candidate = (slope <= 8.0) & (sun >= 0.60)

print(ma.explain(candidate))
print(ma.plan(candidate, output="candidate.tif"))
ma.write(
    "candidate.tif",
    candidate,
    dtype="uint8",
    invalid_value=0,
    overwrite=True,
)
```

`ma.read()` reads all values now and returns `Raster`; `ma.source()` reads only
metadata and returns `RasterExpression`; `ma.compute()` explicitly
materializes an expression; and `ma.write()` evaluates it in bounded windows.

### 3.2 Coordinate rasters

- [ ] `row_indices(grid)` and `column_indices(grid)`.
- [ ] `projected_x(grid, anchor="center")` and
  `projected_y(grid, anchor="center")`.
- [ ] `longitude(grid, anchor="center")` and
  `latitude(grid, anchor="center")`, using the grid's own geodetic CRS.
- [ ] Generate coordinate windows lazily in file mode; do not allocate two
  full coordinate rasters merely to process one output window.
- [ ] Clearly label longitude/latitude units and axis order.
- [ ] Do not provide an implicit WGS84 transform.

### 3.3 Neighborhood and morphology operations

Neighborhood size is odd and positive. Support rectangular windows and an
explicit binary footprint. File-backed execution must read the required halo
and crop to the destination window.

- [ ] `focal_sum`, `focal_mean`, `focal_min`, `focal_max`, `focal_range`,
  `focal_std`, `focal_count`, `focal_median`, and selected percentiles.
- [ ] `convolve(kernel, *, normalize=False)` with finite two-dimensional
  numeric kernels only; no arbitrary callback kernels in file mode.
- [ ] `dilate`, `erode`, `opening`, `closing`, and `majority` for Boolean or
  explicitly classified inputs.
- [ ] Edge modes: `invalid` (default), `constant`, `nearest`, `reflect`, and
  `wrap`; document that `wrap` is mathematical and usually inappropriate for
  regional lunar rasters.
- [ ] Valid-neighbor policy: `require_all`, `ignore_invalid` with
  `min_valid_count`, or `propagate_center`. Record it in expression identity.
- [ ] Integrate existing region cleanup behavior with shared morphology
  helpers without changing current public results.
- [ ] Register `slope`, `aspect`, and `hillshade` as map-algebra operations.
  Eager `Raster` calls delegate to the existing scientific implementations.
  Expression calls create nodes with a one-pixel source halo and window kernels
  that preserve the existing nodata, `compute_edges`, dtype, scale, unit, and
  numerical behavior.
- [ ] Compare each terrain expression over many internal window boundaries
  against its existing whole-array implementation. If exact semantic parity is
  not initially achievable, declare that operation eager-only; never silently
  materialize a file-backed source or publish a seam-bearing result.
- [ ] Consider `roughness`, terrain ruggedness index, topographic position
  index, and curvature only after their definitions, units, edge behavior, and
  GDAL compatibility targets have independent tests.

### 3.4 Region and zonal operations

- [ ] Provide `label_regions`, `region_sizes`, `filter_regions_by_size`, and
  `find_borders` adapters accepting and returning `Raster` while preserving the
  existing array APIs.
- [ ] Add configurable four- or eight-neighbor connectivity in the new API;
  preserve eight-neighbor defaults for existing APIs.
- [ ] `zonal_stats(values, zones, *, statistics, zone_nodata=...)` returns a
  table-like `ZonalStatistics` value independent of pandas, conceptually:

  ```python
  @dataclass(frozen=True, slots=True)
  class ZonalStatistics:
      zone_ids: NDArray[np.integer]
      columns: tuple[str, ...]
      values: Mapping[str, NDArray[Any]]
      valid: Mapping[str, NDArray[np.bool_]]
      units: Mapping[str, str | None]
  ```

- [ ] Sort rows by numeric zone ID and return one row per observed valid zone
  by default. An explicit `include_zone_ids` may request rows for empty zones.
- [ ] Represent undefined statistics with per-column validity, not an
  overloaded zone or numeric sentinel. Counts remain valid integer zero for an
  explicitly requested empty zone; mean/min/max and similar statistics are
  invalid there.
- [ ] Provide deterministic iteration yielding immutable row records plus
  `to_records()`, `to_dict()`, `to_json()`, and `write_csv()`. Conversion must
  preserve large integer zone IDs exactly.
- [ ] Keep mergeable streaming accumulator state private and separate from the
  finalized `ZonalStatistics` result.
- [ ] Required statistics: count, valid count, invalid count, sum, mean, min,
  max, range, standard deviation, variance, median, and requested percentiles.
- [ ] `zonal_raster(values, zones, statistic=...)` broadcasts one statistic
  back to valid zone cells.
- [ ] Define zone IDs as integer or Boolean values. Reject floating zone IDs
  rather than truncating them.
- [ ] Zone zero is ordinary unless explicitly configured as background.
- [ ] Invalid value cells are excluded from statistics; invalid zone cells are
  not assigned to any zone.
- [ ] Implement bounded accumulation for count/sum/min/max/mean/variance.
  Median and percentile may use an explicit exact in-memory mode or documented
  approximate streaming mode; never silently switch algorithms.
- [ ] Provide CSV/JSON-friendly conversion without requiring pandas.

### 3.5 Global reductions

- [ ] `statistics(raster, ...)` returns count, invalid count, sum, mean, min,
  max, range, variance, and standard deviation with documented accumulator
  precision.
- [ ] `histogram(raster, *, bins, range=None)` supports explicit edges and
  bounded streaming.
- [ ] `unique_counts(raster, *, max_unique=...)` fails predictably when a
  safety bound is exceeded.
- [ ] `percentile(raster, q, *, method="exact"|"approximate", ...)` makes
  memory/accuracy behavior explicit.
- [ ] Reductions return Python/NumPy scalar or result dataclass values, not a
  one-cell georeferenced raster.
- [ ] Empty-valid-domain behavior is defined per reduction and uses structured
  errors or explicit empty results rather than NumPy warnings alone.

### 3.6 Distance fields

Distance fields support hazard clearance and proximity screening without
introducing route policy.

- [ ] `distance_to(seeds, *, metric, units, max_distance=None)` with Boolean
  seeds, canonical validity, and metrics `euclidean`, `taxicab`, and
  `chessboard` where scientifically meaningful.
- [ ] `signed_distance(mask, ...)` defines Boolean `True` pixels as inside and
  leaves input-invalid pixels invalid by default. At a valid `True` pixel, the
  value is the positive center-to-center distance to the nearest valid `False`
  pixel; at a valid `False` pixel, it is the negative center-to-center distance
  to the nearest valid `True` pixel. The class interface therefore lies midway
  between opposite-class pixel centers and normally has no stored zero-valued
  center. Use this convention in eager, file-backed, CPU, and any later CUDA
  implementations; define a structured empty-opposite-class error for all-True
  or all-False inputs unless an explicit finite fallback is requested.
- [ ] Units are `pixels` or physical CRS units. Physical Euclidean distance
  must honor anisotropic and rotated affine basis vectors.
- [ ] In `0.2`, taxicab and chessboard distances are pixel-unit metrics only;
  reject requests to label them as physical distance on anisotropic or rotated
  grids.
- [ ] Reject physical distance for geographic/angular grids in `0.2` unless an
  explicit, reviewed lunar body/geodesic model is supplied. Do not substitute
  Earth geodesics.
- [ ] Valid seed pixels are the only seeds. By default, input-invalid pixels
  remain invalid in the output but do not bend or block straight-line distance
  measured at other pixels. An explicit `invalid_output="compute"` option may
  calculate values there. Barrier-aware distance is deferred with cost-distance
  and path planning.
- [ ] Define deterministic behavior for no seeds, all seeds, seeds on invalid
  pixels, raster edges, and `max_distance` clipping.
- [ ] Implement a validated CPU reference first. Add CUDA only after CPU tests
  and an independent reference comparison pass.
- [ ] For file-backed Euclidean distance, select and document a genuinely
  bounded exact algorithm or explicitly label a tiled approximation and its
  error bound. Do not run `scipy.ndimage.distance_transform_edt` on a silently
  materialized regional raster.
- [ ] Keep accumulated-cost distance, allocation, backlink rasters, and
  least-cost routes out of `0.2`; those cross into the path-planning design.

### 3.7 Alignment and resampling expressions

- [ ] `ma.align(raster, to=..., ...) -> Raster` delegates to the existing eager
  alignment implementation while preserving canonical validity.
- [ ] `ma.resample_to(expression, grid, *, resampling, ...)` is an explicit
  expression node and may not be inserted automatically by another operation.
- [ ] Resample the validity mask conservatively: nearest for categorical
  validity by default, with a documented coverage-threshold option for
  interpolating numeric data.
- [ ] Distinguish categorical from continuous resampling and reject obviously
  unsafe combinations unless explicitly overridden.
- [ ] Add integration tests for differing CRS, shifted origins, partial
  coverage, nodata, rotated grids, and exact 64-bit nodata payloads.

### 3.8 Temporal map algebra

Time is a named axis with UTC coordinates, not an extra spatial band. Do not
add an implicit time axis to `RasterExpression`. Introduce distinct
`TemporalRaster` and `TemporalRasterExpression` types so validation can require
both a common spatial grid and compatible time coordinates.

- [ ] Define eager `TemporalRaster` with `values[time, y, x]`, UTC `times`, one
  spatial `GeoReference`, canonical `valid[time, y, x]`, units, and signal
  name. Provide explicit lossless adapters where possible to and from existing
  `TemporalCube` without changing the existing class.
- [ ] Define immutable `TemporalRasterExpression` nodes for temporal sources,
  layer-wise local operations, explicit temporal alignment, and reductions.
  Do not let ordinary `ma.source()` accept a temporal series.
- [ ] Add `ma.temporal_source(series)` for an open
  `TemporalGeoTiffSeries` or an explicitly opened series manifest. It reads
  manifest and layer metadata without retaining every layer open. Multi-band
  product GeoTIFFs require a separate adapter because their storage contract
  differs from `TemporalGeoTiffSeries`.
- [ ] Any temporal-expression operand makes an operation temporal and lazy. A
  static `Raster` or `RasterExpression` may broadcast across time only after
  exact spatial-grid validation; a scalar broadcasts across time and space.
- [ ] Combining two temporal operands requires exactly equal ordered UTC
  coordinates by default. Temporal resampling, nearest selection, and
  interpolation are separate explicit operations with tolerance, edge, and
  validity rules.
- [ ] Classify temporal nodes as **layer-wise** or **reducing**. Layer-wise
  nodes retain `(time, y, x)` and are processed in bounded spatial-window and
  time batches. Reducing nodes consume time batches and return a spatial
  `Raster` or `RasterExpression` while retaining bounded accumulator state for
  the current output window.
- [ ] Choose time-batch size from an explicit memory budget and record it in
  the execution plan. A series with thousands of layers must not imply one
  resident array or one open dataset per timestamp.
- [ ] Reuse bounded dataset caches and the existing streaming reducer
  infrastructure. State whether execution is layer-major or spatial-window
  major for each source/output layout and report the expected read pattern in
  `ma.plan()`.
- [ ] Reuse existing mean/min/max/std semantics and add count, sum, variance,
  percentile, `any`, `all`, threshold duration, and exceedance count only with
  documented sample, interval, nodata, and all-invalid behavior.
- [ ] Make a temporal reduction an ordinary composable spatial expression. For
  example, `ma.temporal_mean(sun_series) >= 0.60` may combine with a static
  slope expression and be written window by window without first creating a
  complete mean GeoTIFF.
- [ ] File-backed temporal mapping initially writes the existing timestamped
  GeoTIFF-series format through `TemporalGeoTiffSeriesWriter`. Generic
  multi-band BigTIFF expression output is deferred until its mask, timestamp,
  band-count, and resume contracts are reviewed.
- [ ] `compute()` is the only operation that may explicitly materialize a
  complete temporal result. Preflight its estimated bytes and require an
  explicit override above a documented safety threshold.
- [ ] Do not generalize specialized mission-duration or safe-haven reducers
  into a vague temporal expression if doing so would lose their scientific
  interval contracts.

## 4. Internal modules and helper functions

Use focused modules rather than one large `map_algebra.py`. Names beginning
with `_` are private and may be adjusted, but their responsibilities and shared
semantics must remain centralized.

```text
src/lunarscout/
  raster.py                         # public eager Raster value
  map_algebra/
    __init__.py                     # curated public namespace
    local.py                        # public local functions
    focal.py                        # public neighborhood functions
    zonal.py                        # public zonal functions/results
    reductions.py                   # public global reductions/results
    distance.py                     # public distance functions
    coordinates.py                  # public coordinate expressions
    temporal.py                     # explicit temporal adapters
    expression.py                   # public expression/source facade
    _model.py                       # node and operation descriptors
    _temporal_model.py              # temporal values and expression nodes
    _registry.py                    # sealed operation registry
    _serialization.py               # canonical typed JSON representation
    _identity.py                    # scientific/restart/cache identities
    _explain.py                     # human explanation and dry-run reports
    _validation.py                  # operands, parameters, grids, units
    _validity.py                    # mask combination and numeric domains
    _dtypes.py                      # promotion, casting, overflow
    _units.py                       # conservative unit rules
    _eager.py                       # eager dispatcher
    _planner.py                     # graph validation and window plan
    _temporal_planner.py            # spatial-window/time-batch plans
    _windows.py                     # window/halo enumeration and cropping
    _sources.py                     # GeoTIFF/in-memory/coordinate sources
    _kernels.py                     # CPU NumPy/SciPy kernels
    _reducers.py                    # streaming reduction accumulators
    _distance_cpu.py                # reference distance implementations
    _writer.py                      # staged GeoTIFF expression output
    _manifest.py                    # expression identity/restart metadata
```

### 4.1 Validation helpers

Implement and unit-test private helpers with single responsibilities:

- [ ] `_as_raster_operand(value, *, argument)` accepts only `Raster` or a real
  scalar in eager operations.
- [ ] `_as_expression_operand(value, *, argument, grid_hint)` accepts only
  `RasterExpression`, `Raster`, or a real scalar.
- [ ] `_require_raster_shape(values, georef)` validates exactly two dimensions.
- [ ] `_require_common_grid(operands)` reports every differing grid field in a
  structured error.
- [ ] `_normalize_scalar(value)` rejects arrays disguised as scalars, complex
  values, and unsupported scalar dtypes.
- [ ] `_normalize_footprint(size, footprint)` validates odd dimensions and
  calculates halo on all four sides.
- [ ] `_normalize_numeric_errors`, `_normalize_overflow`,
  `_normalize_edge_mode`, and `_normalize_valid_neighbor_policy` return typed
  literals and stable error codes.
- [ ] `_validate_output_encoding(dtype, nodata, invalid_value)` shares GeoTIFF
  representability checks rather than duplicating them.

### 4.2 Validity helpers

- [ ] `_valid_from_nodata(values, nodata)` handles exact integer nodata, finite
  floating nodata, and NaN nodata without lossy coercion.
- [ ] `_combine_validity_strict(*rasters)` intersects raster validity.
- [ ] `_where_validity(condition, x, y)` implements selected-branch validity.
- [ ] `_coalesce_values_and_validity(...)` performs ordinary per-pixel
  selection after all operands needed for the current window are available.
  The first implementation reads every coalesce operand window; correctness
  must not depend on static validity analysis.
- [ ] Defer coalesce read short-circuiting. A later optional runtime
  optimization may stop requesting later operand windows only when values
  already computed for that output window leave no unresolved invalid pixels.
  Failure to prove that condition always falls back to reading all operands;
  do not add general SSA-style validity proof to the `0.2` planner.
- [ ] `_apply_numeric_domain(valid, values, policy, operation)` handles new
  non-finite/domain errors consistently.
- [ ] `_fill_invalid_exact(values, valid, fill, dtype)` validates exact
  representation and never mutates an input array.
- [ ] `_read_rasterio_validity(dataset, band, window)` combines the band mask,
  dataset mask, nodata, and alpha semantics with dedicated tests and returns
  both canonical validity and a normalized provenance description. Inspect
  Rasterio mask flags so a synthesized all-valid/nodata mask is not mislabeled
  as an explicitly stored mask.

### 4.3 Dtype helpers

- [ ] `_result_dtype(operation, operand_dtypes, scalars, parameters)` is the
  sole dtype-inference entry point.
- [ ] `_accumulator_dtype(operation, source_dtype)` defines streaming
  accumulator precision.
- [ ] `_checked_integer_kernel(...)` detects overflow without allocating
  unbounded temporary arrays.
- [ ] `_cast_values_and_fill(...)` applies casting and output encoding in a
  deterministic order.
- [ ] Add table-driven tests covering every supported dtype pair and boundary
  scalar, including `uint64` values beyond float exact range.

### 4.4 Operation registry

Define an internal immutable descriptor such as:

```python
OperationSpec(
    id="local.add",
    version=1,
    arity=2,
    category="local",
    infer_dtype=...,
    infer_units=...,
    infer_halo=...,
    eager_kernel=...,
    window_kernel=...,
    validity_rule="strict",
)
```

- [ ] Registration occurs at import from static library code only; users
  cannot register arbitrary kernels in `0.2`.
- [ ] Registry import must not initialize CUDA, open datasets, or import SPICE.
- [ ] Reject duplicate identifiers and invalid versions at test time.
- [ ] Ensure every public operation has an operation spec, documentation,
  validity test, dtype test, and tests for every execution mode its descriptor
  claims to support.
- [ ] Generate an internal coverage report from the registry so eager-only,
  windowed, multi-pass, and unsupported modes are explicit rather than
  accidental.
- [ ] Generate machine-readable public operation descriptions from the same
  descriptors. Parameter documentation, defaults, execution support, and
  validity rules must be testable against actual signatures.

### 4.5 Expression planner

- [ ] Topologically validate the graph and detect cycles defensively.
- [ ] Enforce documented limits on graph nodes, depth, source count,
  normalized-parameter bytes, footprint dimensions, and requested output
  bands. Limits prevent accidental or generated expressions from exhausting
  planning resources and fail before source execution or output staging.
- [ ] Infer one output grid, dtype, units, validity behavior, and maximum halo
  before creating output staging.
- [ ] Fuse consecutive local operations into one window task to avoid
  unnecessary full-window writes; correctness comes before aggressive fusion.
- [ ] Do not fuse across global reductions, resampling, distance transforms, or
  operations with incompatible halos.
- [ ] Reuse a source window within a task when multiple nodes request it.
- [ ] Bound source dataset handles, decoded windows, and output queues.
- [ ] Select window sizes from output block geometry with a conservative
  default of 128 by 128; record the choice in progress metadata but not
  scientific identity.
- [ ] Calculate halos in source pixel coordinates and crop exactly once.
- [ ] Emit a readable plan description for diagnostics and tests.
- [ ] Implement read-only `ma.plan()` and `ma.explain()` on top of normalized
  graph and planner data. Neither function may execute numerical kernels,
  create staging, or write output.

### 4.6 File-backed sources and output

- [ ] `ma.source(path, *, band=1, units=None, identity="stat"|"sha256")` reads
  metadata only and returns an expression without retaining an open dataset.
- [ ] Validate source existence, driver, band, dtype, grid, nodata, and mask
  before output modification.
- [ ] Open datasets lazily during execution and close them deterministically.
- [ ] Extend or reuse the existing durable product-storage patterns for
  staging, overwrite protection, cancellation, progress, journaling, and
  atomic publication.
- [ ] A completed-window journal is authoritative. Restart recomputes an
  unjournaled window even when its TIFF block contains plausible data.
- [ ] Bind restart metadata to expression JSON, source identities, grid,
  dtype, units, validity/nodata encoding, window layout, and algorithm
  versions.
- [ ] Store scientific, restart, and execution-cache identities separately;
  never invalidate scientific provenance merely because a worker count or JIT
  cache changed.
- [ ] Never delete a previous complete output until its staged replacement has
  closed and validated successfully.
- [ ] Define safe `start_fresh` cleanup using exact resolved staging paths.
- [ ] Support single-band GeoTIFF output in the first slice. Add generic
  multiband expression output only after its band metadata contract is defined.

### 4.7 Backends

- [ ] Implement and validate eager NumPy/SciPy CPU behavior first.
- [ ] Implement bounded windowed CPU behavior second and compare it exactly or
  within documented tolerance to eager results.
- [ ] Use Numba CPU only where benchmarks show a useful improvement and cache
  behavior is acceptable in installed wheels.
- [ ] Do not require CUDA for core map algebra.
- [ ] Add CUDA per operation only after a CPU reference, backend-independent
  semantics, correctness comparison, memory bound, and realistic benchmark
  exist.
- [ ] Follow existing backend semantics: CPU never probes CUDA; explicit CUDA
  never falls back; auto may fall back only for capability/availability, not
  after a CUDA execution failure.
- [ ] Do not advertise `backend=` on operations with no supported alternative
  backend.

## 5. Structured errors

Add an error hierarchy under `LunarscoutError`:

```text
MapAlgebraError
  RasterValidationError
  MapAlgebraGridError
  MapAlgebraDTypeError
  MapAlgebraUnitError
  MapAlgebraExpressionError
  MapAlgebraOperationError
  MapAlgebraStorageError
  DistanceFieldError
```

- [x] Reuse ``GridMismatchError``, ``GeoTiffError``, and
  ``OperationCancelledError`` where their existing public meaning is exact; wrap
  only when more algebra context is needed.
- [x] Assign stable ``code=`` values for invalid operands, grid mismatch,
  unsupported dtype, unsafe cast, overflow, unit mismatch, invalid expression,
  unavailable source, invalid footprint, empty reduction, output conflict,
  restart mismatch, and unsupported physical distance.
- [ ] Include operation ID, argument name, dtype, grid differences, units,
  source path, output path, or window coordinates in `details=` as applicable.
- [ ] Never expose a raw Rasterio, SciPy, NumPy, Numba, or CUDA exception as the
  only public diagnostic.

## 6. Implementation phases and progress checklist

Each phase should be a reviewable change set. Do not begin CUDA optimization or
the long-tail operation inventory before the semantic foundation passes.

### Phase A: Contract tests and public skeleton

- [x] Add ``tests/map_algebra/`` and shared fixtures for north-up, anisotropic,
  rotated, shifted, differing-CRS, masked, nodata, and partial-coverage grids.
- [x] Add the public error classes and import-boundary tests.
- [x] Add the ``Raster`` model, constructors, explicit adapters, and repr.
- [x] Add explicit whole-raster comparison helpers and tests proving that
  ``==``/``!=`` are cell-by-cell algebra while hashing and implicit truth testing
  are unavailable by design.
- [x] Add ``map_algebra`` namespace with placeholder-free public exports.
- [x] Add ``ma.read()`` and a private test writer to prove mask round trips. Do
  not expose ``ma.write()`` until its atomic output contract is implemented in
  Phase D.
- [x] Verify ``import lunarscout`` still initializes no CUDA/SPICE context, opens
  no raster, performs no network access, and writes no files.
- [ ] Review and freeze Sections 2 and 3 before expanding operations.

Acceptance evidence:

- [x] Raster values, validity, grid, dtype, units, and name round-trip in
  memory.
- [x] GDAL validity masks round-trip independently of valid zero values and
  nodata payload.
- [x] Validity provenance distinguishes explicit mask, alpha, nodata-derived,
  caller-supplied, and all-valid sources.
- [x] Existing public tests pass unchanged.

### Phase B: Eager local algebra

- [x] Implement shared validation, validity, dtype, unit, and numeric-error
  helpers.
- [x] Implement arithmetic and comparison operators.
- [x] Implement and test the mixed-mode rule: eager-only operands return
  ``Raster``; any expression operand returns ``RasterExpression``.
- [x] Implement strict Boolean operations and truth-test diagnostics.
- [x] Implement ``where``, ``coalesce``, validity functions, clip, cast, and the
  core math inventory.
- [ ] Implement reclassification and stack combination.
- [ ] Add property-based-style randomized tests using deterministic seeds;
  compare valid cells with direct NumPy reference calculations.
- [ ] Test every invalidity, dtype, overflow, unit, scalar, and grid branch.
- [ ] Convert the landing-site screening example to a new additional example,
  retaining the old array-oriented example as compatibility evidence.

Acceptance evidence:

- [x] A complete terrain-plus-lighting candidate expression needs no manual
  mask bookkeeping after input construction.
- [x] Mismatched georeferenced rasters fail before numerical calculation.
- [x] Results match reference NumPy values and the documented validity rules.

### Phase C: Expressions and bounded local execution

- [x] Implement immutable expression nodes and the sealed operation registry.
  *(expression nodes implemented; registry deferred)*
- [x] Implement GeoTIFF, in-memory, scalar, and coordinate sources.
  *(coordinate sources deferred)*
- [x] Implement expression operator overloads and stable JSON identity.
- [ ] Implement canonical typed serialization plus distinct scientific,
  restart, and execution-cache identities with golden fixtures.
  *(scientific identity via SHA-256 implemented; restart/cache identities deferred)*
- [x] Implement ``describe()``, ``ma.explain()``, ``ma.plan()``, and machine-readable
  operation introspection without executing kernels or writing files.
  *(explain and plan implemented)*
- [ ] Implement the planner, window enumeration, local fusion, source cache,
  cancellation checks, and progress events.
- [ ] Implement window kernels for every Phase B local operation.
- [ ] Test many window/block sizes, including outputs smaller than one block
  and dimensions not divisible by 128.
- [ ] Measure peak memory against increasing raster dimensions and prove it is
  bounded by window/graph complexity rather than total raster area.

Acceptance evidence:

- [ ] Eager and windowed outputs have identical payload and validity for
  integer/Boolean operations and documented tolerances for floating operations.
- [ ] Source datasets and caches close after success, failure, and cancellation.

### Phase D: Durable expression output

- [x] Implement output preflight, staged GeoTIFF creation, deterministic
  invalid payload, atomic publication, and GDAL mask writing (via
  ``write_mask()`` at dataset creation time).
- [ ] Implement journal-based resume. *(manifest identity check exists
  but no per-window journal; full recalculation on mismatch)*
- [x] Bind restarts to expression scientific identity, output dtype,
  invalid fill, and grid dimensions.
- [ ] Add injected-failure tests before write, during calculation, after block
  write, before journal update, during close, and before publish.
- [ ] Add cancellation/resume tests and concurrent-output conflict tests.
- [x] Confirm failed overwrite preserves the previous complete output.
  *(two-phase atomic staging: new TIFF+manifest written to temp dir,
  old files replaced only after both succeed; overwrite=True required)*

Acceptance evidence:

- [ ] A killed multi-window operation resumes without trusting unjournaled
  blocks. *(no multi-window execution exists yet)*

### Phase E: Focal and morphology operations

- [x] Implement footprint/halo/edge/valid-neighbor contracts.
  *(five edge modes, three valid-neighbor policies, ``cval`` parameter)*
- [x] Implement the required focal statistics and convolution.
  *(sum, mean, min, max, range, std with ddof, count, median, convolve)*
- [x] Implement shared morphology and region adapters.
  *(dilate, erode, opening, closing, majority with validity masking)*
- [ ] Implement or explicitly defer windowed terrain nodes for slope, aspect,
  and hillshade based on whole-array parity tests. *(deferred)*
- [ ] Compare eager and tiled halo results across internal window boundaries.
  *(no tiled execution exists yet)*
- [ ] Test rotated/anisotropic grids and document which focal operations are
  pixel-neighborhood rather than physical-radius operations. *(deferred)*
- [x] Benchmark SciPy, NumPy sliding windows, and Numba candidates before
  choosing optimized kernels. *(SciPy selected as baseline; NumPy sliding
  windows and Numba candidates not yet benchmarked)*

Acceptance evidence:

- [ ] No seams occur at tile boundaries, and edge/invalid behavior matches an
  independent whole-array reference. *(no tiled execution)*

### Phase F: Global and zonal reductions

- [ ] Implement stable streaming accumulator objects with merge/finalize
  tests.
- [ ] Implement global statistics, histogram, unique counts, and exact versus
  approximate percentiles.
- [ ] Implement zonal tabular statistics and broadcast zonal rasters.
- [ ] Implement the finalized `ZonalStatistics` row ordering, per-column
  validity, immutable iteration, exact zone-ID conversions, and serializers.
- [ ] Define deterministic zone ordering and JSON/CSV conversion.
- [ ] Test large zone IDs, sparse IDs, negative IDs, zero, Boolean zones,
  empty zones, all-invalid zones, and accumulator precision.
- [ ] Test window-order independence where floating-point tolerances allow it.

Acceptance evidence:

- [ ] Streaming and eager results agree within a stated tolerance without
  memory proportional to raster area, except explicitly selected exact
  percentile modes.

### Phase G: Distance fields

- [ ] Freeze distance metrics, units, affine handling, and invalid-area rules.
- [ ] Implement small CPU reference algorithms and independent analytic test
  cases.
- [ ] Implement eager distance fields.
- [ ] Evaluate exact bounded file-backed algorithms; document the selected
  algorithm, complexity, temporary storage, and failure recovery.
- [ ] Implement file-backed distance only after the review accepts exactness or
  a clearly quantified approximation.
- [ ] Add physical-distance tests for square, anisotropic, and rotated projected
  lunar grids.
- [ ] Add explicit rejection tests for unconfigured angular/geographic grids.
- [ ] Benchmark representative hazard masks, sparse seeds, dense seeds, and
  empty/all-seed edge cases.
- [ ] Evaluate CUDA only after CPU acceptance; do not make it a release blocker
  unless separately approved.

Acceptance evidence:

- [ ] Results match SciPy or analytic references where their assumptions match,
  and memory/temporary-disk bounds are recorded.

### Phase H: Temporal adapters

- [ ] Implement `TemporalRaster`, explicit `TemporalCube` adapters, and
  `TemporalRasterExpression` without changing existing temporal classes.
- [ ] Implement explicit layer-wise local expression nodes and static spatial
  raster broadcasting.
- [ ] Implement `ma.temporal_source()` and bounded spatial-window/time-batch
  mapping over `TemporalGeoTiffSeries`.
- [ ] Add time-coordinate equality and explicit alignment validation.
- [ ] Make approved temporal reducers produce composable spatial expressions
  using existing streaming accumulators where semantics match.
- [ ] Add documented sample/interval, validity, empty-domain, and output-unit
  semantics for every reducer.
- [ ] Add approximately 3,000-layer execution tests proving bounded dataset
  handles, bounded resident batches, and accurate planning estimates.
- [ ] Ensure no temporal helper constructs a full file-backed cube unless the
  caller explicitly requests materialization.

Acceptance evidence:

- [ ] Layer-wise eager and streamed results match and preserve UTC metadata,
  masks, grids, signal names, and units.

### Phase I: Documentation, examples, and release gate

- [ ] Add a map-algebra chapter to `docs/USER_GUIDE.md` covering eager versus
  file-backed workflows, grids, validity, dtypes, units, and lunar constraints.
- [ ] Update `docs/ARCHITECTURE.md` with the accepted model, execution planner,
  and storage flow.
- [ ] Add API reference tables for every operation and its validity/dtype/unit
  behavior.
- [ ] Publish the machine-readable operation catalog, canonical expression
  schema, identity distinctions, and examples of `explain()` and `plan()`.
- [ ] Add runnable examples for terrain-lighting screening, weighted scoring,
  hazard clearance, focal cleanup, zonal candidate summaries, large file-backed
  expressions, and temporal threshold summaries.
- [ ] Use synthetic lunar grids and downloadable lunar products where needed;
  no example may depend on an unmentioned Earth dataset.
- [ ] Include a QGIS inspection example proving valid zero values remain visible
  and invalid pixels are transparent through the dataset mask.
- [ ] Add an "assistant proposes, human reviews, library validates" example in
  which an expression is explained and dry-run before execution. Keep tool
  authorization in the example application, not Lunarscout.
- [ ] Record CPU correctness and bounded-memory benchmarks.
- [ ] Build wheel and sdist, inspect contents, run Twine checks, and test the
  installed artifacts without the checkout on `PYTHONPATH`.
- [ ] Run the complete ordinary CPU suite with:

  ```bash
  .venv/bin/python -m pytest -q
  ```

- [ ] Run any implemented CUDA comparisons only with
  `LUNARSCOUT_REQUIRE_NUMBA_CUDA=1` on a visible supported NVIDIA device.
- [ ] Publish and independently install a `0.2.0rc1` TestPyPI candidate before
  describing the map-algebra API as accepted.

## 7. Test matrix

Every operation family must cover the following relevant dimensions. Use
small analytic arrays for semantics and larger generated rasters for execution
and memory behavior.

- [ ] Dtypes: bool, signed/unsigned integers at supported widths, float32, and
  float64.
- [ ] Values: zero, negative, extrema, NaN, positive/negative infinity,
  division by zero, and values adjacent to thresholds.
- [ ] Validity: all valid, all invalid, sparse invalid, nodata-only, GDAL-mask
  only, both agreeing, and both conflicting.
- [ ] Grids: same shape/different CRS, same CRS/shifted affine, anisotropic,
  rotated, north-up, partial overlap, and non-128-multiple dimensions.
- [ ] Operands: scalar-left, scalar-right, raster/raster, repeated source, and
  deep but bounded expression graphs.
- [ ] Mode mixing: eager/eager, eager/expression, expression/eager, rejected
  paths, and explicit in-memory constant nodes.
- [ ] Identity: canonical JSON golden files, typed scalar boundaries,
  normalization aliases, scientific versus restart versus cache changes, and
  deterministic hashes across supported Python versions.
- [ ] Execution: eager, one window, many windows, cancellation, resume,
  overwrite, and injected failure.
- [ ] Numeric policy: keep/invalid/raise, wrap/promote/raise, safe/unsafe casts,
  and accumulator precision.
- [ ] Units: matching, mismatching, unknown, scalar interactions, explicit
  overrides, and angle requirements.
- [ ] Storage: integer/float nodata, valid zero, Boolean encoding, exact uint64
  nodata, mask round trip, compression, and atomic replacement.
- [ ] Import boundary: base CPU install, CUDA extra absent, SPICE absent from
  `sys.modules`, read-only cache, and a working directory with no write access.
- [ ] Spatial boundary: non-georeferenced arrays are rejected by `Raster` and
  expression constructors with guidance to use NumPy or supply a real grid;
  no sentinel grid passes compatibility checks.
- [ ] Agent-error cases: Python Boolean keywords, comparison precedence,
  implicit alignment, unknown units, encoded fraction confusion, unsafe zero
  nodata, unconfigured geographic distance, excessive graph depth, and dry-run
  guarantees.

## 8. Performance and resource requirements

Absolute pixels-per-second requirements would be misleading before reference
hardware, storage, compression, dtype, mask density, and expression complexity
are fixed. Performance acceptance therefore uses a checked-in benchmark
definition, a recorded same-machine baseline, relative regression gates, and
hard resource-scaling requirements. Freeze the baseline table before accepting
optimized Phase C or later kernels; do not move its targets merely to make a
regression pass.

- [ ] Define benchmark classes for: one-source local arithmetic, three-source
  Boolean overlay, a five-node fused local expression, 3x3 and 31x31 focal
  operations, global reduction, sparse/dense zonal reduction, sparse/dense
  distance seeds, one temporal layer-wise expression, and one temporal
  reduction over approximately 3,000 layers.
- [ ] For every benchmark record grid dimensions, dtype, valid fraction,
  compression, block size, source/output storage, cold/warm state, backend,
  worker count, dependency versions, CPU/GPU, RAM, and storage device.
- [ ] Report planning, source open/read, JIT/compile, host-device transfer,
  kernel, synchronization, mask, reduction, compression, journal, close, and
  publication time where applicable rather than only wall-clock total.
- [ ] Record median and dispersion over at least five warm runs after one
  untimed warm-up for short operations. Long regional benchmarks may use three
  runs with the reason recorded.
- [ ] On the same reference environment, fail performance review for an
  unexplained warm median regression greater than 15 percent or peak-memory
  regression greater than 10 percent against the accepted baseline.
- [ ] Accept a more complex optimized kernel only when it improves the target
  workload by at least 20 percent in warm elapsed time, or provides a separately
  documented material memory/I/O benefit, without violating correctness.
- [ ] Measure simple local file-backed expressions against a tiled
  read-and-write copy baseline. Record the ratio and explain compute overhead;
  set operation-specific target ratios after the first reference
  implementation rather than inventing hardware-independent throughput.
- [ ] Establish eager-size guidance in documentation rather than guessing from
  available RAM automatically.
- [ ] File-backed local/focal execution peak memory must be expressible as
  `O(active_sources * window_with_halo + active_intermediates * window)`.
- [ ] Verify bounded-memory behavior at three increasing raster dimensions.
  With window and concurrency settings held constant, peak resident memory may
  not grow by more than 10 percent plus measurement noise when raster area
  increases by at least 16 times; temporary/output disk may scale with area.
- [ ] Dataset-handle caches, window caches, queues, worker counts, and temporary
  files have explicit configurable upper bounds.
- [ ] Default worker counts must not multiply memory beyond the documented
  bound.
- [ ] Global/zonal reducers use bounded accumulator state, except explicitly
  requested exact algorithms whose memory cost is preflighted and documented.
- [ ] Progress reports planning, reading/calculation, reduction passes, writing,
  and publication with completed/total bounded work units.
- [ ] Cancellation is checked between source reads, kernels, reduction merges,
  output writes, journal flushes, and publication.
- [ ] Benchmarks include a representative lunar DEM and masks but do not make
  correctness depend on private mission data.
- [ ] Store benchmark commands, machine-readable results, and a short
  interpretation under `scripts/` and `docs/`; do not encode one developer's
  absolute throughput as a universal hardware promise.

## 9. Documentation required for each public operation

No operation is complete until its docstring and user documentation state:

- [ ] accepted operand kinds and shapes;
- [ ] grid and alignment requirements;
- [ ] mathematical definition;
- [ ] dtype and overflow behavior;
- [ ] validity and newly non-finite behavior;
- [ ] units and angle conventions;
- [ ] edge, footprint, connectivity, or distance semantics where applicable;
- [ ] eager and file-backed availability;
- [ ] CPU/CUDA availability where applicable;
- [ ] memory behavior and number of passes for file-backed execution;
- [ ] structured exceptions and stable error codes; and
- [ ] one minimal lunar-analysis example;
- [ ] machine-readable parameter metadata and one canonical example suitable
  for operation discovery; and
- [ ] the corresponding `explain()` language for scientifically significant
  choices.

## 10. Explicit deferrals

Keep these visible so an implementation agent does not expand scope while
trying to make an example convenient:

- [ ] No automatic reprojection or grid selection during algebra.
- [ ] No sentinel `GeoReference` or coordinate-free `Raster`; use NumPy for
  non-spatial arrays.
- [ ] No arbitrary Python callbacks in serializable/file-backed expressions.
- [ ] No string expression parser, SQL syntax, or remote execution contract.
- [ ] No implicit unit conversion or dimensional-analysis framework.
- [ ] No pandas/xarray/Dask/CuPy dependency in the base public contract. These
  may receive adapters after the NumPy/Rasterio contract is stable.
- [ ] No vector GIS overlay or rasterization beyond separately reviewed helper
  APIs.
- [ ] No Earth-only environmental, hydrologic, road-network, land-cover, or
  weather operations.
- [ ] No assumption that a lunar projected CRS has meters unless CRS metadata
  proves it or the caller supplies an explicit unit contract.
- [ ] No geodesic physical distance on angular grids without an explicit body
  model.
- [ ] No cost-distance, route extraction, rover policy, energy model, thermal
  model, or path optimizer in `0.2`.
- [ ] No silent full-raster materialization in a file-backed operation.
- [ ] No CUDA-only core algebra operation unless separately justified with the
  same explicit exception used for horizon generation.

## 11. Final acceptance definition

The broad map-algebra milestone is complete only when all of the following are
checked:

- [ ] The eager API supports the accepted local, focal, zonal, global, and
  distance inventory with consistent grids, validity, dtype, and units.
- [ ] The accepted file-backed inventory executes with bounded memory and
  durable, resumable, atomic output.
- [ ] Eager and file-backed implementations agree against independent
  references.
- [ ] Dataset masks survive read, calculation, and write without conflating
  valid zero with invalid data.
- [ ] Lunar projected, anisotropic, and rotated grid cases pass; unsafe
  Earth-specific or body-ambiguous assumptions are absent or rejected.
- [ ] Existing terrain, temporal, region, horizon, lighting, and scenario APIs
  remain compatible and their tests pass.
- [ ] Documentation and runnable examples cover both notebook-sized and
  mission-region workflows.
- [ ] Operation discovery, expression explanation, dry-run planning, canonical
  provenance, and repair-oriented structured errors are sufficient for a
  future assisting model to propose an auditable calculation without granting
  it arbitrary execution inside Lunarscout.
- [ ] Clean installed base-wheel tests pass without CUDA initialization or
  hidden source-tree dependencies.
- [ ] A `0.2.0rc1` candidate has been independently installed and evaluated,
  and its limitations are recorded before promotion.
