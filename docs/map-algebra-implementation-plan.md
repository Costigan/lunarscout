# Core Map-Algebra API Implementation Plan

Status: The eager ``0.2`` map-algebra surface, expression construction and
materialization, bounded zero-halo local/coordinate writes, halo-aware
terrain execution, explicit cross-grid resampling, documentation, examples,
writer progress/cancellation, compact durable restart journaling, and local
release-artifact checks are implemented. Registry metadata, identity
separation, remaining eager API gaps, structured-error
normalization, per-operation reference documentation, and adversarial test
coverage remain partial. All further work whose primary purpose is processing
maps too large for memory is deferred by project decision and moved to
`docs/map-algebra-large-raster-plan.md`. TestPyPI publication is skipped by
project decision; publishing will resume at a later milestone intended for
real PyPI.

Target: `0.2.0rc1`

Last updated: 2026-07-22 (example-facing eager numeric audit and plan reconciliation)

This plan defines a broad, reusable map-algebra surface for Lunarscout. It is
intended to be detailed enough for an implementation agent to work through one
checked milestone at a time without relying on the superseded managed-runtime
plans. `docs/ARCHITECTURE.md`, `docs/USER_GUIDE.md`, `docs/PLAN1.md`, and
`AGENTS.md` remain authoritative where this plan is silent.

A checked item means implementation and the stated verification evidence both
exist. Draft code alone is not sufficient.

Unchecked items are annotated where their state is not simply "not started":

- **PARTIAL** means a useful subset exists, but the complete checkbox claim is
  not supported by implementation and evidence.
- **DEFERRED -- LARGE-RASTER PLAN** means the item is intentionally moved to
  `docs/map-algebra-large-raster-plan.md` and is not on the current schedule.
- **SKIPPED BY DECISION** means the work will not be performed for this
  milestone and is not a release blocker.
- **NOT APPLICABLE** means the conditional requirement was reviewed but no
  matching implementation exists in this milestone.

A plain unchecked item under a section-level annotation inherits that section's
state. Any other plain unchecked item is not started or remains unresolved; it
is not counted as partial merely because adjacent functionality exists.

### Reconciled milestone summary

| State | Scope |
| --- | --- |
| Completed | Public value types and adapters; eager local/classification/normalization and stack operations; expression construction and eager ``compute``; bounded zero-halo local and coordinate ``write`` execution; coordinate expression sources; canonical typed JSON and scientific identity; eager focal/morphology, connected-region adapters, global, zonal, and distance operations; temporal adapters and approved streaming reductions; source-sensitive eager focal sum/mean/std/count/min/max precision; atomic output; halo-aware terrain and explicit cross-grid resampling ``write`` execution; writer progress/cancellation and compact checkpoint resume; public terrain/resample wrappers with categorical safety and validity policies; user guide, architecture, examples, and the ordinary CPU suite. |
| Partial | Operation catalog metadata and coverage; analyst-facing ``explain`` detail; canonical identity golden fixtures; full numeric-policy and dtype centralization beyond the audited example-facing eager surface; structured-error normalization; exhaustive API tables; and adversarial/boundary tests. |
| Deferred -- large-raster plan | All further bounded/windowed execution work: general halos and focal kernels, local fusion, cross-window region reconciliation, streaming global/zonal reducers, bounded distance fields, temporal spatial-window/time-batch mapping, concurrency controls, and empirical resource scaling. Current completed bounded capabilities remain supported. |
| Skipped by decision | TestPyPI publication for ``0.2.0rc1``. Local artifact construction, inspection, and isolated installation remain completed evidence. |

### Next implementation sequence

The active, non-large-raster sequence is:

1. centralize numeric policy, dtype inference, validity helpers, and structured
   error translation across eager and expression construction paths;
2. complete canonical identity fixtures, registry coverage, generated operation
   descriptions, `explain()`, and public API reference tables;
3. close adversarial and boundary-test gaps and reconcile examples and release
   documentation to the implemented surface.

The completed bounded writer remains maintained. Any expansion of bounded
operation coverage follows `docs/map-algebra-large-raster-plan.md` only after
that plan is explicitly resumed.

The example-facing portion of sequence item 1 is complete: local arithmetic,
selection, classification, normalization, focal statistics, global/zonal
summaries, temporal reductions, nodata/fill encoding, and unit-bearing power
have focused dtype, validity, unit, non-finite, and exact-integer coverage.
Remaining sole-helper centralization and exhaustive dtype cross-products are
not prerequisites for expanding the notebook-sized examples.

### Project-owner priority preference

**The project owner prefers that remaining core-plan work proceed in the
following criticality order, taking the highest-priority item allowed by its
dependencies.** This ordering excludes the separately deferred large-raster
plan and supersedes document order as a scheduling signal; it does not permit
an incomplete dependency or scientific contract to be skipped.

1. **Critical -- scientific and numeric consistency.** Make shared dtype
   inference, casting, invalid-fill, nodata, accumulator, validity, and unit
   rules authoritative across eager and expression paths. Preserve FP32 and
   exact integer behavior, including values beyond the FP64 exact-integer
   range.
2. **Critical -- public errors and defensive validation.** Normalize
   dependency failures into structured Lunarscout errors, make diagnostic
   details consistent, complete planning limits, and preflight explicit
   temporal materialization.
3. **High -- identity and provenance guarantees.** Separate and independently
   version scientific, restart, and execution-cache identities and protect
   canonical representations with golden fixtures.
4. **High -- operation-registry and execution-claim audit.** Make public
   signatures, registry parameters, scientific rules, and advertised
   execution modes agree, with generated coverage checks.
5. **High -- operation discovery, `explain()`, and `plan()`.** Generate
   complete read-only descriptions from normalized planner and registry data,
   including scientifically significant choices and output contracts.
6. **Medium -- remaining eager API contracts.** Close accepted notebook-sized
   gaps such as selected focal percentiles, direct ``ZonalStatistics`` row
   iteration, shared region-cleanup morphology internals, and reviewed
   extreme-nodata behavior without expanding deferred bounded execution.
   Unit-bearing power, eager morphology, region adapters, immutable zonal
   records, and exact example-facing integer reductions are already complete.
7. **Medium -- adversarial and boundary-test closure.** Complete relevant
   dtype, value, validity, grid, identity, unit, storage, import-boundary, and
   likely-user-error matrices through the public API in fresh processes.
8. **Medium-low -- public reference documentation and examples.** Finish
   per-operation contracts and the missing weighted-score, hazard-clearance,
   zonal-summary, QGIS-mask, and human-review workflow examples.
9. **Low -- eager/core CPU evidence.** Retain reproducible notebook-sized
   correctness benchmarks and practical eager-size guidance; large-raster
   scaling evidence remains in the deferred plan.

## 1. Outcome and boundaries

The outcome is a scientifically consistent array-oriented map-algebra API that
is pleasant in notebooks. It preserves the bounded file-backed subset already
implemented, but further regional-scale execution expansion is governed by the
separate deferred large-raster plan. The core API must preserve Lunarscout's
rules for explicit grids, validity, lazy optional capabilities, structured
errors, and durable output.

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
- reducing rasters or temporal products with explicit eager memory behavior;
  future bounded regional/time-batch execution is deferred separately.

The implementation must not assume that terrestrial datasets or terrestrial
semantics are available. In particular:

- [x] Require every raster input explicitly; do not download or silently
  consult Earth basemaps, SRTM, land cover, roads, coastlines, hydrology, a
  geoid, magnetic models, or weather datasets.
- [x] Treat "Earth visibility" as a lunar celestial-geometry product, not as an
  Earth-surface raster dependency.
- [x] Do not assume WGS84, mean sea level, north-up grids, square pixels, or an
  Earth radius.
- [x] Use the input CRS and affine transform. Reject operations whose requested
  physical interpretation cannot be derived safely from them.
- [x] Keep generic numerical operations planetary-neutral. Put operations that
  require a lunar datum, radius, gravity, or other body model in a clearly
  named terrain or lunar-science API with explicit parameters.
- [x] Do not fold mission policy into scientific operations. Thresholds,
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
- [x] Keep ``TemporalRaster`` and ``TemporalRasterExpression`` under
  ``lunarscout.map_algebra`` for ``0.2`` until temporal usage shows they belong in
  the already curated package root. *(``TemporalRaster`` also exported at root
  via ``ls.TemporalRaster`` for visibility.)*
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
- [x] ``ma.write()`` must write both deterministic payload and a GDAL validity
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

- [x] Serialize nodes in deterministic topological order and dictionaries with
  sorted keys, fixed separators, UTF-8 encoding, and no insignificant
  whitespace. *(topological sort + sort_keys=True in to_json(); hex-float
  scalar encoding and versioned normalization helper deferred.)*
- [x] Encode scalars with an explicit type. Preserve arbitrary-size integers
  as decimal text and floating values with an exact hexadecimal form; do not
  emit JSON NaN or Infinity tokens.
- [x] Normalize enums, paths, CRS text, affine values, dtype strings,
  footprints, percentile lists, and other structured parameters in one
  versioned helper.
- [x] Reject parameters that cannot be represented canonically rather than
  falling back to `repr()` or pickle.
- [x] Make parsing untrusted expression JSON explicitly out of scope for
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
  **PARTIAL:** scientific identity correctly excludes them, but restart and
  execution-cache identities are not yet distinct.
- [ ] A non-semantic implementation refactor must not change scientific or
  restart identity. A change that can alter scientific values or validity must
  bump the operation semantic version; a change to staged-storage
  compatibility must bump the storage/restart version; an implementation-only
  kernel change must invalidate execution-cache identity as needed.
  **PARTIAL:** operation semantic versions exist, but restart and
  execution-cache identities are not independently versioned.
- [ ] Version identity algorithms independently and add golden canonical-JSON
  and digest fixtures so accidental changes are detected during review.
  **PARTIAL:** canonical JSON and scientific digests exist; independent
  identity versions and golden fixtures do not.
- [x] Provide `expression.describe()` and `expression.to_canonical_json()`;
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
- [x] Preserve rotated and anisotropic affine transforms for local operations.
  *(local ops are pixel-by-pixel; output GeoReference inherits from input.)*
- [ ] Neighborhood and distance operations must calculate halo and physical
  spacing from both affine basis vectors, not merely ``abs(pixel_size_x)``.
  **PARTIAL:** distance uses both vectors; focal operations are defined in
  pixel neighborhoods and no physical-radius/window-halo planner exists.

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
- [x] Division by zero, invalid logarithm/square-root domains, and newly
  generated NaN/inf follow a public `numeric_errors=` option with values
  `"invalid"` (default), `"keep"`, and `"raise"`.
  Pointwise arithmetic and math operations that can generate these conditions
  expose the common policy. Comparisons, Boolean/selection, classification,
  and normalization reducers retain their dedicated validity contracts.
- [x] File outputs always fill invalid cells deterministically before writing
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
  modes. **PARTIAL:** ``result_dtype()`` is shared by arithmetic, unary, and
  selection construction/execution. Exact ``where``/``coalesce`` Python
  integer scalar inference, FP32 preservation, and incompatible 64-bit integer
  rejection are now proven across eager, expression, and windowed paths.
  Reclassification output inference also uses the helper, including complete
  source-domain preservation. Normalization eager/expression construction and
  execution share the same FP32-first rule, but the helper is not yet proven to
  be the sole path for every operation and scalar boundary.
- [x] Use documented NumPy 2.x promotion (``np.result_type``) for ordinary
  arithmetic and minimum/maximum. Selection follows it with an exact-integer
  refinement: Python integer branches use their smallest exact dtype, and a
  signed/unsigned combination with no exact supported dtype raises rather than
  selecting FP64.
- [x] Comparisons and Boolean operations return ``bool`` in memory.
- [x] True division returns at least ``float32``; use ``float64`` if an operand is
  ``float64`` or safe scalar inference requires it.
- [x] Preserve ``float32`` execution for ``float32`` inputs and inferred
  ``float32`` outputs. Do not introduce ``float64`` intermediates as a general
  strategy for domain checks, overflow checks, eager/window parity, or future
  accelerator kernels. Consumer-grade GPUs are the expected deployment
  hardware and commonly have substantially lower FP64 throughput. FP64 is
  used only when an input, requested dtype, accumulator contract, or documented
  scientific result explicitly requires it.
- [x] Treat FP64 and 64-bit integer arithmetic as CPU correctness/interchange
  capabilities, not GPU execution prerequisites. Consumer NVIDIA GPUs have
  poor FP64 throughput and software-emulate general ``int64``/``uint64`` ALU
  work. Normal future CUDA hot paths therefore use FP32 and Boolean/8/16/32-bit
  integers. A planner must reject or explicitly route required 64-bit work to
  CPU unless a separately benchmarked kernel establishes an acceptable
  contract; successful CUDA compilation alone is not evidence of suitability.
- [x] Detect signed and unsigned integer overflow entirely in the integer
  domain. In particular, never convert ``int64`` or ``uint64`` values to
  ``float64`` to decide representability: values beyond ``2**53`` must remain
  exact. Promotion selects an exact supported integer dtype when one exists;
  otherwise raise a structured error rather than silently using an inexact
  floating dtype. Add/subtract/multiply/floor-divide/remainder/power and unary
  negate/absolute/square satisfy this contract in eager, expression, and
  windowed execution. Integer power uses bounded repeated squaring.
- [x] Integer reductions use an accumulator dtype that prevents ordinary small
  raster overflow; document exact sum/count/mean/std output rules.
  *(`accumulator_dtype()` is enforced by eager and layer-streamed temporal
  reducers: sums use int64 for signed and Boolean inputs and uint64 for
  unsigned inputs; FP32 mean/std/sum remain FP32; FP64 remains FP64; integer
  mean/std use the CPU/interchange FP64 correctness path; count uses int64;
  and min/max preserve the source dtype. The same rule is enforced for eager
  focal sum/mean/std/count/min/max construction and execution; focal integer
  mean/std use exact-integer CPU moments before their FP64 result. Fixed-width
  sum overflow retains NumPy behavior beyond the accumulator range. Global
  statistics and zonal summaries now also calculate integer sums, extrema,
  ranges, and centered moments without first converting source values to
  FP64.)*
- [x] Integer arithmetic does not silently saturate. ``overflow="wrap"`` follows
  NumPy, ``overflow="raise"`` is the public default for checked eager integer
  operations, and ``overflow="promote"`` promotes to a safe supported dtype.
  Exact integer-domain checks and exact promotion cover add, subtract,
  multiply, floor divide, remainder, power, negate, absolute, and square.
- [x] ``cast()`` supports ``casting="safe"``, ``"same_kind"``, and ``"unsafe"`` and an
  explicit overflow policy. The default value-level policy raises on overflow;
  explicit wrapping is restricted to integer-to-integer casts.
- [x] Boolean GeoTIFF output requires an explicit integer encoding, defaulting
  to ``uint8`` values 0 and 1 with a separate validity mask.
  *(write() auto-converts bool dtype to uint8.)*

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
- [x] Unit-bearing raster powers require a scalar exponent and explicit output
  units unless exponent is one. Unit-bearing bases reject
  raster exponents; scalar exponent one preserves units, other scalar powers
  require a non-empty declaration, and raster exponents carrying no unit
  metadata cannot claim one fixed output unit. Eager, expression, compute,
  identity, registry, and supported windowed execution enforce the same
  contract.
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
  examples. **PARTIAL:** the sealed catalog exposes identifiers and basic rule
  summaries, public parameter types/defaults and enumerated choices, and
  audited execution modes. Numeric ranges and canonical examples remain
  incomplete.
- [ ] Provide `ma.describe_operation(id)` and `ma.list_operations(...)` using
  the same metadata that drives validation and documentation, so descriptions
  cannot silently diverge from code. **PARTIAL:** both discovery APIs are
  public and expression-node construction consults the registry. Public
  parameter metadata is generated from callable signatures and audited by
  tests; prose documentation is still maintained separately.
- [x] Provide `ma.explain(expression)` with an ordered plain-language account
  of sources, explicit alignments, thresholds, units, validity choices,
  reductions, output encoding, and scientific algorithm versions.
  *(The registry-backed explanation includes stable node ordering, source
  identity descriptors, scalar operands and parameters, dtype/unit/validity
  rules, halo, output mask/storage encoding, and semantic versions.)*
- [x] State that `explain()` is an audit aid, not evidence that thresholds or
  policy choices are scientifically appropriate. Human or application review
  remains required.
- [x] Provide `ma.plan(expression, *, output=None)` as a read-only dry run. It
  validates the graph and reports output grid/dtype/units, source identities,
  passes, halos, estimated peak memory, temporary disk, output size, backend
  availability, and unsupported nodes without calculating or writing pixels.
  *(The JSON-serializable report includes canonical identity, reviewed nodes,
  output grid/contract/preflight, windows, passes, halos, peak-memory and
  temporary-storage estimates, backend availability, and an empty
  unsupported-node list after validation. Unsupported operations fail with a
  structured error naming the operation before output creation.)*
- [ ] Make threshold inclusivity, angle units, connectivity, edge behavior,
  invalid policy, resampling, and approximate algorithms explicit parameters
  rather than context-dependent defaults where a silent choice could change a
  mission conclusion.
  **PARTIAL:** implemented families expose and catalog the relevant comparison,
  angle, connectivity, edge, invalid, resampling, and numeric choices. Some
  approximation policies and numeric parameter ranges remain absent.
- [x] Return structured errors with stable codes and repair-oriented details
  such as acceptable values, differing grid fields, and the operation/argument
  that failed. Do not include speculative instructions in the library error.
- [ ] Record canonical expression JSON, an analyst-facing explanation, source
  identities, library version, and output contract in durable provenance so a
  human can audit what an assisting model proposed. **PARTIAL:** whole-raster
  manifests record scientific identity and a small output contract, not the
  complete provenance bundle.
- [ ] Provide deterministic `repr`/description ordering and compact examples
  that avoid aliases. Equivalent supported expressions should normalize to the
  same scientific identity where semantics truly match. **PARTIAL:** canonical
  node ordering and basic identity stability are tested; equivalence
  normalization and golden examples are not.
- [ ] Add adversarial tests for plausible model mistakes: omitted parentheses
  around comparisons, Python `and`/`or`, shape-only matches, degrees versus
  radians, fraction versus encoded byte values, implicit Earth/WGS84
  assumptions, unsafe nodata zero, and accidental eager materialization.
  **PARTIAL:** truth testing, grid mismatch, unit, validity, and geographic
  distance mistakes are covered; the complete adversarial set is not.
- [x] Keep approval, filesystem allowlists, external job submission, and tool
  authorization in the calling application. Lunarscout validates and explains
  calculations but does not decide that an assistant is authorized to execute
  them.
- [x] Do not accept generated JSON or text as executable input in `0.2`. A
  future governed parser must have a separately reviewed schema, size/depth
  limits, source policy, and security model.

## 3. Proposed public API inventory

Exact spelling may change during API review, but implementation must not begin
until this inventory is accepted and examples read naturally.

### 3.1 Local cell-by-cell operations

Support both functions and appropriate `Raster`/`RasterExpression` operators:

- [x] Arithmetic: ``add``, ``subtract``, ``multiply``, ``divide``, ``floor_divide``,
  ``remainder``, ``power``, ``negative``, ``positive``, ``absolute``.
- [x] Pairwise/stack combination: ``minimum``, ``maximum``, ``sum_layers``,
  ``mean_layers``, ``min_layers``, ``max_layers``.
  Variadic helpers compose the ordinary local operations without allocating a
  three-dimensional stack, preserving their checked dtype, numeric, grid,
  unit, expression, and windowed-write contracts.
- [x] Comparisons: ``equal``, ``not_equal``, ``less``, ``less_equal``, ``greater``,
  ``greater_equal``, ``isclose``.
- [x] Boolean: ``logical_not``, ``logical_and``, ``logical_or``, ``logical_xor``;
  require Boolean operands rather than treating all nonzero numbers as true.
- [x] Conditional/validity: ``where``, ``coalesce``, ``is_valid``, ``is_invalid``,
  ``set_invalid``, ``fill_invalid``.
- [x] Range and conversion: ``clip``, ``cast``, ``round``, ``floor``, ``ceil``, ``trunc``.
- [x] Math: ``sqrt``, ``square``, ``exp``, ``log``, ``log10``, ``sin``, ``cos``, ``tan``,
  ``arcsin``, ``arccos``, ``arctan``, ``arctan2``, ``degrees``, ``radians``, ``hypot``.
- [x] Classification: ``reclassify_values``, ``reclassify_ranges``, ``digitize``,
  and ``one_hot``. Require explicit default behavior for unmatched valid cells:
  preserve, set a value, or invalidate.
- [x] Normalize: ``normalize_minmax`` and ``standardize``, using supplied statistics
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

- [x] `row_indices(grid)` and `column_indices(grid)`.
- [x] `projected_x(grid, anchor="center")` and
  `projected_y(grid, anchor="center")`.
- [x] `longitude(grid, anchor="center")` and
  `latitude(grid, anchor="center")`, using the grid's own geodetic CRS.
- [x] Generate coordinate windows lazily in file mode; do not allocate two
  full coordinate rasters merely to process one output window.
- [x] Clearly label longitude/latitude units and axis order.
- [x] Do not provide an implicit WGS84 transform.

### 3.3 Neighborhood and morphology operations

Neighborhood size is odd and positive. Support rectangular windows and an
explicit binary footprint. File-backed execution must read the required halo
and crop to the destination window; that future execution work is deferred to
`docs/map-algebra-large-raster-plan.md`.

- [ ] `focal_sum`, `focal_mean`, `focal_min`, `focal_max`, `focal_range`,
  `focal_std`, `focal_count`, `focal_median`, and selected percentiles.
  **PARTIAL:** the listed statistics except selected focal percentiles are
  implemented for eager ``Raster`` inputs. Expression construction exists,
  but bounded expression execution and window parity are deferred to the
  large-raster plan. Selected eager focal percentiles remain core work.
- [ ] `convolve(kernel, *, normalize=False)` with finite two-dimensional
  numeric kernels only; no arbitrary callback kernels in file mode.
  **PARTIAL:** eager convolution and validation exist; file-backed execution
  does not.
- [ ] `dilate`, `erode`, `opening`, `closing`, and `majority` for Boolean or
  explicitly classified inputs.
  **PARTIAL:** eager Boolean morphology exists; executable expression nodes do
  not. Bounded execution is deferred to the large-raster plan.
- [ ] Edge modes: `invalid` (default), `constant`, `nearest`, `reflect`, and
  `wrap`; document that `wrap` is mathematical and usually inappropriate for
  regional lunar rasters. **PARTIAL:** all modes are implemented eagerly, but
  the documented caution and file-backed parity evidence are incomplete.
- [x] Valid-neighbor policy: `require_all`, `ignore_invalid` with
  `min_valid_count`, or `propagate_center`. Record it in expression identity.
  The eager contract and expression validation/identity are implemented;
  bounded execution remains deferred to the large-raster plan.
- [ ] Integrate existing region cleanup behavior with shared morphology
  helpers without changing current public results. This eager/shared-helper
  cleanup remains core work.
- [x] Register `slope`, `aspect`, and `hillshade` as map-algebra operations.
  Eager `Raster` calls delegate to the existing scientific implementations.
  Expression calls create nodes with a one-pixel source halo and window kernels
  that preserve the existing nodata, `compute_edges`, dtype, scale, unit, and
  numerical behavior.
- [x] Compare each terrain expression over many internal window boundaries
  against its existing whole-array implementation. If exact semantic parity is
  not initially achievable, declare that operation eager-only; never silently
  materialize a file-backed source or publish a seam-bearing result.
- [ ] Consider `roughness`, terrain ruggedness index, topographic position
  index, and curvature only after their definitions, units, edge behavior, and
  GDAL compatibility targets have independent tests.

### 3.4 Region and zonal operations

- [x] Provide `label_regions`, `region_sizes`, `filter_regions_by_size`, and
  `find_borders` adapters accepting and returning `Raster` while preserving the
  existing array APIs. The eager adapter slice is complete; bounded labeling/
  filtering and cross-window reconciliation are deferred to the large-raster
  plan.
- [x] Add configurable four- or eight-neighbor connectivity in the new API;
  preserve eight-neighbor defaults for existing APIs. Bounded reconciliation
  remains deferred.
- [x] `zonal_stats(values, zones, *, statistics, zone_nodata=...)` returns a
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

- [x] Sort rows by numeric zone ID and return one row per observed valid zone
  by default. An explicit `include_zone_ids` may request rows for empty zones.
- [x] Represent undefined statistics with per-column validity, not an
  overloaded zone or numeric sentinel. Counts remain valid integer zero for an
  explicitly requested empty zone; mean/min/max and similar statistics are
  invalid there.
- [ ] Provide deterministic iteration yielding immutable row records plus
  `to_records()`, `to_dict()`, `to_json()`, and `write_csv()`. Conversion must
  preserve large integer zone IDs exactly. **PARTIAL:** the serializers and
  immutable ``to_records()`` rows exist and preserve large IDs, but the result
  itself does not implement row iteration.
- [ ] Keep mergeable streaming accumulator state private and separate from the
  finalized `ZonalStatistics` result. **DEFERRED -- LARGE-RASTER PLAN:** only
  eager whole-raster accumulation exists.
- [x] Required statistics: count, valid count, invalid count, sum, mean, min,
  max, range, standard deviation, variance, median, and requested percentiles.
- [x] `zonal_raster(values, zones, statistic=...)` broadcasts one statistic
  back to valid zone cells.
- [x] Define zone IDs as integer or Boolean values. Reject floating zone IDs
  rather than truncating them.
- [x] Zone zero is ordinary unless explicitly configured as background.
- [x] Invalid value cells are excluded from statistics; invalid zone cells are
  not assigned to any zone.
- [ ] Implement bounded accumulation for count/sum/min/max/mean/variance.
  Median and percentile may use an explicit exact in-memory mode or documented
  approximate streaming mode; never silently switch algorithms.
  **DEFERRED -- LARGE-RASTER PLAN.**
- [x] Provide CSV/JSON-friendly conversion without requiring pandas.

### 3.5 Global reductions

- [x] `statistics(raster, ...)` returns count, invalid count, sum, mean, min,
  max, range, variance, and standard deviation with documented accumulator
  precision.
- [ ] `histogram(raster, *, bins, range=None)` supports explicit edges and
  bounded streaming. **PARTIAL:** explicit edges work eagerly; bounded
  streaming is deferred to the large-raster plan.
- [x] `unique_counts(raster, *, max_unique=...)` fails predictably when a
  safety bound is exceeded.
- [x] `percentile(raster, q, *, method="exact"|"approximate", ...)` makes
  memory/accuracy behavior explicit.
- [x] Reductions return Python/NumPy scalar or result dataclass values, not a
  one-cell georeferenced raster.
- [x] Empty-valid-domain behavior is defined per reduction and uses structured
  errors or explicit empty results rather than NumPy warnings alone.

### 3.6 Distance fields

Distance fields support hazard clearance and proximity screening without
introducing route policy.

- [x] `distance_to(seeds, *, metric, units, max_distance=None)` with Boolean
  seeds, canonical validity, and metrics `euclidean`, `taxicab`, and
  `chessboard` where scientifically meaningful.
- [x] `signed_distance(mask, ...)` defines Boolean `True` pixels as inside and
  leaves input-invalid pixels invalid by default. At a valid `True` pixel, the
  value is the positive center-to-center distance to the nearest valid `False`
  pixel; at a valid `False` pixel, it is the negative center-to-center distance
  to the nearest valid `True` pixel. The class interface therefore lies midway
  between opposite-class pixel centers and normally has no stored zero-valued
  center. Use this convention in eager, file-backed, CPU, and any later CUDA
  implementations; define a structured empty-opposite-class error for all-True
  or all-False inputs unless an explicit finite fallback is requested.
- [x] Units are `pixels` or physical CRS units. Physical Euclidean distance
  must honor anisotropic and rotated affine basis vectors.
- [x] In `0.2`, taxicab and chessboard distances are pixel-unit metrics only;
  reject requests to label them as physical distance on anisotropic or rotated
  grids.
- [x] Reject physical distance for geographic/angular grids in `0.2` unless an
  explicit, reviewed lunar body/geodesic model is supplied. Do not substitute
  Earth geodesics.
- [x] Valid seed pixels are the only seeds. By default, input-invalid pixels
  remain invalid in the output but do not bend or block straight-line distance
  measured at other pixels. An explicit `invalid_output="compute"` option may
  calculate values there. Barrier-aware distance is deferred with cost-distance
  and path planning.
- [x] Define deterministic behavior for no seeds, all seeds, seeds on invalid
  pixels, raster edges, and `max_distance` clipping.
- [x] Implement a validated CPU reference first. Add CUDA only after CPU tests
  and an independent reference comparison pass. *(CPU-only in 0.2.)*
- [ ] For file-backed Euclidean distance, select and document a genuinely
  bounded exact algorithm or explicitly label a tiled approximation and its
  error bound. Do not run `scipy.ndimage.distance_transform_edt` on a silently
  materialized regional raster. **DEFERRED -- LARGE-RASTER PLAN:** no
  file-backed distance operation is advertised.
- [x] Keep accumulated-cost distance, allocation, backlink rasters, and
  least-cost routes out of `0.2`; those cross into the path-planning design.

### 3.7 Alignment and resampling expressions

- [x] `ma.align(raster, to=..., ...) -> Raster` matches the existing eager
  alignment contract through the shared resampling core while preserving
  canonical validity.
- [x] `ma.resample_to(expression, grid, *, resampling, ...)` is an explicit
  expression node and may not be inserted automatically by another operation.
- [x] Resample the validity mask conservatively: nearest for categorical
  validity by default, with a documented coverage-threshold option for
  interpolating numeric data.
- [x] Distinguish categorical from continuous resampling and reject obviously
  unsafe combinations unless explicitly overridden.
- [ ] Add integration tests for differing CRS, shifted origins, partial
  coverage, nodata, rotated grids, and exact 64-bit nodata payloads.
  **PARTIAL:** differing CRS, shifts, partial coverage, explicit masks,
  rotations, and exact signed/unsigned 64-bit value payloads are covered.
  Extreme declared GeoTIFF nodata metadata remains limited by Rasterio/GDAL
  representability and needs a separately reviewed contract.

### 3.8 Temporal map algebra

Time is a named axis with UTC coordinates, not an extra spatial band. Do not
add an implicit time axis to `RasterExpression`. Introduce distinct
`TemporalRaster` and `TemporalRasterExpression` types so validation can require
both a common spatial grid and compatible time coordinates.

- [x] Define eager `TemporalRaster` with `values[time, y, x]`, UTC `times`, one
  spatial `GeoReference`, canonical `valid[time, y, x]`, units, and signal
  name. Provide explicit lossless adapters where possible to and from existing
  `TemporalCube` without changing the existing class.
- [x] Define immutable `TemporalRasterExpression` nodes for temporal sources,
  layer-wise local operations, explicit temporal alignment, and reductions.
  Do not let ordinary `ma.source()` accept a temporal series.
- [x] Add `ma.temporal_source(series)` for an open
  `TemporalGeoTiffSeries` or an explicitly opened series manifest. It reads
  manifest and layer metadata without retaining every layer open. Multi-band
  product GeoTIFFs require a separate adapter because their storage contract
  differs from `TemporalGeoTiffSeries`.
- [x] Any temporal-expression operand makes an operation temporal and lazy. A
  static `Raster` or `RasterExpression` may broadcast across time only after
  exact spatial-grid validation; a scalar broadcasts across time and space.
- [x] Combining two temporal operands requires exactly equal ordered UTC
  coordinates by default. Temporal resampling, nearest selection, and
  interpolation are separate explicit operations with tolerance, edge, and
  validity rules. *(time matching validated; resampling/interpolation deferred.)*
- [ ] Classify temporal nodes as **layer-wise** or **reducing**. Layer-wise
  nodes retain `(time, y, x)` and are processed in bounded spatial-window and
  time batches. Reducing nodes consume time batches and return a spatial
  `Raster` or `RasterExpression` while retaining bounded accumulator state for
  the current output window.
  **PARTIAL:** node classification exists; general bounded spatial-window/time-
  batch execution is deferred to the large-raster plan.
- [ ] Choose time-batch size from an explicit memory budget and record it in
  the execution plan. A series with thousands of layers must not imply one
  resident array or one open dataset per timestamp.
  **PARTIAL:** series handles are opened and closed per execution, but the
  memory-budget scheduler is deferred to the large-raster plan.
- [ ] Reuse bounded dataset caches and the existing streaming reducer
  infrastructure. State whether execution is layer-major or spatial-window
  major for each source/output layout and report the expected read pattern in
  `ma.plan()`. **PARTIAL:** streaming reducers are reused; the plan does not
  report the read pattern. Further scheduling work is deferred.
- [ ] Reuse existing mean/min/max/std semantics and add count, sum, variance,
  percentile, `any`, `all`, threshold duration, and exceedance count only with
  documented sample, interval, nodata, and all-invalid behavior.
  **PARTIAL:** mean/min/max/std/sum/count are implemented with documented
  semantics; variance, percentile, any/all, duration, and exceedance count are
  deferred.
- [ ] Make a temporal reduction an ordinary composable spatial expression. For
  example, `ma.temporal_mean(sun_series) >= 0.60` may combine with a static
  slope expression and be written window by window without first creating a
  complete mean GeoTIFF. **PARTIAL:** reductions are composable expressions,
  but the bounded spatial writer rejects temporal nodes. Callers must
  explicitly ``compute()`` the reduction before writing until
  the large-raster plan is resumed.
- [ ] File-backed temporal mapping initially writes the existing timestamped
  GeoTIFF-series format through `TemporalGeoTiffSeriesWriter`. Generic
  multi-band BigTIFF expression output is deferred until its mask, timestamp,
  band-count, and resume contracts are reviewed.
  **DEFERRED -- LARGE-RASTER PLAN:** series-format expression mapping is not
  integrated.
- [ ] `compute()` is the only operation that may explicitly materialize a
  complete temporal result. Preflight its estimated bytes and require an
  explicit override above a documented safety threshold.
  **PARTIAL:** ``compute_temporal()`` is explicit, but byte-estimate safety
  preflight and override controls are not implemented.
- [x] Do not generalize specialized mission-duration or safe-haven reducers
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

- [x] `_as_raster_operand(value, *, argument)` accepts only `Raster` or a real
  scalar in eager operations.
- [x] `_as_expression_operand(value, *, argument, grid_hint)` accepts only
  `RasterExpression`, `Raster`, or a real scalar.
- [ ] `_require_raster_shape(values, georef)` validates exactly two dimensions.
  **PARTIAL:** ``Raster`` validates two-dimensional shape, but there is no
  shared helper with this contract.
- [x] `_require_common_grid(operands)` reports every differing grid field in a
  structured error.
- [x] `_normalize_scalar(value)` rejects arrays disguised as scalars, complex
  values, and unsupported scalar dtypes.
- [ ] `_normalize_footprint(size, footprint)` validates odd dimensions and
  calculates halo on all four sides.
  **PARTIAL:** focal validation exists, but halo calculation is not centralized
  for a window planner.
- [x] `_normalize_numeric_errors`, `_normalize_overflow`,
  `_normalize_edge_mode`, and `_normalize_valid_neighbor_policy` return typed
  literals and stable error codes.
  Numeric-error, arithmetic-overflow, cast-overflow, edge, and neighbor
  policies use typed literals and stable structured codes.
- [x] `_validate_output_encoding(dtype, nodata, invalid_value)` shares GeoTIFF
  representability checks rather than duplicating them. The canonical raster
  validator is used by eager conversion, GeoTIFF metadata validation, and
  writer preflight; public boundaries translate its failure into their domain
  exception without changing the exact-representation rule.

### 4.2 Validity helpers

- [x] `_valid_from_nodata(values, nodata)` handles exact integer nodata, finite
  floating nodata, and NaN nodata without lossy coercion. It validates and
  normalizes through the shared encoding helper before comparing payloads.
- [x] `_combine_validity_strict(*rasters)` intersects raster validity.
- [x] `_where_validity(condition, x, y)` implements selected-branch validity.
- [ ] `_coalesce_values_and_validity(...)` performs ordinary per-pixel
  selection after all operands needed for the current window are available.
  The first implementation reads every coalesce operand window; correctness
  must not depend on static validity analysis.
  **PARTIAL:** eager, expression, and supported window execution select exact
  first-valid values without FP64 intermediates, but the logic remains in the
  eager semantic dispatcher rather than a separately named shared helper.
- [x] Defer coalesce read short-circuiting. A later optional runtime
  optimization may stop requesting later operand windows only when values
  already computed for that output window leave no unresolved invalid pixels.
  Failure to prove that condition always falls back to reading all operands;
  do not add general SSA-style validity proof to the `0.2` planner.
- [x] `_apply_numeric_domain(valid, values, policy, operation)` handles new
  non-finite/domain errors consistently.
  The common helper implements invalid/keep/structured-raise consistently for
  applicable pointwise arithmetic and math operations. Selection,
  classification, and normalization reducers retain dedicated contracts.
- [x] `_fill_invalid_exact(values, valid, fill, dtype)` validates exact
  representation and never mutates an input array. ``Raster.filled()``, eager
  ``fill_invalid``, ``to_existing``, and windowed output all use it.
- [ ] `_read_rasterio_validity(dataset, band, window)` combines the band mask,
  dataset mask, nodata, and alpha semantics with dedicated tests and returns
  both canonical validity and a normalized provenance description. Inspect
  Rasterio mask flags so a synthesized all-valid/nodata mask is not mislabeled
  as an explicitly stored mask.
  **PARTIAL:** whole-raster reads cover the validity sources and provenance;
  the specified window helper and all flag combinations are not complete.

### 4.3 Dtype helpers

- [ ] `_result_dtype(operation, operand_dtypes, scalars, parameters)` is the
  sole dtype-inference entry point.
- [ ] `_accumulator_dtype(operation, source_dtype)` defines streaming
  accumulator precision. **PARTIAL:** the shared rule is authoritative for
  eager and layer-streamed temporal reducers and for eager focal
  sum/mean/std/count/min/max construction and execution. Focal range and
  median now use reviewed FP32/FP64 output rules, and convolution preserves
  FP32 source/result execution while using FP64 for integer inputs. These
  remain explicit operation-specific paths, so the helper is not yet the sole
  accumulator entry point for every reduction family.
- [x] `_checked_integer_kernel(...)` detects overflow without allocating
  unbounded temporary arrays and without converting integer operands or
  results to floating point. Checks must be expressible as bounded native
  integer comparisons suitable for later CPU kernels and bounded accelerator
  implementations. Native integer checks cover add/subtract/multiply/
  floor-divide/remainder/power and unary negate/absolute/square. Exact 64-bit
  cases are CPU reference/correctness paths and must not become a CUDA hot-path
  dependency.
- [ ] `_cast_values_and_fill(...)` applies casting and output encoding in a
  deterministic order.
- [ ] Add table-driven tests covering every supported dtype pair and boundary
  scalar, including `uint64` values beyond float exact range. Include tests
  proving that ``float32`` operations retain ``float32`` intermediates/results
  unless their documented contract explicitly promotes them.

**PARTIAL:** centralized dtype, accumulator, checked-integer, and cast helpers
exist, but they are not yet the sole path and the full operation-by-dtype
matrix is not proven. Complete supported-dtype pair matrices now cover ordinary
addition inference and representable unsafe casts; focused boundary tests cover
exact ``int64``/``uint64`` values beyond ``2**53``, power, cast overflow, and
FP32 preservation. Selection coverage additionally proves exact Python integer
fallbacks, exact ``uint64`` coalescing through windowed GeoTIFF output,
incompatible signed/unsigned 64-bit rejection, and eager/expression unit
parity. Exact encoding coverage proves shared nodata/fill rejection and
normalization across raster ingestion, eager conversion, GeoTIFF validation,
and windowed output, including ``uint64`` values beyond ``2**53``. A combined
``_cast_values_and_fill`` sole path remains incomplete. Reclassification now
uses ``result_dtype`` for exact output classes and defaults, avoids gratuitous
``int64`` for small Python integer classes, preserves typed FP32, includes the
complete input dtype for ``default="preserve"``, and proves exact eager,
expression, and multi-window ``uint64`` behavior.

Normalization now uses ``result_dtype`` for eager/expression inference and
performs ordinary FP32 and Boolean/8/16-bit work in FP32; explicit FP64
statistics, out-of-FP32-range statistics, FP64 sources, and 32/64-bit integer
sources select the documented CPU/interchange FP64 path. Int32 remains there
until a native-32-bit centering algorithm proves large-adjacent-integer
correctness. Temporal mean/std/sum now use the shared accumulator rule in both
eager and layer-streamed execution, preserve FP32, and sum signed/unsigned
integers through exact fixed-width CPU accumulators without an FP64
intermediary. Eager focal sum/mean/std/count/min/max now share that construction
and execution rule as well: FP32 stays FP32; fixed-width integer sums never use
FP64; integer mean/std use exact-integer CPU moments; count uses bounded native-
width working state with an int64 public result; and min/max preserve exact
source values. Focal range avoids source-dtype subtraction overflow, focal
median orders integer samples before conversion, and convolution preserves an
FP32 source/result path. Global and zonal integer summaries likewise avoid
pre-conversion to FP64 for sums, extrema, ranges, and centered moments. The
general accumulator-centralization checkbox remains partial because these
families retain operation-specific public result types.

Unit-bearing power now uses one shared unit helper across eager and expression
construction. The declared derived unit is replayed by compute/windowed
execution and participates in identity. Raster exponents remain supported only
for bases carrying no unit metadata and cannot declare one fixed output unit.

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

- [x] Registration occurs at import from static library code only; users
  cannot register arbitrary kernels in `0.2`.
- [x] Registry import must not initialize CUDA, open datasets, or import SPICE.
- [x] Reject duplicate identifiers and invalid versions at test time.
  *(Registry construction rejects both, with dedicated tests.)*
- [ ] Ensure every public operation has an operation spec, documentation,
  validity test, dtype test, and tests for every execution mode its descriptor
  claims to support.
  **PARTIAL:** the registry covers the current public catalog and public
  parameters are audited against signatures. Execution claims are separated
  into eager, expression-compute, direct windowed, composed windowed, and
  temporal streaming modes and direct window claims are audited against the
  executor. A complete per-operation documentation and behavioral-test matrix
  remains unfinished.
- [x] Generate an internal coverage report from the registry so eager-only,
  windowed, multi-pass, and unsupported modes are explicit rather than
  accidental.
  *(Machine-readable list/describe output exposes and filters each distinct
  execution mode; tests audit direct window claims against the executor and
  public parameter claims against callable signatures.)*
- [ ] Generate machine-readable public operation descriptions from the same
  descriptors. Parameter documentation, defaults, execution support, and
  validity rules must be testable against actual signatures.
  **PARTIAL:** discovery is public and JSON-serializable; parameter types,
  defaults, enumerated choices, signatures, scientific rules, and execution
  modes are linked and tested. Numeric ranges, canonical examples, and
  generated prose documentation remain incomplete.

### 4.5 Expression planner

Completed planner behavior below remains supported. Unchecked work specifically
about window selection, fusion, general halos, or resource scaling is
**DEFERRED -- LARGE-RASTER PLAN**; validation limits and `explain()` quality
remain core work.

- [x] Topologically validate the graph and detect cycles defensively.
- [ ] Enforce documented limits on graph nodes, depth, source count,
  normalized-parameter bytes, footprint dimensions, and requested output
  bands. Limits prevent accidental or generated expressions from exhausting
  planning resources and fail before source execution or output staging.
  **PARTIAL:** node, depth, and source limits are enforced and tested;
  normalized-parameter bytes, footprint dimensions, and output-band limits are
  not.
- [ ] Infer one output grid, dtype, units, validity behavior, and maximum halo
  before creating output staging.
  **PARTIAL:** grid, dtype, units, and cumulative one-pixel terrain halos are
  validated before staging; general validity inference and arbitrary focal
  footprint halos are deferred.
- [ ] Fuse consecutive local operations into one window task to avoid
  unnecessary full-window writes; correctness comes before aggressive fusion.
  **DEFERRED -- LARGE-RASTER PLAN:** the execution graph is processed per-window
  with no full-raster intermediate materialization; explicit fusion is absent.
- [x] Do not fuse across global reductions, resampling, distance transforms, or
  operations with incompatible halos.
  *(unsupported operations are rejected during planning.)*
- [x] Reuse a source window within a task when multiple nodes request it.
  *(SourceWindowCache with LRU eviction; same ``(node_id, window_idx)`` key.)*
- [x] Bound source dataset handles, decoded windows, and output queues.
  *(explicit ``max_datasets`` and ``max_windows`` bounds; datasets and caches
  close on success and failure; output is synchronous with no queue.)*
- [ ] Select window sizes from output block geometry with a conservative
  default of 128 by 128; record the choice in progress metadata but not
  scientific identity.
  **DEFERRED -- LARGE-RASTER PLAN:** dimensions are configurable with a
  128-by-128 default, but are not selected from block geometry.
- [ ] Calculate halos in source pixel coordinates and crop exactly once.
  **DEFERRED -- LARGE-RASTER PLAN:** one-pixel terrain halos are implemented and
  crop exactly once; general footprint-derived asymmetric halos are absent.
- [x] Emit a readable plan description for diagnostics and tests.
  *(``plan()`` now reports window layout, source count, estimated per-window
  memory, and other planner metadata.)*
- [x] Implement read-only `ma.plan()` and `ma.explain()` on top of normalized
  graph and planner data. Neither function may execute numerical kernels,
  create staging, or write output.
  *(Both are read-only, use normalized expression/planner and registry data,
  and propagate structured validation failures before output creation.)*

### 4.6 File-backed sources and output

This section records the completed and supported bounded subset. Expansion to
additional operation families is owned by the deferred large-raster plan.

- [x] `ma.source(path, *, band=1, units=None, identity="stat"|"sha256")` reads
  metadata only and returns an expression without retaining an open dataset.
- [ ] Validate source existence, driver, band, dtype, grid, nodata, and mask
  before output modification.
  **PARTIAL:** core metadata is preflighted; complete mask-flag validation is
  deferred to reads.
- [x] Open datasets lazily during execution and close them deterministically.
- [x] ``ma.write()`` evaluates expressions in bounded windows; peak working
  memory depends on active sources, graph complexity, and window size -- not
  total raster area. Output is synchronous and coordinate rasters are generated only for
  the requested window.
- [x] Source window reads are cached (LRU) and reused within one task.
- [x] Dataset handles and caches have explicit bounds (``max_datasets``,
  ``max_windows``) and are closed after success and failure.
- [x] Supported operations reuse the eager semantic dispatcher, preserving
  grid, validity, dtype, units, numeric policy, and deterministic invalid-fill
  behavior. File-backed normalization requires caller-supplied statistics;
  measured statistics fail during planning rather than becoming tile-local.
- [x] Unsupported focal, global, zonal, distance, and temporal nodes are
  rejected during planning before output staging or pixel execution. Reviewed
  terrain and resampling nodes are accepted by the windowed planner.
- [x] Existing atomic overwrite guarantees and restart-manifest identity
  checks are preserved.
- [x] Extend or reuse the existing durable product-storage patterns for
  staging, overwrite protection, cancellation, progress, journaling, and
  atomic publication.
- [x] A completed-window journal is authoritative. Restart recomputes an
  unjournaled window even when its TIFF block contains plausible data.
- [x] Bind restart metadata to expression JSON, source identities, grid,
  dtype, units, validity/nodata encoding, window layout, and algorithm
  versions.
  *(The journal identity binds the expression scientific identity, which
  includes normalized nodes, source identities, units, and operation versions,
  plus the complete output grid, dtype/fill, layout, checkpoint interval,
  validity encoding, and enforced GeoTIFF options. The completed-output
  manifest remains smaller.)*
- [ ] Store scientific, restart, and execution-cache identities separately;
  never invalidate scientific provenance merely because a worker count or JIT
  cache changed.
  **PARTIAL:** scientific and restart identities exist; a separate execution-
  cache identity does not.
- [x] Never delete a previous complete output until its staged replacement has
  closed and validated successfully.
- [x] Define safe `start_fresh` cleanup using exact resolved staging paths.
  *(Only exact paths derived from the resolved output are removed; the caller
  explicitly opts into removal of the completed output and manifest.)*
- [x] Support single-band GeoTIFF output in the first slice. Add generic
  multiband expression output only after its band metadata contract is defined.

### 4.7 Backends

- [x] Implement and validate eager NumPy/SciPy CPU behavior first.
- [ ] Implement bounded windowed CPU behavior second and compare it exactly or
  within documented tolerance to eager results.
  **PARTIAL:** local, coordinate, terrain, and reviewed resampling operations
  have validated windowed execution; general focal/global/zonal/distance and
  temporal windowed execution is **DEFERRED -- LARGE-RASTER PLAN**.
- [x] Use Numba CPU only where benchmarks show a useful improvement and cache
  behavior is acceptable in installed wheels.
  **NOT APPLICABLE:** no Numba map-algebra kernels were selected.
- [x] Do not require CUDA for core map algebra.
- [x] Add CUDA per operation only after a CPU reference, backend-independent
  semantics, correctness comparison, memory bound, and realistic benchmark
  exist.
  **NOT APPLICABLE:** no CUDA map-algebra operations are implemented.
- [x] Follow existing backend semantics: CPU never probes CUDA; explicit CUDA
  never falls back; auto may fall back only for capability/availability, not
  after a CUDA execution failure.
  **NOT APPLICABLE:** map algebra advertises no backend selection.
- [x] Do not advertise `backend=` on operations with no supported alternative
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
  **PARTIAL:** structured details are present in many validation/storage paths,
  but coverage is not consistent across every operation and there are no
  window coordinates before the ``0.3`` planner.
- [ ] Never expose a raw Rasterio, SciPy, NumPy, Numba, or CUDA exception as the
  only public diagnostic.
  **PARTIAL:** primary public paths translate dependency failures, but some
  internal/publicly reachable policy and coordinate paths can still emit raw
  ``ValueError`` or ``TypeError`` diagnostics.

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
  **PARTIAL:** this reconciliation records the implemented subset and explicit
  ``0.3`` deferrals; the full broad inventory is not frozen as ``0.2`` scope.

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
- [x] Implement reclassification and stack combination.
  Reclassification, digitization, one-hot, pairwise min/max, and compositional
  variadic stack helpers are implemented.
- [ ] Add property-based-style randomized tests using deterministic seeds;
  compare valid cells with direct NumPy reference calculations.
  **PARTIAL:** deterministic analytic coverage is extensive; the requested
  randomized matrix is incomplete.
- [ ] Test every invalidity, dtype, overflow, unit, scalar, and grid branch.
  **PARTIAL:** major branches and focused exact ``int64``/``uint64`` values
  beyond ``2**53`` are covered. Supported-dtype pair matrices cover addition
  inference and representable unsafe casts, but the exhaustive Section 7
  operation/policy cross-product remains open.
- [ ] Convert the landing-site screening example to a new additional example,
  retaining the old array-oriented example as compatibility evidence.
  **PARTIAL:** the additional map-algebra screening example exists; explicit
  side-by-side compatibility evidence is incomplete.

Acceptance evidence:

- [x] A complete terrain-plus-lighting candidate expression needs no manual
  mask bookkeeping after input construction.
- [x] Mismatched georeferenced rasters fail before numerical calculation.
- [x] Results match reference NumPy values and the documented validity rules.

### Phase C: Expressions and bounded local execution

- [x] Implement immutable expression nodes and the sealed operation registry.
  *(both expression nodes and the sealed static registry are implemented)*
- [x] Implement GeoTIFF, in-memory, scalar, and coordinate sources.
  *(coordinate sources materialize only the requested window during
  ``ma.write()``; ``ma.compute()`` remains explicitly whole-raster)*
- [x] Implement expression operator overloads and stable JSON identity.
- [ ] Implement canonical typed serialization plus distinct scientific,
  restart, and execution-cache identities with golden fixtures.
  **PARTIAL:** scientific and restart identities are implemented; an execution-
  cache identity and golden compatibility fixtures remain incomplete.
- [x] Implement ``describe()``, ``ma.explain()``, ``ma.plan()``, and
  machine-readable operation introspection without executing kernels or
  writing files.
  *(These read-only interfaces now use registry, normalized expression, and
  planner metadata; public signatures, distinct execution claims, scientific
  choices, identities, output encoding, and preflight behavior have focused
  public tests.)*
- [x] Implement the planner, window enumeration, source cache, cancellation
  checks, and progress events.
- [ ] Fuse compatible consecutive local operations.
  **DEFERRED -- LARGE-RASTER PLAN:** current nodes execute within bounded
  window tasks but are not explicitly fused into a single kernel/pass.
- [x] Implement window kernels for every Phase B local operation.
  *(all local binary, unary, and special operations dispatch through
  ``_windowed.py``; parity tested against eager results.)*
- [x] Test many window/block sizes, including outputs smaller than one block
  and dimensions not divisible by 128.
  *(``test_planner_windows.py`` covers smaller-than-block, non-divisible,
  partial edges, and multiple window sizes.)
- [ ] Measure peak memory against increasing raster dimensions and prove it is
  bounded by window/graph complexity rather than total raster area.
  **PARTIAL:** the planner estimate is invariant with raster area and the code
  contains no area-sized window list or output mask, but an estimate is not an
  empirical peak-memory measurement.

The remaining Phase C fusion and resource measurement work is
**DEFERRED -- LARGE-RASTER PLAN**.

Acceptance evidence:

- [x] Eager and windowed outputs have identical payload and validity for
  integer/Boolean operations and documented tolerances for floating operations.
  *(parity tests in ``test_planner_windows.py``.)*
- [x] Source datasets and caches close after success, failure, and cancellation.
- [x] ``ma.write()`` no longer materializes the complete expression; it reads
  bounded source windows and processes per-window. Repeated source windows
  reuse cached data (``test_repeated_window_read_cached``).

### Phase D: Durable expression output

- [x] Implement output preflight, staged GeoTIFF creation, deterministic
  invalid payload, atomic publication, and GDAL mask writing (via
  ``write_mask()`` at dataset creation time).
- [x] Implement journal-based resume with per-window checkpoint journal.
  Journal identity binds to expression identity, output dtype, invalid fill,
  complete grid, window layout, checkpoint interval, validity encoding, and
  GeoTIFF options. The compact journal records the contiguous row-major
  completed prefix, and a matching staged TIFF is structurally validated
  before it is reused.
- [x] Bind restarts to expression scientific identity, output dtype,
  invalid fill, and grid dimensions.
- [x] Add injected-failure tests covering progress callback failures,
  cancellation before and during execution, overwrite preservation on
  cancellation, and journal incompatibility.
- [ ] Add cancellation/resume tests and concurrent-output conflict tests.
  **PARTIAL:** cancellation/resume is covered; concurrent writers targeting the
  same output remain unsupported. A locking contract is deferred to the
  large-raster plan.
- [x] Confirm failed overwrite preserves the previous complete output.

Acceptance evidence:

- [x] A killed multi-window operation resumes without trusting unjournaled
  blocks. The checkpoint journal records a completed prefix; resume skips that
  prefix and recomputes uncommitted windows. Fault injection covers callback,
  value-before-mask, journal-update, and paired-publication failures.

### Phase E: Focal and morphology operations

- [x] Implement footprint/halo/edge/valid-neighbor contracts.
  *(five edge modes, three valid-neighbor policies, ``cval`` parameter)*
- [x] Implement the required focal statistics and convolution.
  *(sum, mean, min, max, range, std with ddof, count, median, convolve)*
- [x] Reconcile eager focal sum/mean/std/count/min/max/range/median/convolution accumulator and output
  dtypes across public calls, expression inference, and whole-raster
  ``compute()``. FP32 work remains FP32; exact integer sums and large-adjacent
  integer mean/std use documented CPU correctness paths. Range subtraction is
  exact before floating conversion, median orders integer samples before
  conversion, and convolution preserves its inferred FP32 or FP64 path.
- [x] Implement shared morphology and region adapters.
  Dilate, erode, opening, closing, and majority exist; eager Boolean-Raster
  adapters delegate region labeling, filtering, sizing, and borders to the
  established array algorithms.
- [x] Implement or explicitly defer windowed terrain nodes for slope, aspect,
  and hillshade based on whole-array parity tests.
- [ ] Compare eager and tiled halo results across internal window boundaries
  for general focal statistics and morphology.
  *(terrain operations have validated seamless tiled execution; general focal
  execution is deferred to the large-raster plan.)*
- [ ] Test rotated/anisotropic grids and document which focal operations are
  pixel-neighborhood rather than physical-radius operations.
  **PARTIAL:** eager grid cases are covered; the complete documentation and
  eager scientific contract remains core work, while tiled evidence is
  deferred to the large-raster plan.
- [ ] Benchmark SciPy, NumPy sliding windows, and Numba candidates before
  choosing optimized kernels. *(SciPy selected as baseline; NumPy sliding
  windows and Numba candidates not yet benchmarked)*

Acceptance evidence:

- [ ] No seams occur at tile boundaries, and edge/invalid behavior matches an
  independent whole-array reference.
  **PARTIAL:** terrain nodes have seamless tiled parity; general focal and
  morphology tiled execution is deferred to the large-raster plan.

### Phase F: Global and zonal reductions

- [x] Implement global statistics, histogram, unique counts, and exact
  percentile. *(in-memory reductions; streaming accumulators deferred)*
- [x] Implement zonal tabular statistics and broadcast zonal rasters.
- [x] Implement the finalized ``ZonalStatistics`` row ordering (sorted
  zone IDs), per-column validity, integer-typed counts, float64 statistics,
  ``include_zone_ids``, zone-ID preservation, and serializers
  (``to_dict``, ``to_json``, ``to_records``, ``write_csv``).
- [x] Define deterministic zone ordering and JSON/CSV conversion.
- [x] Valid statistics enumerated; invalid statistics raise structured error.
- [x] Test zone-ID types (int, uint64, bool), empty zones, all-invalid zones,
  and zonal percentiles (p25, p75, p90).
- [ ] Test window-order independence where floating-point tolerances allow it.
  **DEFERRED -- LARGE-RASTER PLAN.**

Acceptance evidence:

- [ ] Streaming and eager results agree within a stated tolerance without
  memory proportional to raster area, except explicitly selected exact
  percentile modes.
  **DEFERRED -- LARGE-RASTER PLAN.**

### Phase G: Distance fields

- [x] Freeze distance metrics (``euclidean``, ``taxicab``, ``chessboard``),
  units (``pixels``, ``physical``), invalid-output behavior, and affine
  handling. Physical Euclidean distance uses both affine basis vectors;
  physical taxicab and chessboard distance remain unsupported.
- [x] Implement small CPU reference algorithms (scipy EDT for Euclidean,
  2-pass for taxicab/chessboard) and independent analytic test cases.
- [x] Implement eager distance fields: ``distance_to()`` and
  ``signed_distance()``.
- [ ] Evaluate exact bounded file-backed algorithms.
  **DEFERRED -- LARGE-RASTER PLAN.**
- [x] Add physical-distance tests for square, anisotropic, rotated, and skewed
  projected grids, including a non-metre projected CRS.
- [x] Add explicit rejection for geographic CRS (via pyproj ``is_geographic``
  check) and for taxicab/chessboard with physical units.
- [ ] Benchmark representative hazard masks.
  **DEFERRED -- LARGE-RASTER PLAN.**

Acceptance evidence:

- [ ] Results match SciPy or analytic references where their assumptions match,
  and memory/temporary-disk bounds are recorded.
  **PARTIAL:** eager correctness matches references; bounded-memory and
  temporary-disk evidence are deferred to the large-raster plan.

### Phase H: Temporal adapters

- [x] Implement `TemporalRaster`, explicit `TemporalCube` adapters, and
  `TemporalRasterExpression` without changing existing temporal classes.
- [x] Implement explicit layer-wise local expression nodes and static spatial
  raster broadcasting.
- [ ] Implement `ma.temporal_source()` and bounded spatial-window/time-batch
  mapping over `TemporalGeoTiffSeries`.
  **PARTIAL:** the source and streaming reducers exist; general layer-wise
  spatial-window/time-batch mapping is deferred to the large-raster plan.
- [x] Add time-coordinate equality and explicit alignment validation.
- [x] Make approved temporal reducers produce composable spatial expressions
  using existing streaming accumulators where semantics match.
- [x] Add documented sample/interval, validity, empty-domain, and output-unit
  semantics for every reducer.
- [ ] Add approximately 3,000-layer execution tests proving bounded dataset
  handles, bounded resident batches, and accurate planning estimates.
  *(126 tests cover construction, adapters, expressions, eager compute,
  scalar-left ops, grid rejection, time contract, reducer semantics,
  file-backed execution, and `compute()` integration; 3,000-layer streaming
  not yet exercised. The 3,000-layer case is in-memory; file-backed stress
  coverage currently uses 200 layers. Further scaling evidence is deferred to
  the large-raster plan.)*
- [x] Ensure no temporal helper constructs a full file-backed cube unless the
  caller explicitly requests materialization.

Acceptance evidence:

- [x] Layer-wise eager and streamed results match and preserve UTC metadata,
  masks, grids, signal names, and units.

### Phase I: Documentation, examples, and release gate

- [x] Add a map-algebra chapter to `docs/USER_GUIDE.md` covering eager versus
  file-backed workflows, grids, validity, dtypes, units, and lunar constraints.
- [x] Update `docs/ARCHITECTURE.md` with the accepted model, execution planner,
  and storage flow.
- [ ] Add API reference tables for every operation and its validity/dtype/unit
  behavior. *(covered by USER_GUIDE.md per-family summaries; per-operation
  reference table deferred.)*
- [ ] Publish the machine-readable operation catalog, canonical expression
  schema, identity distinctions, and examples of `explain()` and `plan()`.
  **PARTIAL:** the public catalog, canonical expression JSON, identity, and
  explanation/planning usage are documented in the user guide. A standalone
  generated operation reference and identity-distinction reference remain
  incomplete.
- [ ] Add runnable examples for terrain-lighting screening, weighted scoring,
  hazard clearance, focal cleanup, zonal candidate summaries, and temporal
  threshold summaries. Large file-backed examples are owned by the deferred
  large-raster plan.
  **PARTIAL:** terrain-lighting screening, weighted scoring, focal cleanup,
  temporal reduction/composition, and basic hazard-distance output exist in
  examples 22, 25, 27, and 31. A dedicated zonal candidate summary and an
  explicit user-selected hazard-clearance workflow remain incomplete in the
  core plan.
- [x] Add progressive eager introductions for raster/local algebra; canonical
  validity, ``where``, and ``coalesce``; grids and explicit alignment; and
  units, dtypes, overflow, casting, and numerical policies.
  *(Runnable public-API examples 18--21 use deterministic synthetic lunar
  rasters and execute in fresh-process tests.)*
- [x] Gate the example-facing public surface with fresh-process tests for valid
  zero versus invalid GeoTIFF pixels, Boolean storage and dataset masks, grid
  and unit mismatch, exact ``uint64`` boundaries, non-finite and all-invalid
  inputs, eager/expression parity, read-only explanation/planning, and
  pre-output rejection of unsupported file-backed operations.
- [x] Use synthetic lunar grids and downloadable lunar products where needed;
  no example may depend on an unmentioned Earth dataset.
- [ ] Include a QGIS inspection example proving valid zero values remain visible
  and invalid pixels are transparent through the dataset mask.
  **PARTIAL:** ``09_qgis_vrt.py`` covers VRT inspection, not the specified
  map-algebra dataset-mask workflow.
- [ ] Add an "assistant proposes, human reviews, library validates" example in
  which an expression is explained and dry-run before execution. Keep tool
  authorization in the example application, not Lunarscout.
  **PARTIAL:** registry-backed explanation and complete read-only planning now
  support the workflow; the runnable example itself remains to be added.
- [ ] Record CPU correctness benchmarks for the eager/core surface.
  **PARTIAL:** correctness tests exist; a concise retained benchmark record is
  absent. Bounded-memory benchmarks are deferred to the large-raster plan.
- [x] Build wheel and sdist, inspect contents, run Twine checks, and test the
  installed artifacts without the checkout on `PYTHONPATH`.
- [x] Run the complete ordinary CPU suite with:

  ```bash
  .venv/bin/python -m pytest -q
  ```

- [x] Run any implemented CUDA comparisons only with
  `LUNARSCOUT_REQUIRE_NUMBA_CUDA=1` on a visible supported NVIDIA device.
  **NOT APPLICABLE:** the map-algebra surface is CPU-only; no CUDA comparison
  is implemented.
- [ ] Publish and independently install a `0.2.0rc1` TestPyPI candidate before
  describing the map-algebra API as accepted.
  **SKIPPED BY DECISION:** the project has no external testers yet and has
  already validated the local artifact/install workflow in an earlier release.
  TestPyPI publication is not a ``0.2`` release gate; publishing resumes when
  a later milestone is ready for the real PyPI.

## 7. Test matrix

Every operation family must cover the following relevant dimensions. Use small
analytic arrays for core semantics. Large generated rasters, window-order
matrices, cancellation/resume matrices, and memory-scaling evidence are owned
by the deferred large-raster plan.

**PARTIAL:** eager tests cover much of this matrix, but no row is checked until
the accepted core operations cover its relevant cross-product. Identity remains
core work; large-raster execution and resource dimensions are deferred.

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

All large-raster performance, memory-scaling, cache/queue bound, and
file-backed throughput requirements have moved to
`docs/map-algebra-large-raster-plan.md` and are **DEFERRED by project
decision**. Existing ordinary correctness tests do not constitute empirical
resource-scaling evidence.

Core-plan performance work is limited to avoiding obvious eager regressions and
documenting notebook-size guidance where evidence already exists. No new
large-raster benchmark gate is required while the separate plan is deferred.

## 9. Documentation required for each public operation

No operation is complete until its docstring and user documentation state:

**PARTIAL:** family-level guidance exists, but the complete per-operation
contract below is not yet present or linked to registry metadata.

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

- [x] No further large-raster execution expansion in the active core plan.
  Windowed focal kernels, cross-window reconciliation, streaming reducers,
  bounded distance/temporal execution, fusion, and scaling evidence are tracked
  only in `docs/map-algebra-large-raster-plan.md` and are deferred by decision.
- [x] No automatic reprojection or grid selection during algebra.
- [x] No sentinel `GeoReference` or coordinate-free `Raster`; use NumPy for
  non-spatial arrays.
- [x] No arbitrary Python callbacks in serializable/file-backed expressions.
- [x] No string expression parser, SQL syntax, or remote execution contract.
- [x] No implicit unit conversion or dimensional-analysis framework.
- [x] No pandas/xarray/Dask/CuPy dependency in the base public contract. These
  may receive adapters after the NumPy/Rasterio contract is stable.
- [x] No vector GIS overlay or rasterization beyond separately reviewed helper
  APIs.
- [x] No Earth-only environmental, hydrologic, road-network, land-cover, or
  weather operations.
- [x] No assumption that a lunar projected CRS has meters unless CRS metadata
  proves it or the caller supplies an explicit unit contract.
- [x] No geodesic physical distance on angular grids without an explicit body
  model.
- [x] No cost-distance, route extraction, rover policy, energy model, thermal
  model, or path optimizer in `0.2`.
- [x] No silent full-raster materialization in a file-backed operation.
  ``ma.write()`` now evaluates expressions in bounded windows; source reads,
  coordinate generation, intermediate computation, and output writing are all
  window-bounded. ``ma.compute()`` remains the explicit whole-raster path.
- [x] No CUDA-only core algebra operation unless separately justified with the
  same explicit exception used for horizon generation.

## 11. Final acceptance definition

The broad map-algebra milestone is complete only when all of the following are
checked:

- [ ] The eager API supports the accepted local, focal, zonal, global, and
  distance inventory with consistent grids, validity, dtype, and units.
  **PARTIAL:** the implemented eager subset is well tested, but the full
  accepted inventory and every policy branch are not complete.
- [x] The currently advertised file-backed inventory executes with bounded
  windows and durable, resumable, atomic output. Expansion of that inventory is
  not a core-plan acceptance gate and is deferred to the large-raster plan.
- [x] The currently advertised eager/file-backed overlap has parity evidence
  for local, coordinate, terrain, and reviewed resampling operations. Future
  operation families are deferred to the large-raster plan.
- [x] Dataset masks survive read, calculation, and write without conflating
  valid zero with invalid data.
- [x] Lunar projected, anisotropic, and rotated grid cases pass; unsafe
  Earth-specific or body-ambiguous assumptions are absent or rejected.
- [x] Existing terrain, temporal, region, horizon, lighting, and scenario APIs
  remain compatible and their tests pass.
- [ ] Documentation and runnable examples cover the accepted notebook-sized
  core workflows.
  **PARTIAL:** the four introductory eager workflows and several analysis
  examples are runnable and tested, but the generated operation reference and
  remaining Phase I analysis examples are incomplete.
- [ ] Operation discovery, expression explanation, dry-run planning, canonical
  provenance, and repair-oriented structured errors are sufficient for a
  future assisting model to propose an auditable calculation without granting
  it arbitrary execution inside Lunarscout.
  **PARTIAL:** the foundations exist, but planning, identity separation,
  registry metadata, and error coverage remain incomplete.
- [x] Clean installed base-wheel tests pass without CUDA initialization or
  hidden source-tree dependencies.
- [x] A `0.2.0rc1` candidate has been independently installed and evaluated,
  and its limitations are recorded before promotion.
  *(Wheel and sdist were built, checked, and installed outside the checkout.
  TestPyPI publication is explicitly skipped by project decision until a later
  milestone is ready for the real PyPI.)*
