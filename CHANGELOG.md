# Changelog

## Versioning policy

Lunarscout uses Semantic Versioning. Before 1.0, public APIs are provisional and breaking changes may occur in minor releases. Intentional breaking changes must be recorded here. Patch releases should not intentionally break documented behavior. The 1.0.0 milestone is reserved for a documented stable API surface and standalone native runtime story.

## Unreleased

- **Map-algebra numeric consistency, part 9: eager focal accumulators.** Made
  the shared accumulator-dtype rule authoritative for eager focal sum, mean,
  standard deviation, count, minimum, and maximum across public calls,
  expression inference, and explicit whole-raster `compute()`. FP32
  sum/mean/std now use FP32 working arrays and return FP32; FP64 remains FP64.
  Signed, unsigned, and Boolean sums use fixed-width `int64`, `uint64`, and
  `int64` CPU accumulators without an FP64 intermediary, preserving exact
  `uint64` values beyond `2**53`; overflow follows documented NumPy wrap
  behavior. Integer/Boolean mean and standard deviation use exact-integer CPU
  totals and second moments before returning FP64, so large adjacent values do
  not collapse before centering. Count uses bounded `uint32` neighborhood
  state with the existing public `int64` encoding, and min/max preserve exact
  source-dtype values. Added focal expression dispatch to `compute()`, strict
  non-negative-integer `ddof` validation and boundary invalidity, registry
  dtype metadata, fresh-process public coverage, and semantic version 3 for
  sum/mean/std/min/max/range. Focal range, median, convolution, general
  windowed halos, and GPU focal execution remain deferred; median/convolution
  still use double-precision SciPy paths. Verification: 1787 passed, 17 skipped
  in the ordinary CPU suite; 1325 map-algebra tests passed; 441 focused focal
  and numeric-policy tests passed; fresh-process public API and import-side-
  effect audits passed.

- **Map-algebra numeric consistency, part 8: temporal accumulators.** Made the
  shared accumulator-dtype policy authoritative for eager and layer-streamed
  temporal reductions. FP32 mean, standard deviation, and sum now execute and
  return FP32 instead of being unconditionally promoted to FP64; FP64 sources
  remain FP64. Signed, unsigned, and Boolean sums use `int64`, `uint64`, and
  `int64` CPU accumulators respectively, preserving exact `uint64` values
  beyond `2**53` instead of routing integer payloads through FP64. Integer
  mean/std retain the documented CPU/interchange FP64 correctness path, count
  remains `int64`, and min/max preserve source dtype. Eager, in-memory
  expression, and file-backed expression behavior now share the contract.
  Bumped `temporal.mean`, `temporal.std`, and `temporal.sum` to semantic
  version 2 and aligned registry metadata, user guidance, architecture, and
  the implementation plan. Verification: 1739 passed, 17 skipped in the
  ordinary CPU suite; 1277 map-algebra tests passed; 141 focused temporal
  tests passed; fresh-process public API and import-side-effect audit passed.

- **Map-algebra numeric consistency, part 7: unit-bearing power.** Added the
  public `output_units` contract to `ma.power`. A unit-bearing raster base now
  requires a scalar exponent; exponent one preserves source units, while every
  other exponent requires an explicit non-empty derived-unit declaration.
  Raster exponents must carry no units, are rejected for unit-bearing bases,
  and cannot claim one fixed output unit. The shared helper runs before eager
  kernels and during expression construction; compute and windowed execution
  replay the declaration, and it participates in scientific/restart identity.
  Equivalent whitespace and explicit-`None` declarations are canonicalized.
  Bumped `local.power` to semantic version 3 and aligned its signature,
  docstring, registry parameters, user guidance, architecture, and plan.
  Verification: 1730 passed, 17 skipped in the ordinary CPU suite; 1268
  map-algebra tests passed; 518 focused numeric, unit, identity, and window
  tests passed; fresh-process public API and import-side-effect audit passed.

- **Map-algebra numeric consistency, part 6: FP32 normalization.** Routed
  `normalize_minmax` and `standardize` through the shared result-dtype policy
  for both eager and expression construction. FP32 and Boolean/8/16-bit
  sources now calculate data-derived statistics and normalized arrays in FP32
  instead of unconditionally converting all work to FP64. Explicit NumPy FP64
  statistics, Python statistics outside the finite FP32 range, FP64 sources,
  and 32/64-bit integer sources retain the documented CPU/interchange FP64
  correctness path; int32 is not demoted because large adjacent integers can
  collapse before centering.
  Eager, expression, compute, all-invalid, and supported multi-window execution
  now agree on output dtype. Bumped both normalization semantic versions to 2
  and aligned registry metadata, docstrings, user guidance, architecture, and
  the implementation plan. Verification: 1714 passed, 17 skipped in the
  ordinary CPU suite; 1252 map-algebra tests passed; 502 focused numeric,
  normalization, classification, and window tests passed; fresh-process public
  API and import-side-effect audit passed.

- **Map-algebra numeric consistency, part 5: exact reclassification.** Routed
  eager and expression `reclassify_values`/`reclassify_ranges` output inference
  through the shared exact dtype engine. Small Python integer classes now use
  the smallest supported common dtype instead of forcing `int64`; typed FP32
  classes and exactly FP32-representable Python floats remain FP32; exact
  Python `uint64` values beyond `2**53` remain integers; and incompatible
  signed/unsigned 64-bit output sets raise
  `map_algebra_no_exact_promotion` rather than passing through FP64.
  `default="preserve"` now includes the complete source dtype in inference
  instead of sampling one payload value. Eager, expression, compute, and
  multi-window GeoTIFF execution share the contract. Bumped both
  reclassification semantic versions to 2 and aligned registry metadata,
  docstrings, user guidance, architecture, and the implementation plan.
  Verification: 1697 passed, 17 skipped in the ordinary CPU suite; 1235
  map-algebra tests passed; 483 focused numeric/classification/window tests
  passed; fresh-process public API and import-side-effect audit passed.

- **Map-algebra numeric consistency, part 4: exact nodata and invalid-fill
  encoding.** Centralized exact encoding validation across raster ingestion,
  `Raster.filled()`, eager `fill_invalid`, `to_existing`, GeoTIFF metadata, and
  windowed output. Finite fills must round-trip through the target dtype;
  out-of-range or fractional integers, Boolean-as-integer fills, infinities,
  and lossy Python-float-to-FP32 encodings now raise structured errors instead
  of wrapping or rounding. Explicit already-rounded `np.float32` encodings,
  NaN floating nodata, GDAL-style integral floating integer metadata, and exact
  0/1 Boolean metadata remain supported. Windowed writes preflight the encoding
  before changing staging state, preserve an existing destination on failure,
  and use the same non-mutating fill helper per window. Added exact `uint64`
  coverage beyond `2**53`, masked-ingestion, eager/expression parity, stable
  boundary errors, and multi-window storage round trips. Bumped
  `local.fill_invalid` semantic version to 2. Verification: 1684 passed, 17
  skipped in the ordinary CPU suite; 1222 map-algebra tests passed; fresh-
  process public API/import and exact-output audit passed.

- **Map-algebra numeric consistency, part 3: exact selection.** Routed eager
  and expression ``where``/``coalesce`` dtype inference through the shared
  numeric-policy helper. Python integer branches and fallbacks now use their
  smallest exact supported dtype instead of NumPy weak-scalar wrapping;
  incompatible signed ``int64``/``uint64`` domains raise structured
  ``map_algebra_no_exact_promotion`` rather than passing through FP64.
  ``coalesce`` now copies only first-valid values directly into the inferred
  dtype, removing its unconditional FP64 intermediate and preserving exact
  ``uint64`` values beyond ``2**53`` through eager, expression, and windowed
  GeoTIFF execution. Selection preserves FP32 for representable Python
  floating scalars, promotes out-of-range scalars to FP64, and consistently
  validates/preserves raster units. Bumped ``local.where`` and
  ``local.coalesce`` semantic versions to 2 and aligned registry metadata,
  public docstrings, user guidance, architecture, and the implementation plan.
  Recorded the project owner's criticality-first preference for all remaining
  core-plan work. Verification: 1654 passed, 17 skipped in the ordinary CPU
  suite; 1196 map-algebra tests passed; fresh-process public API/import audit
  and ``git diff --check`` passed.

- **Map-algebra eager connected-region adapters.** Added eager Boolean-Raster
  adapters for connected-region labeling, per-cell region sizes, size
  filtering, and internal borders. Adapters preserve the complete grid and
  canonical validity mask, clear units, return deterministic ``int32`` labels/
  sizes or Boolean filters/borders, reject numeric truthiness and expressions,
  and delegate to the established array algorithms. Both the existing
  ``ls.*`` array APIs and new map-algebra adapters accept explicit four- or
  eight-neighbor connectivity; existing calls retain the eight-neighbor
  default. Cleanup morphology follows the selected connectivity. Registry
  metadata declares these operations eager-only/global-cost. Cross-window
  connected-component reconciliation remains deferred. Verification: 1641
  passed, 17 skipped in the ordinary CPU suite; 1183 map-algebra tests passed.

- **Map-algebra eager API completion: layer stacks and focal validity
  thresholds.** Added public ``sum_layers``, ``mean_layers``, ``min_layers``,
  and ``max_layers`` helpers with strict validity intersection, grid/unit
  validation, checked overflow and numeric-error policies, mixed
  eager/expression dispatch, deterministic identity, and bounded windowed
  writes through composition of existing local nodes. The helpers do not
  allocate a three-dimensional layer stack. Added ``min_valid_count`` to eager
  focal statistics and convolution for ``valid_neighbor="ignore_invalid"``,
  with footprint-aware bounds, structured validation, expression identity,
  and preservation of the existing default zero-count contract. Focal
  registry metadata and expression-time parameter validation now match the
  public signatures. General bounded focal execution remains deferred.
  Verification: 1621 passed, 17 skipped in the ordinary CPU suite; 1163
  map-algebra tests passed.

- **Map-algebra numeric consistency, part 2.** Completed exact checked integer
  power using bounded repeated squaring, with ``raise``, exact ``promote``, and
  explicit ``wrap`` overflow policies and numeric-domain handling for negative
  integer exponents. Extended value-level cast safety with
  ``overflow="raise"`` by default and integer-to-integer ``"wrap"``; checks
  handle mixed signed/unsigned 64-bit boundaries and floating-to-integer limits
  without FP64 conversion of integer payloads. Extended
  ``numeric_errors="invalid"|"keep"|"raise"`` consistently across applicable
  pointwise arithmetic, math, angle, hypot, and rounding kernels. Policies are
  carried through eager, expression, windowed, registry, and canonical
  identity paths. Added complete supported-dtype pair matrices for ordinary
  addition inference and representable unsafe casts, plus exact ``uint64``,
  FP32 overflow, scalar-left power, cast-boundary, registry, and multi-window
  parity coverage. CPU/reference support for FP64 and 64-bit integers remains
  intentionally separate from future GPU hot paths. Verification: 1584
  passed, 17 skipped in the ordinary CPU suite; 1126 map-algebra tests passed.

- **Map-algebra numeric consistency, part 1.** Recorded consumer-grade GPU
  precision policy: inferred FP32 calculations remain FP32, and FP64 is used
  only when an input or documented result/accumulator contract requires it.
  Replaced FP64-based checked-integer intermediates for add, subtract,
  multiply, floor divide, remainder, negate, absolute, and square with exact
  native-integer boundary checks. Exact ``int64``/``uint64`` values beyond
  ``2**53`` are preserved; ``overflow="promote"`` selects an exact supported
  integer dtype or raises a structured error when none exists, while explicit
  ``"wrap"`` retains NumPy behavior. Centralized arithmetic and unary
  expression dtype inference through the shared dtype helper, including
  correct Boolean results for integer comparisons and preservation of explicit
  NumPy scalar precision. Added public ``numeric_errors="invalid"|"keep"|"raise"``
  handling to division, floor division, remainder, square root, exponential,
  logarithm, arcsine/arccosine, and square operations, with structured
  ``map_algebra_numeric_error`` failures and eager/expression/window parity.
  Numeric policies are recorded in expression identity and operation-registry
  metadata. Integer power, cast overflow, and broader dtype matrices were left
  for part 2.
  The accelerator contract also records that FP64 and software-emulated
  ``int64``/``uint64`` are CPU correctness/interchange capabilities rather
  than CUDA hot-path dependencies; future GPU planning must reject or
  explicitly route such work unless separately benchmarked.

- **Map-algebra planning split.** Further work whose primary purpose is
  processing maps too large for memory is deferred by project decision and
  moved from the core implementation plan into
  ``docs/map-algebra-large-raster-plan.md``. The new plan records the completed
  planner, bounded local/coordinate/terrain/resampling execution, durable writer
  lifecycle, and read-only diagnostics, plus the deferred halo/focal,
  cross-window region, streaming global/zonal, bounded distance, temporal
  batching, concurrency, and empirical resource-scaling work. Existing bounded
  capabilities remain supported. The active core plan now focuses on numeric
  and validity consistency, eager API gaps, identity/registry completeness,
  structured errors, documentation, and adversarial tests.

- **Map-algebra 0.3 critical-path slice 3: writer lifecycle control.**
  ``ma.write()`` now accepts optional ``progress_callback`` (reports completed
  windows, total windows, and current window index after each successfully
  written window; completion is reported exactly once) and
  ``cancellation_requested`` (checked before execution and between windows;
  raises ``OperationCancelledError`` with code ``map_algebra_cancelled`` and
  never publishes a partial output).  Progress callbacks that raise propagate
  the exception after resource cleanup; progress does not affect scientific
  results.
  Implemented durable checkpoint journaling (format 2): a hidden staged
  GeoTIFF (``.{name}.lunarscout-partial.tif``) and journal
  (``.{name}.lunarscout-partial.journal.json``) enable resumable writes.
  Journal identity binds the expression scientific identity, output dtype,
  invalid fill, complete destination grid, window layout, checkpoint interval,
  validity encoding, and enforced GeoTIFF write options through a deterministic
  SHA-256 hash.
  The journal stores a compact contiguous completed-window prefix rather than
  an area-sized set. Windows are journaled at checkpoint boundaries (default 16
  windows) after the staged TIFF is closed to flush data.  On resume,
  journaled windows are skipped; uncommitted or ambiguous windows are
  recomputed.  Incompatible, malformed, truncated, stale, or out-of-range
  journal state is safely ignored. A staged TIFF is reused only when its
  identity, dtype, CRS, transform, nodata, dimensions, and block layout match.
  Journal updates are atomic (complete write + fsync + rename + directory
  sync). TIFF and manifest publication uses paired exception rollback, and
  journal/staging artifacts are cleaned up only after success. Failed
  overwrites preserve the previous complete output and manifest, and
  deterministic backups left by an interrupted rename sequence are recovered
  on the next call. Cancellation and resume work correctly with retained
  restart state and with ``start_fresh=True``. Dataset handles and caches close
  deterministically
  after success, failure, and cancellation.
  ``ExecutionPlan`` now exposes enforced journal identity inputs, journal,
  progress, and cancellation capability flags, and the actually resumable
  ``windowed_execution`` stage; public ``ma.plan()`` reports the same enforced
  default-write diagnostics without executing kernels.
  Added 55 lifecycle-focused tests across the writer suites covering progress
  monotonicity and completion, callback failure cleanup, cancellation before
  execution,
  cancellation after several windows, cancellation with an existing
  destination, resource cleanup, journal creation and incremental updates,
  successful resume with proof that completed kernels are skipped, incompatible
  expression/dtype/grid/window/checkpoint/nodata/write-option state,
  malformed/truncated/stale/out-of-range journal handling, injected
  value-before-mask and journal-update failures, paired publication rollback,
  interrupted-publication recovery, non-divisible edge windows, terrain halos
  and resampling under cancellation/resume, integer nodata metadata, and no
  regression in ordinary non-restart writes.
  Verification: 1278 passed, 17 skipped (ordinary CPU suite); 820 map-algebra
  tests passed.

- **Map-algebra 0.3 critical-path slice 1: bounded spatial window execution.**
  ``ma.write()`` no longer materializes the complete spatial expression; it
  evaluates the expression graph per output window, reading bounded source
  windows with an LRU source-window cache and writing results block-by-block.
  Peak working memory depends on active sources, graph complexity, window
  size, and synchronous window output---not total raster area. Coordinate
  rasters are generated only for the requested window. Repeated source windows within a
  task are reused from the cache. Dataset handles and caches have explicit
  bounds and close after success and failure. Existing atomic overwrite
  guarantees remain intact. ``ma.compute()`` remains the explicit
  whole-raster materialization path.
  New private modules:
  - ``_planner.py``: topological validation, node/depth/source limits,
    unsupported-operation rejection (focal, global, zonal, distance, temporal),
    window size selection, and planner metadata reporting.
  - ``_windows.py``: output window enumeration, bounded source-window reading
    with LRU caching, coordinate window generation, and explicit cache bounds.
  - ``_windowed.py``: per-window expression evaluation through the shared eager
    semantic dispatcher for local binary, unary, classification, conditional,
    and supplied-statistic normalization operations.
  Updated ``plan()`` to report window layout, source count, estimated
  per-window memory, and planner metadata. Registry now advertises
  ``file_backed`` support for classification/normalization operations
  exercised through the windowed executor. Measured min/max or mean/standard
  deviation normalization is rejected in file-backed mode until a bounded
  multi-pass reducer is implemented. Focal/global/zonal/distance/temporal nodes
  are rejected before staging; callers must use explicit ``compute()`` where
  whole-raster materialization is intended. The temporal example now makes
  that boundary explicit.
  Verification: 1088 passed, 17 skipped (ordinary CPU suite).
- **Map-algebra 0.3 critical-path slice 2: halo-aware terrain execution and
  explicit cross-grid resampling.**  ``ma.slope()``, ``ma.aspect()``, and
  ``ma.hillshade()`` accept ``Raster`` or ``RasterExpression`` operands.
  Expression nodes carry a one-pixel halo; ``ma.write()`` expands each output
  window by one pixel per terrain node, evaluates through the existing
  scientific terrain kernels, and crops back to the exact output window.
  Cumulative halo propagation, destination-window-to-source-window mapping, and
  exact nearest-neighbour sampling for ``int64``/``uint64`` payloads are
  implemented.  ``ma.resample_to()`` creates explicit cross-grid nodes with
  categorical-vs-continuous safety rules (interpolation rejected for
  categorical data; ``mode`` rejected for continuous data; Boolean
  interpolation requires ``allow_unsafe=True``), safe output-dtype checks,
  rejection of continuous interpolation into integer output unless explicitly
  overridden, default nearest-neighbour categorical validity, optional
  validity-coverage thresholds, and GDAL-free nearest resampling that preserves
  64-bit integers beyond float precision. Categorical inference uses the source
  dtype, independent of the requested output dtype. ``ma.align()`` is the eager
  ``Raster`` adapter and supports source-preserving, explicit, or disabled
  output-nodata metadata. Implicit resampling is never inserted by other
  operations. Registry metadata for all four new
  operation IDs is enriched with parameter descriptions, dtype/unit/validity
  rules, cost classes, and examples.  Public functions, tests, documentation,
  and registry claims are verified end-to-end.
  New private module:
  - ``_spatial.py``: terrain/resample expression construction, conservative
    destination-to-source window mapping, exact
    nearest-neighbour resampling, and eager evaluators.
  Added 110 public terrain and resampling tests covering eager/expression
  parity, windowed write parity, slope degrees/percent/scale, aspect flat-cell
  invalidity, hillshade azimuth/altitude/scale/z-factor, compute-edges, invalid
  cells across window boundaries, numeric nodata collisions, dtypes/units,
  structured parameter errors, canonical identity changes, resampling onto
  finer/coarser/shifted/rotated/differing-CRS grids, partial/no coverage,
  explicit validity masks, default nearest validity, coverage-threshold
  stability, exact int64/uint64 nearest payloads, categorical safety
  rejection, explicit overrides, no implicit resampling in binary operations,
  registry filtering, file-backed claim audits, Boolean categorical mode,
  zero-threshold coverage, output-nodata handling, and dtype-safety rules.
  Key limitations still deferred: general focal kernel window execution,
  footprint-derived asymmetric halos, local fusion, completed-window
  journal/resume, cancellation/progress hooks, global/zonal/distance bounded
  execution, region adapters, temporal spatial-window/time-batch mapping, and
  an exact extreme-``int64``/``uint64`` declared GeoTIFF nodata contract beyond
  Rasterio/GDAL representability. Verification: 1223 passed, 17 skipped
  (ordinary CPU suite); 765 map-algebra tests passed.
- Completed the remaining small and medium Phase I map-algebra inventory:
  exact-value and half-open-range reclassification, digitization, one-hot
  class rasters, min/max normalization, standardization, and lazy row,
  column, projected-coordinate, longitude, and latitude expression sources.
  Coordinate units now come from the grid CRS rather than assuming metres,
  longitude/latitude use the grid's own geodetic CRS, and classification
  preserves exact ``uint64`` payloads. Added explicit eager/expression operand
  adapters and expression evaluation for the new operations. Added a sealed
  static operation registry with public ``describe_operation()`` and
  ``list_operations()`` discovery, concise expression descriptions, and a
  versioned canonical JSON schema with typed arbitrary-size integers,
  hexadecimal floats, normalized CRS/affine/dtype/structured parameters,
  semantic operation versions, stable node ordering, and rejection of
  unsupported parameter types. Updated map-algebra documentation and the
  proposed example portfolio, and added regression coverage for eager and
  expression behavior, exact integer boundaries, validity and units,
  non-metre grids, registry sealing, and spatial/temporal serialization.
  Verification: 991 passed, 17 skipped (CPU suite).
- Implemented Phase I of the broad map-algebra plan: added a comprehensive
  Map Algebra chapter to ``docs/USER_GUIDE.md`` covering value types, eager
  vs file-backed workflows, constructors/adapters, grid/validity/dtype/unit
  rules, and all five operation families (local, focal, zonal, global,
  distance) plus temporal map algebra.  Added map-algebra architecture
  documentation to ``docs/ARCHITECTURE.md`` (section 20) describing the
  value model, operation registry, execution architecture, temporal
  execution flow, storage flow, dispatch rules, and module inventory.
  Added three runnable examples: ``18_map_algebra_screening.py`` (terrain-
  lighting screening with weighted scoring), ``19_map_algebra_focal.py``
  (focal smoothing, morphology opening, distance fields), and
  ``20_map_algebra_temporal.py`` (temporal source, temporal reduction
  composed with spatial algebra).  Updated ``examples/README.md`` with the
  new scripts.  Exported ``TemporalRaster`` from the package root with
  the alias ``from_temporal_cube_to_raster``.  Built clean wheel and sdist,
  verified ``pip check``, and confirmed smoke import from installed wheel.
  Verification: 931 passed, 17 skipped (CPU suite).
- Implemented Phase H of the broad map-algebra plan: eager
  ``TemporalRaster`` values with canonical three-dimensional validity and
  strict UTC-coordinate validation; explicit ``TemporalCube`` adapters;
  sealed ``TemporalRasterExpression`` graphs; scalar, temporal, and static
  spatial layer-wise algebra with exact grid and ordered-time validation; and
  descriptor-based ``TemporalGeoTiffSeries`` sources that do not retain live
  dataset handles. Added explicit temporal materialization plus composable
  spatial mean, minimum, maximum, standard-deviation, sum, and count
  reductions. File-backed reductions evaluate one temporal layer at a time,
  preserve masks, grids, signal metadata, units, and documented output dtypes,
  and keep all-invalid count cells as valid zero. Added 132 temporal
  map-algebra tests, including scalar-left operators, closed-series execution,
  reduction composition, grid rejection, reducer semantics, and a 3,000-layer
  bounded-time execution check. Also updated stale renumbered example paths and
  the relocated historical HDF5 benchmark smoke test.
- Implemented Phase A of the broad map-algebra API plan
  (``docs/map-algebra-implementation-plan.md``): the public ``Raster`` frozen
  dataclass with explicit spatial and validity metadata, custom cell-by-cell
  ``==``/``!=`` operators, and whole-raster comparison helpers
  (``array_equal``, ``allclose``, ``same_grid``, ``same_metadata``); a
  ``map_algebra`` namespace (``ls.map_algebra``) with eager constructors
  (``raster()``, ``from_masked_array()``, ``from_existing()``,
  ``to_existing()``) and a provenance-aware GeoTIFF ``read()``; nine
  structured error classes under ``MapAlgebraError``; mandatory
  representability validation for integer and floating nodata; preservation
  of masked-array masks through the ``raster()`` constructor; an explicit
  ``validity_provenance`` field distinguishing explicit-caller, nodata,
  all-valid, masked-array, and geotiff:... sources; and ``__bool__`` raising
  ``TypeError`` so implicit truth testing is unavailable by design.
  Added ``tests/map_algebra/`` with 104 tests and fixtures for north-up,
  anisotropic, rotated, shifted, differing-CRS, masked-raster, and
  partial-coverage grids.  Verification: 559 passed, 17 skipped (CPU suite).
- Implemented Phase B of the map-algebra plan: eager local algebra with
  arithmetic (``add``, ``subtract``, ``multiply``, ``divide``, ``negative``,
  ``absolute``), pairwise (``minimum``, ``maximum``), comparisons (``less``,
  ``less_equal``, ``greater``, ``greater_equal``, ``equal``, ``not_equal``),
  strict Boolean operations (``logical_and``, ``logical_or``, ``logical_xor``,
  ``logical_not`` requiring boolean dtypes), conditional/validity helpers
  (``where`` with ``ma.invalid`` sentinel, ``coalesce``, ``is_valid``,
  ``is_invalid``, ``set_invalid``, ``fill_invalid``), value operations (``clip``,
  ``cast``), and math functions (``sqrt``, ``square``, ``exp``, ``log``,
  ``log10``, ``sin``, ``cos``, ``tan``, ``arcsin``, ``arccos``, ``arctan``,
  ``arctan2``, ``hypot``, ``degrees``, ``radians``, ``floor``, ``ceil``,
  ``trunc``, ``round``).  Added operator overloads to ``Raster`` for ``+``,
  ``-``, ``*``, ``/``, ``//``, ``%``, ``**``, ``-`` (neg), ``+`` (pos),
  ``abs()``, ``<``, ``<=``, ``>``, ``>=``, ``&``, ``|``, ``^``, ``~``,
  ``round()``, ``math.floor``, ``math.ceil``, ``math.trunc`` with unit-aware
  dispatch, grid validation, and validity-intersection semantics.  Internal
  modules: ``_dtypes.py`` (promotion, overflow, casting), ``_units.py``
  (conservative unit equality), ``_validity.py`` (mask combination, numeric
  domain), ``_validation.py`` (operand/grid normalization), ``_kernels.py``
  (eager NumPy kernels), ``_eager.py`` (raster/raster, raster/scalar dispatch).
  Added 99 tests covering arithmetic, comparisons, booleans, ``where``/
  ``coalesce``, validity helpers, clip/cast, math, operator overloads, and
  compound expression examples.  Verification: 658 passed, 17 skipped (CPU
  suite).
- Corrected six Phase B P1 defects found during review: (1) wired integer
  overflow detection into eager dispatch so ``uint8(255) + 1`` raises
  ``MapAlgebraDTypeError`` instead of returning valid zero; (2) added angle-unit
  validation to ``sin``/``cos``/``tan`` (require "degrees" or "radians"), set
  proper output units on inverse trig (``radians``), and required explicit
  ``output_units`` for multiplication/division of two unit-bearing rasters; (3)
  fixed ``coalesce()`` to preserve original argument order instead of evaluating
  all rasters before all scalars; (4) corrected scalar-left ``less``/
  ``less_equal``/``greater``/``greater_equal`` to use the semantically correct
  swapped kernel; (5) made ``set_invalid()`` intersect ``mask.valid`` with
  ``mask.values`` so invalid mask cells cannot alter validity; (6) fixed
  ``where()`` to accept scalar-only branches (``where(cond, 1, 2)``,
  ``where(cond, 1, ma.invalid)``).  Also added missing public functions
  (``floor_divide``, ``remainder``, ``power``, ``positive``, ``isclose``), made
  ``Raster.__eq__``/``__ne__`` handle scalar operands, fixed reverse floor-div/
  mod/power operators to avoid dtype-truncating ``np.full_like``, made
  ``fill_invalid()`` validate fill value representability, and fixed
  ``round(raster, ndigits)`` to pass through the ndigits argument.
  Verification: 658 passed, 17 skipped (CPU suite).
- Implemented Phase C of the map-algebra plan (revised after review): a
  ``RasterExpression`` immutable DAG with sealed constructor (``_sealed``
  sentinel), Kahn-algorithm topological sort for shared-dependency DAGs,
  operator overloads for all Phase B arithmetic, comparison, Boolean, and
  unary operations, metadata-only ``ma.source()`` (reads GeoTIFF profile
  without loading band data, ``identity="stat"|"sha256"`` option, structured
  ``GeoTiffOpenError``), ``ma.compute()`` delegating to Phase B eager
  functions so all unit/dtype/overflow/boolean policies are preserved,
  ``ma.explain()`` for human-readable node tree, ``ma.plan()`` for dry-run
  validation, deterministic sequential node IDs with ``sha256:``-prefixed
  scientific identity hashes that distinguish distinct values, and canonical
  JSON serialization with full CRS, affine, and nodata metadata.  Wrapped 19
  unary functions (``sqrt``, ``sin``, etc.) with expression dispatch so
  ``ma.sqrt(expr)`` returns a ``RasterExpression`` node.  ``Raster``
  operators promote to ``RasterExpression`` when mixed with expression
  operands.  Internal modules: ``_model.py`` (topological sort, sealed
  construction, identity), ``_sources.py`` (metadata-only reads, identity
  modes), ``expression.py`` (compute via Phase B delegation, explain, plan).
  Added 26 tests.  Verification: 684 passed, 17 skipped (CPU suite).
- Implemented Phase D of the map-algebra plan (revised after review):
  ``ma.write(path, expression)`` with GDAL mask writing via ``write_mask()``
  at dataset creation, two-phase atomic publication (TIFF + manifest written
  to staging directory, atomically moved together), preflight validation
  (dtype, grid, fill) before ``compute()``, safe widening-only dtype
  conversion (bool→uint8, float→wider float, int→wider int, int→float),
  multi-field restart identity (scientific identity, output dtype, fill,
  grid dimensions), and ``Raster.expression()`` for creating constant
  expression nodes from eager rasters.  All 10 review findings addressed.
  Verification: 700 passed, 17 skipped (CPU suite).
- Implemented Phase E of the map-algebra plan (revised after review): an eager
  SciPy-based focal and morphology API with 14 operations — ``focal_sum``,
  ``focal_mean``, ``focal_min``, ``focal_max``, ``focal_range``,
  ``focal_std`` (with ``ddof`` forwarding), ``focal_count``,
  ``focal_median``, ``convolve`` (custom kernels, ``normalize`` option),
  ``dilate``, ``erode``, ``opening``, ``closing``, and ``majority``.
  Features: five edge modes (``invalid``, ``constant`` with ``cval``,
  ``nearest``, ``reflect``, ``wrap``), three valid-neighbor policies
  (``require_all``, ``ignore_invalid`` using nan-aware reductions,
  ``propagate_center``), safe output dtypes (``int64`` for sum/count,
  ``float32``/``float64`` for mean/std/median/range), morphology validity
  masking (invalid cells forced to ``False`` before SciPy ops), validated
  finite convolution kernels, zero-width halo guarding, and expression
  dispatch via ``_wrap_focal`` so ``Raster.expression()`` operands return
  expression nodes.  Internal module: ``focal.py`` (560 lines).  Added 38
  tests covering all operations, dtypes, edge modes, validity policies,
  expression dispatch, ddof, and edge cases.
  Verification: 738 passed, 17 skipped (CPU suite).
- Implemented Phase F of the map-algebra plan (revised after three review
  rounds): global and zonal reduction operations — ``statistics()`` returning
  a ``RasterStatistics`` dataclass (count, invalid_count, sum, mean, min, max,
  range, variance, std), ``histogram()``, ``percentile()`` (exact linear /
  approximate nearest-rank, both in-memory with documented float64 precision
  limit), and ``unique_counts()`` (with ``max_unique`` safety bound); zonal
  ``ZonalStatistics`` dataclass with sorted zone IDs, per-column int64/float64
  arrays, ``include_zone_ids`` (dtype-validated), ``zone_nodata``, structured
  errors for unknown statistics, ``to_dict`` / ``to_json`` / ``to_records``
  (tuple of immutable ``MappingProxyType`` rows) / ``write_csv`` serializers,
  and write-protected result arrays; ``zonal_stats()`` correctly separates
  zone validity from value validity for accurate count/valid_count/
  invalid_count; and ``zonal_raster()`` broadcasting to all zone-valid cells.
  Internal modules: ``reductions.py``, ``zonal.py``.  Added 31 tests covering
  statistics, histogram, percentile, unique_counts, zonal stats with
  validity separation, percentiles, uint64 boundary IDs, zone_nodata,
  include_zone_ids, empty zones, and immutable records.  Streaming/bounded
  accumulators remain deferred.  Verification: 769 passed, 17 skipped
  (CPU suite).
- Implemented Phase G of the map-algebra plan (revised after review): eager
  distance fields with ``distance_to()`` and ``signed_distance()`` supporting
  three metrics (``euclidean`` via scipy EDT, ``taxicab``/``chessboard`` via
  2-pass algorithms), two unit modes (``pixels`` and ``physical``), and
  ``max_distance`` clipping in output units.  Corrected signed-distance
  semantics (True→dist to False, False→−dist to True).  Physical units
  require square isotropic projected CRS (validated via pyproj
  ``is_geographic`` and affine analysis); taxicab/chessboard physical
  rejected.  Input validity preserved through distance computation.
  Structured errors for no seeds, all-true, all-false, invalid max_distance,
  geographic CRS, anisotropic grids, and unknown units/metrics.  Internal
  module: ``distance.py``.  Added 21 tests covering metrics, units,
  max_distance, validity preservation, signed distance, and rejection of
  invalid inputs.  Verification: 790 passed, 17 skipped (CPU suite).
- Added ``docs/map-algebra-implementation-plan.md``, the reviewed ``0.2.0rc1``
  execution plan for a broad lunar map-algebra API. The plan defines eager
  ``Raster`` values, bounded and resumable ``RasterExpression`` execution,
  canonical validity and dtype semantics, local/focal/zonal/global/temporal
  operations, distance fields, explicit non-terrestrial assumptions, durable
  output, performance gates, and machine-readable explanation/provenance
  features intended to support future human-reviewed LLM-assisted mission
  analysis. This is a planning and API-design addition; it does not claim that
  the map-algebra surface is implemented.
- Revamped the example suite: renumbered scripts 00–10 to 01–10 for
  contiguous ordering; moved the historical HDF5 storage benchmark to
  ``benchmarks/``; deleted stale ``.pyc`` files; and added four new
  examples covering incremental temporal writing (``07``), SPICE body
  vectors and azimuth/elevation (``11``), body elevation and horizon
  plotting on synthetic data (``12``), and a CPU synthetic lightmap
  with explicit vectors (``13``).  Added a synthetic 256×256 DEM
  generator script, a GitHub-Releases-backed download-and-cache helper,
  a SHA-256 manifest, and a rewritten ``examples/README.md`` with a
  data-requirements table and per-script guidance.  Rewrote the
  horizon-generation example (``16``) with ``--primary-dem`` and
  ``--surrounding-dem`` arguments.
- Updated ``docs/USER_GUIDE.md`` with an Examples section linking to
  ``examples/README.md`` and an ``examples/`` domain summary table.
- Published ``synthetic-horizon-data-v1.tar.gz`` as a GitHub Release
  asset with tag ``synthetic-horizon-data-v1`` (37 MB, 256×256
  south-polar DEM with bowl, ridge, cone, and bump terrain plus four
  128×128 compressed ``.cbin`` horizon tiles).
- Bumped candidate version to ``0.1.0rc3``.  Updated the version assertion
  in ``tests/test_dependency_boundary.py`` to match.
- Deferred product performance benchmarking to a later release.  Marked the
  remaining Section 5 benchmarks in ``docs/PLAN1.md`` as deferred; correctness,
  restart, cancellation, and disabled-CUDA fallback evidence is complete.
  Noted the existing ``0.1.0rc1`` and ``0.1.0rc2`` TestPyPI publications in
  the plan's Sections 12 and M5.
- Added ``CONTRIBUTING.md`` covering dev setup, test commands, project layout,
  code conventions, version bumping, the release process, CUDA configuration,
  and the dependency boundary.
- Tuned the production Numba horizon kernel by capping it at 80 registers per
  thread, selecting 128-thread blocks, and keeping coordinate interpolation
  entirely in float32. The compiled kernel now uses zero local memory and
  contains zero PTX float64 operations. On the RTX 5090 Laptop validation
  system, isolated four-DEM kernel time improved by 11.0 percent and sustained
  16-patch throughput improved by 8.6 percent, from 0.17931 to 0.19465
  patches/s, with unchanged peak GPU memory. All 16 compressed horizon output
  hashes matched the previously accepted run exactly. Verification: 458
  ordinary tests passed with 17 skips; 139 explicitly CUDA-gated horizon tests
  passed with 1 skip.
- Corrected the monthly safe-haven streaming reducer so a low-Sun run that
  overlaps an Earth outage continues accumulating after Earth clears, through
  the actual end of the low-Sun run. The previous state transition truncated
  these durations at the last below-threshold Earth sample unless the low-Sun
  run was still active at the end of the complete evaluation history.
  Safe-haven restart manifests now record semantics version 2 so staged patches
  produced by the truncated reducer cannot be resumed into corrected outputs.
- Reworked the safe-haven point spot check to use the exact production
  spherical-stereographic pixel frame and geometric Moon-ME vectors, then
  compare an independently reduced full point history with an existing
  safe-haven raster. Added a single-pixel reference calculation and regression
  coverage against the compiled CPU fraction and horizon-margin paths. Direct
  reproduction showed that the previously reported 65-degree azimuth and
  37-degree elevation frame discrepancy was not produced by the library frame
  implementations. Regenerated the 4,992 by 5,248 Mons Mouton CPU product for
  September 2027 through April 2028 at two-hour resolution, then independently
  verified all eight bands at 20 spatially distributed locations (160 value
  comparisons, zero failures); visual inspection in QGIS also passed.
- Bumped candidate version to ``0.1.0rc2``.  Reviewed and enhanced
  ``pyproject.toml`` metadata: expanded description, keywords (PSR,
  safe-haven, mission-duration, SPICE, Numba, geospatial, raster),
  classifiers (Astronomy and GIS topics), and ``[dev]`` extras (coverage).
  Removed ``License`` classifier in favour of the PEP 639 license
  expression (``License-Expression: Apache-2.0``).
- Completed a full PLAN1.md audit, checking off 70+ items across all
  sections for which evidence now exists: horizon-format documentation,
  algorithm identifiers, metadata-field compatibility promises, signature/
  docstring freeze, exception and operational tests, independent file
  validation (horizon and GeoTIFF), pyproject.toml review, dev extras,
  and M0-M5 milestone gates.  Remaining unchecked items are deferred
  complexity (disk-full injection, per-stage timing), TestPyPI
  installation verification (M5), and post-0.1 roadmap items (M7/M8).
  Verification (depends-on-doc changes only): 456 passed, 17 skipped.
- Ran real-GPU acceptance suite on RTX 5090 Laptop GPU (Numba 0.66.0, CUDA
  driver 13.0, compute capability 12.0): 137 passed, 1 skipped including the
  public CPU/CUDA matrix (all 9 products agree), safe-haven CPU/CUDA stream
  identity, and the complete Phase 4-6b CUDA horizon/PSR/lightmap/elevation/
  mission-duration gated suites.  Rebuilt clean wheel (165,898 bytes, 52
  entries, sha256 ``39130245...5cd02c``) and sdist (142,924 bytes, 65
  entries, sha256 ``2de0c2af...639fe9``) from commit ``9841beb``; twine
  check passed; installed wheel passed 54 public CPU tests outside the
  checkout.  (deepseek)
- Added 17 M2 validation tests: corrupt ``.cbin`` tile handling (truncated
  file, invalid block length, missing tile all produce all-invalid masks);
  process-termination recovery (``os._exit(23)`` mid-calculation, ``start_fresh``
  resume succeeds); independent ``.bin``/``.cbin`` dimension, dtype, and
  round-trip validation including azimuth-ordering convention check and public
  ``Scenario`` horizon reader; CPU timing sanity checks for lightmap, PSR,
  safe havens, and mission duration; GeoTIFF tiling (128×128), compression
  (deflate / none), nodata, per-band timestamps, and backend metadata
  validation via Rasterio/GDAL.  Verification: 456 passed, 17 skipped
  (CPU suite).  (deepseek)
- Published the horizon-tile file-format, directory-layout, naming, and
  partial-edge contracts in ``docs/USER_GUIDE.md``: ``.bin`` little-endian
  float32 pixel-major layout, ``.cbin`` per-pixel 7/15-bit variable-length
  delta encoding with 16-bit length prefixes, file stem
  ``horizon_{tile_y:05d}_{tile_x:05d}_{height_dm:03d}``, ``{tile_y:05d}``
  subdirectory lookup with compressed-before-raw precedence, and
  ``-50``-degree deterministic partial-edge padding.
- Assigned and documented stable algorithm identifiers and the shared
  ``phase6b-v1`` version for all nine downstream product families: lightmap,
  PSR, Sun/Earth elevation, safe havens, and four mission-duration variants.
  These appear in staged-job manifests, restart metadata, and the
  ``docs/USER_GUIDE.md`` reference table.
- Rewrote the safe-haven algorithm with per-pixel Earth outage detection,
  calendar-month band structure, and a streaming state-machine reducer that
  consumes per-pixel Earth terrain-relative elevation tiles alongside
  sunlight fraction tiles.  Each pixel's outage boundaries are determined
  from its own terrain horizon, not from the DEM center pixel.  Each output
  band represents one calendar month with ``[start_utc, stop_utc)``
  labelling.  Pixels where Earth never crosses the threshold during a month
  (always above or always below) receive NODATA.  The streaming reducer
  uses fixed per-patch arrays proportional to ``(bands × y × x)`` rather
  than allocating full per-pixel boolean timelines.  Updated ARCHITECTURE.md
  section 9.3 with the complete algorithm description and design rationale.
  Verification: 439 passed, 17 skipped (CPU suite).
- Completed ``0.1.0rc1`` release-article preparation from commit ``14e019a``:
  built clean wheel (164,054 bytes, 52 entries) and sdist (141,103 bytes, 65
  entries), passed ``twine check``, and installed the wheel in a clean CPU-only
  environment.  The installed wheel passed 54 public tests with 1 gated skip,
  ``pip check`` found no lunarscout dependency issues, no forbidden modules
  (pythonnet, moonlib, clr, _native_runtime) are importable, ``import lunarscout``
  loads no Numba or SpiceyPy modules, ``backend="auto"`` falls back to CPU,
  explicit ``backend="cuda"`` raises a structured ``CudaError``, and
  ``overwrite=False`` rejects existing outputs.  The full PUBLIC_API_REVIEW.md
  decision checklist is approved for ``0.1.0rc1``.  SHA-256 and record:
  * wheel   ``ee953977a...80de0e9``
  * sdist   ``69eca0f7...0801969``
- Added 30 public tests across six categories: structured exception class, code,
  and output preservation (7 tests); safe-haven boundary and outage behaviour (4
  tests including no-outage, whole-interval outage, adjacent outages, and missing
  horizons); mission-duration inclusive thresholds, candidate-start windows,
  evaluation endpoints, no feasible start, unit conversion, and multi-band output
  (6 tests); public cancellation, restart, failed-overwrite, and invalid-tile
  journaling (5 tests); 65,535-band limit rejection via ``ProductJob.manifest()``
  (1 test); and per-product timestamp and backend metadata tags (5 parametrized
  variants). Verified: 435 passed, 17 skipped (CPU suite). (deepseek)
- Defined the public GeoTIFF metadata compatibility promise for ``0.1.0rc1``:
  dataset-level ``LUNARSCOUT_TIMESTAMPS_UTC`` and
  ``LUNARSCOUT_COMPUTE_BACKENDS``; per-band ``TIMESTAMP_UTC`` on time-series
  products; per-band ``DURATION_UNIT``, ``CANDIDATE_START_UTC``, and
  ``CANDIDATE_STOP_UTC`` on mission-duration products; tiled, compressed BigTIFF
  with integer predictor 2 or float predictor 3. The mask is authoritative for
  both byte and float products.
- Completed the Scenario signature and docstring review: made all nine
  downstream Scenario method signatures explicit with full typed keywords
  mirroring the corresponding root functions. Removed the private
  ``_generator`` test-injection parameter from
  ``Scenario.generate_horizons()``; tests now use ``monkeypatch`` on
  ``lunarscout.horizon.generate_horizons``. Removed the PSR-only
  ``horizons=`` override from ``scenario.psr()`` for a consistent Scenario
  facade. Enhanced every public root function and Scenario method docstring
  to document times/evaluation intervals, vector precedence, backend
  behavior, compress/nodata/mask, output transforms, overwrite/restart,
  return value, progress, cancellation, and scientific thresholds. Updated
  ``docs/PUBLIC_API_REVIEW.md`` to record the completed review findings and
  check the corresponding decision-checklist items. Verification: 405 passed,
  17 skipped (CPU suite). (deepseek)
- Made `compress=True` explicit as the default for every tiled downstream
  GeoTIFF operation. Float products now expose `nodata=np.nan` while retaining
  authoritative dataset masks; byte products retain a zero storage payload
  without treating valid zero values as nodata. Added bounded patch output
  transforms with an explicit NumPy-compatible output dtype and an optional
  restart identity; omitting the identity on both runs is restart-compatible.
  Transform results must preserve patch shape and return the exact requested
  dtype. Added a single detailed public-signature review document covering
  these defaults, scientific meanings, restart rules, and remaining decisions.
- Revised the candidate API review: renamed the sunlight-and-Earth mission
  function to `mission_duration_from_sunlight_and_earth_elevation`, made
  `TimeRange` the ordinary product time input, derived mission samples from
  evaluation start/stop plus a `timedelta` step, added public product-ready
  Moon-ME vector generation, and exposed optional GeoTIFF tile compression.
  Existing outputs are rejected before DEM, SPICE, or CUDA work begins.
- Corrected safe-haven duration semantics to measure the complete contiguous
  low-Sun interval that overlaps an Earth outage, including portions before or
  after the outage. This intentionally changes safe-haven scientific output
  where shadow crosses an outage boundary.
- Added a public downstream-product example that defaults to CPU and can run
  lightmap, PSR, both terrain-relative elevation products, safe havens, and all
  four mission-duration operations from one existing scenario. It documents
  outage bands, duration units, backend behavior, and the absence of implicit
  slope, battery, thermal, or traverse policy; CI checks its command line from
  an installed wheel outside the checkout.
- Documented PSR rendering with valid zero and 255 classes, per-product restart
  recomputation bounds, safe staged-product cleanup through `start_fresh`, and
  troubleshooting for CUDA profiles, hidden devices, SPICE, Rasterio/GDAL,
  grids, and corrupt or missing horizons.
- Completed unit, interval, invalid-pixel, output-unit, and return-value
  documentation for the promoted elevation, safe-haven, and mission-duration
  functions, and removed obsolete managed-era “native runtime” wording from
  public Scenario and example documentation.
- Added the limited-testing support matrix, explicit known limitations, and a
  pre-upload `0.1.0rc1` evidence report. Verified ordinary operation with a
  configured read-only Numba cache and recorded that the wheel vendors no
  third-party code or binaries requiring an additional bundled notice.
  Added an isolated working-directory import test proving that the curated
  root loads no Numba, CUDA, SPICE, or managed modules, opens no raster, and
  creates no working-directory files.
- Completed a current-tree pre-upload diagnostic: the CPU suite passed with
  395 tests and 17 gated skips; wheel and sdist content inspection and Twine
  checks passed; the installed wheel passed 59 public tests with 7 gated skips
  on both Python 3.11 and 3.12; and the installed real-GPU horizon plus complete
  downstream CPU/CUDA matrix passed 2 tests in 14.26 seconds. The diagnostic
  artifacts are explicitly marked dirty and are not upload candidates. With
  CUDA deliberately disabled, the installed CUDA-profile wheel also completed
  an automatic lightmap on CPU and rejected explicit CUDA without creating an
  output.
- Defined the post-`0.1` roadmap without expanding the first candidate: finish
  and evaluate the lighting release through TestPyPI first, design map algebra
  and CPU/CUDA distance fields for `0.2`, and build policy-explicit path
  planning on those accepted raster contracts for `0.3`. Later APIs must
  preserve explicit grids, masks, units, bounded memory, backend truthfulness,
  and separation between scientific products and application policy.
- Added the `lunarscout[cuda]` installation profile for supported NVIDIA
  systems. The extra installs the validated Numba-CUDA CUDA 12 user-space
  stack but not an NVIDIA driver; the base installation remains CPU-only.
  CUDA capability probes and explicit CUDA operations now reject an unrelated
  system Numba CUDA target and provide the exact extra-install command. A clean
  base-wheel environment passed 380 tests with 27 optional or real-GPU skips;
  a separate clean `cuda` installation passed its 159-test installed-wheel
  CUDA suite with one skip on an RTX 5090 Laptop GPU using Numba 0.66.0,
  Numba-CUDA 0.30.4, CUDA toolkit 12.9.2, and compute capability 12.0.
  `ls.cuda.status()` now exposes those runtime versions plus the CUDA driver
  API version and current free/total GPU memory as best-effort diagnostics.
  Numba-CUDA driver, PTX, JIT, and kernel exceptions are classified as
  structured CUDA execution failures at the public horizon and product
  boundaries rather than escaping as raw implementation exceptions.
  Added a gated public API matrix that compares complete CPU and CUDA arrays
  and masks for lightmap, PSR, both elevation products, safe havens, and all
  four mission-duration operations while checking truthful backend metadata.
- Added a clean-snapshot release-artifact workflow that refuses dirty release
  builds and nonempty output directories, builds wheel and sdist in isolation,
  runs Twine, enforces an explicit content policy, and records commit,
  environment, target, sizes, hashes, and entry counts without uploading.
  Added an explicit sdist manifest, installed-metadata `__version__`, and Linux
  Python 3.11/3.12 project classifiers and repository URLs.
- Added CPU CI for Python 3.11 and 3.12 plus a separate clean distribution job
  that inspects artifacts, rebuilds a wheel from the sdist, installs the wheel
  outside the checkout, runs `pip check`, verifies lightweight import, and runs
  installed public smoke tests. Real CUDA acceptance remains separately gated.
- Corrected the Rasterio requirement to `>=1.4.4,<1.6`: Rasterio 1.5 requires
  Python 3.12, while the supported Python 3.11 environment resolves to 1.4.4.
  Both interpreter environments remain covered by the same public API tests.
- Removed the historical HDF5 storage prototype from mandatory base-package
  example acceptance. It remains tested when its manual `h5py` and
  `hdf5plugin` packages are present, while clean base installations skip it.
- Verified the managed-code removal with the complete CPU suite, the explicitly
  gated real-CUDA suite, clean installed-wheel CPU tests, and installed-wheel
  CUDA horizon generation outside the checkout. Clean wheel and source
  distribution inspection found no managed sources or artifacts. The
  verification also exposed that setuptools can reuse stale files from an
  existing `build/lib` directory, so release artifacts must be constructed by
  a clean-build workflow rather than an in-place build with retained caches.
- Removed the superseded managed implementation after the boundary commit
  `c9c4e66`: the complete `native/` C# tree and bundled native assets, C#
  metric/oracle projects, Python.NET bootstrap and wrapper modules, managed-only
  tests and examples, transitional `Native*` exceptions, and managed terrain
  and temporal `Scenario` methods. Generated JSON/Markdown evaluations and
  language-neutral scientific fixtures remain as historical evidence. The
  production source tree, wheel, and source distribution are now
  Python/Numba-only.
- Recorded the final development checkpoint that retains the superseded C#,
  Python.NET, and managed-wrapper sources. They remain recoverable from this
  commit and the `main` branch and will be removed from the Python-only
  production tree in the following change.
- Prepared package metadata for the first immutable TestPyPI candidate,
  `0.1.0rc1`, and added build and Twine tooling to the development extra.
- Removed `ls.native`, `GenerateHorizons`, and `NativeHorizonProgress` from the
  curated package root. Transitional wrapper modules remain importable only by
  their explicit module paths as temporary migration evidence; ordinary
  `import lunarscout` no longer imports or advertises them.
- Promoted CUDA-only Python/Numba horizon generation as
  `ls.generate_horizons()` and `Scenario.generate_horizons()`. The public
  facade validates DEMs and output paths before execution, builds resident
  factor-four pyramids, uses the selected one-producer/one-CUDA-consumer/one-
  writer bounded pipeline, resumes structurally complete tiles, atomically
  publishes `.bin`/`.cbin` files, reports immutable CUDA progress events, and
  maps cancellation, CUDA capability, calculation, and storage failures to
  structured domain exceptions. Horizon generation intentionally has no
  backend argument or CPU fallback.
- Promoted Python-only public facades for lightmaps, PSR, Sun- and Earth-center
  terrain-relative elevation, safe havens, and all four landed
  mission-duration products. Root functions and `Scenario` conveniences share
  explicit vector/time resolution, `backend="auto"`, quiet-by-default verbose
  output, monotonic fraction and immutable structured progress callbacks,
  cooperative cancellation, restart/overwrite arguments, structured domain
  failures, and `Path` results. `Scenario.psr` now uses the Python product;
  historical managed-wrapper tests call that transitional wrapper directly.
- Replaced mandatory Python.NET and prototype HDF5 dependencies in package
  metadata with the validated Numba CPU runtime. SpiceyPy remains a core
  dependency because generated Sun/Earth vectors are part of the product
  scope, but it is imported lazily so explicit-vector operations do not load
  SPICE. No declared dependency or extra installs Python.NET, and HDF5 is not
  advertised because no public product writes HDF5.
- Adopted the initial Python-only public product API decisions: downstream
  products use keyword-only `backend="auto"` with strict `cpu` and `cuda`
  behavior, long operations default to `verbose=False`, and file-producing
  operations return `Path`. Added lazy `ls.cuda.is_available()` and
  `ls.cuda.status()` diagnostics without importing Numba or initializing CUDA
  during `import lunarscout`. Added the domain exception hierarchy that will
  replace transitional `Native*` failures in promoted product wrappers.
- Added durable compute-backend provenance to private staged products. Backend
  identity advances only with flushed, journaled valid patches; resumed jobs
  may truthfully accumulate CPU and CUDA execution, repair a stale manifest
  sidecar from the authoritative journal, and publish the ordered backend set
  in `LUNARSCOUT_COMPUTE_BACKENDS` GeoTIFF metadata. The accepted bounded PSR
  writer and 16-patch checkpoint behavior remain unchanged.
- Added reusable caller-owned horizon decode buffers and selected five pinned
  buffers for the private bounded CUDA PSR pipeline. `.cbin` data decodes
  directly into pinned memory, avoiding both per-patch 94 MiB allocations and
  an extra host copy; static metadata and reduced vectors remain resident after
  their first upload. A complete 1,599-patch run improved from `73.448` to
  `62.690` seconds and from `21.7704` to `25.5064 patches/s`, with identical
  values, masks, metadata, file size, and batched-file SHA-256. CPU and serial
  paths retain ordinary allocation and do not initialize CUDA.
- Replaced per-patch staged-TIFF reopen/close and journal writes in the private
  PSR path with a bounded 16-patch writer checkpoint. TIFF data is closed and
  synchronized before the journal advances, progress remains durable, and an
  interruption can require recomputing at most 16 unjournaled patches. A
  complete 1,599-patch run improved the retained four-reader pipeline from
  `80.184` to `73.448` seconds and from `19.9417` to `21.7704 patches/s`.
  Values, masks, metadata, and file size match; the physical TIFF hash changes
  because the open writer produces a different block layout.
- Added opt-in private PSR pipeline instrumentation separating horizon lookup,
  compressed read, decompression, host preparation, individual CUDA transfers,
  device-event kernel time, synchronization, result copy, TIFF write/close,
  TIFF synchronization, durable journal persistence, patch total, and pipeline
  total. A matched 16-patch all-valid Mons Mouton measurement found `.cbin`
  decompression dominant at `136.432 ms` of `183.680 ms` per durable patch;
  control and instrumented products have identical pixels, masks, metadata,
  and file SHA-256, with effectively zero measured instrumentation overhead.
- Added a bounded private PSR reader/CUDA/writer pipeline with an exact
  capacity-two decoded-horizon bound and one-item writer queue. The compiled
  `.cbin` decoder releases the GIL for overlap without changing arithmetic. A
  complete matched 1,599-patch Mons Mouton run improved from `299.925` to
  `250.052` seconds and from `5.3313` to `6.3947 patches/s`; pixels, masks,
  metadata, file bytes, and SHA-256 match exactly. Cancellation/resume and
  simulated calculation/writer failure draining preserve durable journals and
  atomic publication.
- Added ordered bounded parallel `.cbin` decompression with thread-safe first
  compilation and exact decoded-buffer accounting. A one-to-four-worker matrix
  selects four readers and five total decoded slots as the private default. A
  complete matched 1,599-patch run improves from `286.654` to `80.184` seconds
  and from `5.5782` to `19.9417 patches/s`; values, masks, metadata, file bytes,
  and SHA-256 remain identical. The selected path uses `3.81` CPU cores,
  1.441 GB sampled peak RSS, and the same 348 MiB process GPU memory.
- Added a runnable Python/Numba Mons Mouton PSR example using 108,113 exact
  Moon-ME Sun vectors and explicit CUDA execution. Its optional progress
  callback receives the monotonic fraction of durably completed horizon tiles;
  the example closure reports percentage, elapsed and remaining minutes, and
  estimated local completion time with resume-aware rate calculation. The
  complete 1,599-patch run finished in `306.45` seconds at `5.218 patches/s`
  and wrote 467,099 PSR pixels with no invalid pixels. About five percent GPU
  utilization, limited CPU use, and the much shorter retained warm-kernel time
  identify substantial opportunity for bounded decompression, CUDA, and writer
  overlap, batched durable checkpoints, asynchronous transfers, and
  multi-patch GPU scheduling.
- Unified private downstream backend and operational behavior. PSR now accepts
  explicit `auto`, `cpu`, and `cuda` selection, automatically falls back to
  its CPU implementation only for `auto`, and preserves explicit CUDA
  capability failures. Safe-haven generation now reports immediately flushed
  patch progress, checks cancellation across horizon reads and streamed time
  calculations, leaves resumable staging state, and resumes at the durable
  patch boundary. Forced-unavailable-CUDA tests now cover automatic CPU
  fallback for lightmaps, PSR, safe havens, and mission-duration products.
- Added separate private Sun- and Earth-elevation products. Each streams the
  shared CPU/CUDA body-center local-horizon-margin calculation into one
  timestamped `float32` BigTIFF band per vector, with bounded time batches,
  `auto` CPU fallback, masks, cancellation, and durable patch-level resume.
  Complete CPU and CUDA file pipelines agree in the explicit GPU test.
- Added private landed mission-duration products for four independently named
  conditions: sunlight fraction, Sun-center local-horizon margin, sunlight plus
  Earth-center margin, and Sun-center plus Earth-center margins. Inclusive
  thresholds, explicit candidate-start intervals, following-sample time
  ownership, right-censoring at the evaluation stop, irregular sample spacing,
  `float32` hour/day output, interval helpers, per-band interval metadata,
  missing-horizon masks, cancellation, and durable resume are covered. Shared
  CPU and CUDA lightmap sessions now also emit body-center local-horizon margin.
  A matched two-year real-terrain benchmark covers all four products on CPU and
  CUDA, records bounded memory and sparse threshold-amplified deltas, and
  optimizes margin-only calculation and inactive interval-state handling.
- Defined the initial private safe-haven reference semantics. Earth outages are
  maximal half-open intervals below the center-view threshold, all samples are
  included in the longest low-Sun run, and the first minimum-Earth sample
  supplies the interval timestamp. Default results are `float32` hours without
  the legacy final-sample omission, integer-hour truncation, or 255-hour clamp.
  A time-axis-bounded online reducer now consumes unquantized fraction batches
  from either compiled CPU or CUDA, with `auto` fallback, and writes one
  timestamped resumable BigTIFF band per Earth outage.
- Required CPU fallbacks for all downstream horizon-consuming products when
  NVIDIA CUDA is unavailable, while retaining CUDA-only production horizon
  generation. The private lightmap pipeline now accepts `auto`, `cpu`, and
  `cuda`, and its reusable CUDA session bounds output by configurable time
  batches. CPU and CUDA match the production C# `BuilderSunFraction` byte
  oracle. A Numba-parallel bounded CPU backend provides automatic fallback. In
  a four-patch, 2,921-band real-terrain BigTIFF run, CPU completed in 3.569
  seconds and CUDA in 2.178 seconds; all CPU/CUDA differences were one byte.
- Added the private reference/storage slice for time-series lightmaps. It ports
  the 16-slice C# `BuilderSunFraction` solar-disk calculation, encodes visible
  fraction as truncating `uint8(255 * fraction)`, and processes horizons
  patch-first while lazily yielding one 128 by 128 tile per time. The staged
  BigTIFF writer writes each yielded tile directly to its timestamped band, so
  neither a patch time cube nor a regional time cube is retained. Initial tests
  cover full, half, and zero illumination, timestamp metadata, band interleave,
  missing-patch invalid payloads, masks, and partial output edges. Numba CUDA
  time batching and a C# numerical oracle are now included in the follow-on
  Phase 6B work described above.
- Added the first private Phase 6B downstream-product vertical slice. Python
  now reads complete `.bin`/`.cbin` horizon tiles, accepts explicit timestamped
  Moon-ME vectors or lazily generates geometric SpiceyPy vectors, reproduces
  the five-viewpoint PSR reduction, and computes PSR with CPU-reference and
  reusable-buffer Numba CUDA paths. A dtype-generic staged BigTIFF store writes
  128 by 128 band-interleaved tiles with UTC band metadata, configurable
  invalid payloads, validity masks, durable per-patch restart journals, and
  partial-edge support. Deterministic Python CPU and Numba CUDA outputs match
  the actual C#/ILGPU PSR kernel byte-for-byte, including compressed-horizon
  quantization. A fresh-process compressed-horizon-to-GeoTIFF run loads no
  Python.NET, CLR, or moonlib modules. Real SPICE evidence now covers 108,113
  six-hour samples from 1970 through the start of 2044. Exact per-timestamp
  `utc2et` conversion remains the default, including for future mission
  periods after all published leap seconds. An explicit anchored-linear mode
  exactly reproduces C#, but is selected only where equivalence to `utc2et` is
  demonstrated for the intended calculation. That equivalence was established
  for the retained 16-patch real-terrain PSR product: both modes produced
  byte-identical results at about 1.19 patches/second. The downstream
  scheduler was still serial at that initial slice; the later entries above
  record the lightmap, safe-haven, and mission-duration implementations.
- Expanded the Python/Numba replacement evaluation to require a shared bounded
  downstream product pipeline for time-series lightmaps, optimized Metonic PSR,
  safe-haven maps, landed mission-duration maps, and dtype-generic
  horizon/vector reductions. The new gate requires a patch-major pipeline that
  loads one horizon tile, computes a 128 by 128 tile for each requested time,
  and writes it to that time's timestamped band in a tiled compressed BigTIFF.
  It also requires high-level SpiceyPy vector generation with explicit-vector
  override, full-DEM validity masks with configurable deterministic invalid
  payloads, durable per-patch restart journals, file staging, C#-oracle parity
  where available, and fresh-process execution without Python.NET or moonlib
  before downstream C# product code can be retired.
- Added the first private Phase 6 Numba horizon production pipeline: row-major
  full and partial patch enumeration, structurally validated skip/resume,
  bounded CPU preparation and CUDA work queues, cancellation boundaries,
  immediately flushed progress, fixed-contract partial-edge padding, and
  staged atomic `.bin`/`.cbin` writes. Python compressed files are readable by
  moonlib and existing Scenario readers; a 23,592,960-value real tile has at
  most `0.0007639` degrees of expected compression quantization error. In a
  matched warm four-patch, four-DEM compressed run, the serial pipeline reaches
  `0.1385` patches/second and the one-item-ahead pipeline reaches `0.1673`
  patches/second. A sustained 16-patch run with a bounded writer queue and
  reusable device buffers reaches `0.1793` patches/second, 63.8 percent of the
  matched C# throughput. It uses `4,458` MiB peak GPU memory and `9.02` GB peak
  host RSS. Preparation remains fully hidden after initial pipeline fill, so
  neighboring segment-cache reuse is intentionally deferred. Two- and
  four-stream runs produce byte-identical files but no material throughput gain,
  so the selected default remains one stream. Cross-process Numba disk caching
  reduces combined first CPU/CUDA-call time by `6.54` seconds with a `2.33` MB
  cache and falls back safely when no writable cache locator exists. The full
  failure matrix remains production-pipeline work; the implementation is still
  private.
- Added diagnostic Numba CUDA mechanics on a real GPU, including lazy device
  selection, launch/copy synchronization, pixel/azimuth indexing, C#-matching
  arithmetic helpers, fixed-step and adaptive level-0 traversal, exact
  factor-four max pyramids, and hierarchical traversal with C#/CPU/CUDA traces.
  The prototype pins NVIDIA's external CUDA target and a driver-compatible CUDA
  12.9 toolchain. It records narrow terrain skipped by the primary-DEM
  adaptive step floor. An inherited hierarchy defect at bilinear cell
  boundaries is corrected in both C# and Numba with four-cell culling bounds
  and boundary-capped level-0 steps. Production-shaped device subpatch
  interpolation, full and partial patches, multi-resolution DEM accumulation,
  and final degree buffers now match the selected C# fixtures. A
  hierarchy-enabled 16 by 16 LOLA patch differs by at most `5.9605e-8`
  degrees across 368,640 values. A bounded real two-DEM stack differs by at
  most `4.0412e-5` degrees, about 124 times below the adopted `0.005` degree
  angular acceptance limit; a uniform solar-disk model bounds the corresponding
  sunlight-fraction difference at `1.0291e-4`. The fixture also exposed a
  prototype orchestration bug, now corrected, where later passes did not carry
  prior horizon slopes into hierarchy culling as production C# does.
  Phase 5 preserves ILGPU's pixel-fast warp organization in a 256-thread Numba
  launch, removes local interpolation storage and most unintended float64 PTX,
  retains immutable pyramids on the GPU, and overlaps CPU segment preparation
  with CUDA execution. On a matched four-patch RTX 5090 Laptop benchmark, warm
  CUDA latency is `5.146` seconds, pipelined throughput is `0.1635` patches per
  second, peak GPU memory is `5,558` MiB, and peak host memory is `8.95` GB.
  These are respectively 1.196 times C# bounded wall time per patch, 70.3% of
  C# throughput, 1.141 times C# GPU memory, and 63.4% of C# host memory, passing
  all provisional Phase 5 gates. File output and product integration remain
  unimplemented.
- Ported the experimental horizon host-side geometry to Python/NumPy and an
  optional Numba CPU path, with C# oracle parity for sampling, polynomial ray
  segments, multi-DEM continuity, subpatch halos, deterministic cache reuse,
  real-terrain fitted paths, and bounded performance/memory evidence. This
  completed preprocessing Phase 3 before the diagnostic CUDA work began.
- Defined the private Python/NumPy horizon host/device data contract, including
  checked precision conversions, dense segment tensors, flattened pyramid
  storage, kernel/configuration validation, slope-buffer semantics, verified
  oracle loading, and Phase 1 artifact round-trip tests. CUDA and horizon
  algorithm implementation remain deferred to later evaluation phases.
- Established Phase 0 and Phase 1 evidence for evaluating a full Python/Numba
  horizon-generator port: reproducible C# production baselines and warm
  multi-patch benchmarks, GPU-memory sampling, independent reference-ray
  oracles, synthetic and real-terrain fixture manifests, immutable NPZ test
  artifacts, per-DEM and final horizon buffers, and selected CUDA hierarchy
  traversal traces. No horizon algorithm has been ported yet.
- Created standalone Lunarscout repository skeleton from Lunar Analyst packages/lunarscout.
- Replaced the internal architecture guide with a draft user guide covering purpose, installation, usage, maturity, architecture, examples, and roadmap stubs.
- Added a Lunarscout-specific `requirements.in` and removed inherited Lunar Analyst application dependencies.
- Updated package metadata so core installs include raster/geospatial, native bridge, and HDF5 dependencies.
- Verified the new local virtual environment with `pytest -q`: 195 passed, 1 skipped.
- Added map product catalog support, including product/region models, catalog loading, ordered text search, scenario-safe naming, download-directory resolution, file download helper, public exports, and tests.
- Added lunar map product support assets and utility scripts, including a south-pole overview GeoTIFF, a product catalog JSON file, catalog maintenance/download scripts, and Git LFS tracking for `data/product_overview.tif`.
- Added `ls.GenerateHorizons(...)` and `ls.native.GenerateHorizons(...)` as Python.NET wrappers around `QuadTreeHorizonGenerator.GenerateHorizonsForPatches`, with skip-existing patch filtering, compression selection, progress callbacks, cancellation checks, and tests.
- Added `scripts/run_generate_horizons.py` as an editable local runner for manually validating native horizon generation.
- Added `AGENTS.md` with project-specific guidance for future coding agents.
- Updated the native MaxRev GDAL package-version test to locate the extracted Lunarscout repository layout and scan `native/` projects.
- Added SPICE-backed lunar local-frame APIs for `LonLat`, inclusive datetime iteration, Sun/Earth NED vectors, azimuth/elevation histories, pandas DataFrames, and matplotlib elevation plots.
- Added `ls.spice` kernel helpers for furnishing, lazy default loading, reload/unload/clear state, NAIF kernel download/cache, generated meta-kernels, and SHA-256 verification from the checked-in default kernel manifest.
- Added default SPICE kernel manifests under `data/spice/` and package data, plus implementation tracking in `docs/spice-local-frame-api-plan.md`.
- Updated the user guide with SPICE kernel setup, local NED frame conventions, vector/angle APIs, DataFrame helpers, and plotting examples.
- Added `scripts/get_link_tree.py` for recursive same-site HTML link discovery.
- Updated the lunar map product GUI to display overview-map CRS and longitude/latitude mouse coordinates and copy either coordinate pair to the clipboard with keyboard shortcuts.
- Changed the canonical Scenario horizon directory from `lighting/horizons` to root-level `horizons`, with matching tests, examples, README, and guide updates.
- Added Scenario canonical path helpers for `root_path()`, `hillshade_path()`, `slope_path()`, `aspect_path()`, and `roughness_path()`.
- Added native GDAL-backed terrain product generation through `Scenario.create_hillshade()`, `create_slope()`, `create_aspect()`, and `create_roughness()`, backed by a new `moonlib.TerrainProducts` static helper.
- Added `Scenario.generate_horizons()` as a scenario-aware wrapper around native horizon generation, including `dem_paths` and `surrounding_dems` handling with scenario-relative DEM resolution.
- Added Python horizon tile access helpers on `Scenario`, including patch coordinate conversion, horizon file lookup, `.bin`/`.cbin` single-pixel horizon reads, one-file handle caching, and cache close support.
- Added Scenario longitude/latitude DEM pixel lookup and horizon plotting helpers, including centered azimuth windows and empty azimuth/elevation axes.
- Added Scenario Sun/Earth horizon plot overlays for center points, apparent limb markers, center paths, and upper/lower limb paths; Sun defaults to gold and Earth defaults to blue.
- Suppressed the noisy pyproj WKT-to-PROJ.4 information-loss warning during Lunarscout GeoTIFF metadata reads.
- Expanded `docs/USER_GUIDE.md` with Scenario path, native terrain, horizon generation/access, horizon plotting, and Sun/Earth overlay examples.
- Updated SPICE body geometry helpers and plotting helpers to accept `TimeRange` values returned by `ls.times(...)` directly.
- Changed body path limb rendering from upper/lower dashed lines to a translucent filled limb band, while preserving explicit Matplotlib style overrides.
- Added `ls.body_azimuth_elevation_over_horizon(...)` and `Scenario.body_azimuth_elevation_over_horizon(...)` for azimuth plus elevation over an interpolated 1440-sample horizon.
- Added `horizon=` support to `ls.plot_body_elevations(...)` and added `Scenario.plot_body_elevations(...)` with `over_horizon=True` to fetch scenario horizons automatically.
- Expanded `docs/USER_GUIDE.md` with a succinct function overview table covering root functions, Scenario methods, and file-backed temporal object methods.
