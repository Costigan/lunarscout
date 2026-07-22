# Map-Algebra Writer Lifecycle Implementation Status

Date: 2026-07-22

## Scope

This pass implements and repairs lifecycle control for bounded `ma.write()`:
monotonic progress, cooperative cancellation, compact durable checkpoint
journaling, safe resume, staged-output validation, and paired publication
rollback. A semantic review found and fixed stale-stage reuse, retained-journal
corruption, duplicate completion, manifest poisoning, and missing integer
nodata tags before this report was finalized.

## Files Changed

- `src/lunarscout/map_algebra/_planner.py`: enforced journal-identity inputs and
  lifecycle capability metadata.
- `src/lunarscout/map_algebra/_windowed.py`: callback/cancellation support for
  the private executor without duplicate completion.
- `src/lunarscout/map_algebra/_writer.py`: compact journal, stage coupling and
  validation, checkpoints, cancellation, progress, and paired publication.
- `src/lunarscout/map_algebra/expression.py`: public `ma.plan()` lifecycle and
  default journal-identity diagnostics.
- `tests/map_algebra/test_writer.py`: publication rollback and retry coverage.
- `tests/map_algebra/test_writer_lifecycle.py`: lifecycle and fault-injection
  coverage.
- `docs/USER_GUIDE.md`, `docs/ARCHITECTURE.md`,
  `docs/map-algebra-implementation-plan.md`, and `CHANGELOG.md`: reconciled
  public contract, architecture, status, and evidence.

## Public API Decisions

`ma.write()` retains its existing arguments and adds optional keyword-only
`progress_callback`, `cancellation_requested`, and `checkpoint_interval=16`.
Callbacks must be callable or `None`; checkpoint intervals must be positive
integers. Invalid lifecycle arguments raise structured map-algebra errors before
output modification.

Progress callbacks receive `(completed, total, window_idx)`. Newly executed
windows are reported after their value and mask writes return, counts increase
monotonically, and completion is reported exactly once. The final newly
executed window is durably checkpointed and retains its real zero-based index.
A fully checkpointed resume uses `window_idx=-1`
because no window was recomputed. Callback exceptions propagate without
publishing and leave matching restart artifacts available.

Cancellation is checked before execution and before each uncompleted window.
Prior written windows are checkpointed before `OperationCancelledError` with
code `map_algebra_cancelled` is raised. The error details include completed and
total window counts and, during execution, the next window index.

## Journal Format and Identity

Journal format 2 stores constant-size completion state:

```json
{
  "journal_format": 2,
  "identity": "sha256:<hex>",
  "layout": "row_major_contiguous_prefix",
  "completed_windows": 16,
  "total_windows": 81
}
```

The identity binds the expression scientific identity, output dtype and fill,
complete destination CRS/affine/dimensions, row-major window layout, checkpoint
interval, validity encoding, and enforced GeoTIFF driver, tiling, block,
compression, predictor, and BigTIFF options. The staged TIFF carries the same
identity. Resume also
validates its driver, band count, dtype, CRS, transform, nodata, dimensions,
and block layout. A missing, malformed, truncated, stale, out-of-range, or
incompatible journal—or a missing/incompatible TIFF—causes both artifacts to
be discarded and all windows to be recomputed.

## Failure and Crash Guarantees

- A window enters the journal only after its values and GDAL validity mask have
  been written and the TIFF has closed successfully.
- JSON writes handle short writes, fsync the file, atomically replace the old
  journal, and attempt to fsync the parent directory.
- Uncheckpointed TIFF data is never inferred as complete; it is recomputed.
- Callback, kernel, value-before-mask, and journal-update failures preserve a
  usable matching stage/journal when one exists.
- TIFF and manifest publication use staged files, backups, and paired exception
  rollback. A publication exception restores both the previous output and its
  previous manifest while retaining the complete stage for retry.
- Deterministic backups left by an interrupted rename sequence are rolled back
  on the next call before window execution resumes.
- Successful publication removes restart and backup artifacts.
- These tests exercise injected Python/GDAL call failures. They do not claim a
  multi-file atomic transaction across an operating-system or power crash
  between the TIFF and manifest renames.

## Planner Diagnostics

`ExecutionPlan` reports total windows, `journal_available`,
`supports_progress`, `supports_cancellation`, the enforced journal identity or
its complete inputs, and `resumable_stages=("windowed_execution",)`. Publication
is deliberately not advertised as resumable.

## Verification

- Focused writer tests: **72 passed**.
- Map-algebra tests: **820 passed**.
- Full ordinary CPU suite: **1278 passed, 17 skipped**.
- `git diff --check`: clean.

The exact broader totals above are recorded after the final verification run.

## Limitations and Deferred Work

- Concurrent processes writing the same destination have no locking contract.
- A process/power crash between the two final TIFF/manifest renames is not a
  transactional multi-file commit; normal raised exceptions are rolled back.
- General focal statistics/convolution/morphology window kernels and arbitrary
  footprint-derived halos remain deferred.
- Local fusion, region adapters, bounded global/zonal/distance execution,
  temporal spatial-window/time-batch mapping, and empirical resource-scaling
  evidence remain deferred.
- TestPyPI publication remains intentionally skipped by project decision.

## Next Dependency-Ordered Task

Implement general focal statistics, convolution, and morphology window
execution with footprint-derived asymmetric halos, preserving eager/windowed
parity and the lifecycle guarantees above.
