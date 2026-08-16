# Lunarscout Test Coverage and Specificity Audit

Generated: 2026-08-05

## Scope and Scale

| Metric                                               | Count   |
| ---------------------------------------------------- | ------- |
| Source modules (`.py` files under `src/lunarscout/`) | 73      |
| Source lines (excluding blank/comments)              | ~32,000 |
| Test files                                           | 55      |
| Test functions (`def test_*`)                        | 1,324   |
| Collected test items (`pytest --co -q`)              | 1,852   |
| Test data files (`tests/data/`)                      | 6       |

Test files are organized in three directories matching the source structure:
`tests/` (19 files, 264 test functions), `tests/map_algebra/` (21 files, 927 test
functions), and `tests/numba_horizon/` (15 files, 133 test functions).

## Overall Assessment

The test suite is well-organized and thorough in its core domains. Every public
source module has at least indirect test coverage, and most have dedicated test
files. The `numba_horizon` tests use a robust oracle pattern: static pre-captured
reference data (generated earlier from a C# tool, stored as `.npz`/`.json` in
`tests/data/numba_horizon/`) verified with cryptographic hashes. No .NET
runtime is required to run the tests — `load_reference_artifact` is pure NumPy. The `map_algebra`
tests systematically verify eager/expression parity. The top-level tests
exercise the public `ls.*` API surface and validate import-time invariants
(e.g., no unintended dependency pulls, no working-directory writes).

**Principal weaknesses:**

1. Two exported public API functions (`download_map_product`, `sunlight_fraction`)
   have zero direct test coverage.
1. Six internal implementation modules with nontrivial logic have no dedicated
   tests and are exercised only indirectly through public wrappers.
1. `ProgressEvent` has no unit tests for its fields, construction, or serde.
1. Platform- and filesystem-fault error-recovery paths are absent.
1. The `__init__.py` `__all__` list is not validated for completeness or
   staleness.

______________________________________________________________________

## 1. Public API Functions with Zero Direct Coverage

### 1.1 `ls.download_map_product`

- **Source**: `src/lunarscout/map_products.py:209`
- **Exported**: `ls.__all__` line 208
- **Tests**: None. No test file calls `download_map_product`.
- **Risk**: Network-dependent product downloads are entirely untested. Callers
  may encounter HTTP errors, partial downloads, or corrupted files with no
  test-validated error handling.
- **Recommendation**: Add at least one test with a mocked HTTP session to
  verify the download flow and error codes (`ProductCatalogError` on 4xx/5xx
  responses, file cleanup on failure). The function is used by
  `scripts/lunar_map_products.py` but never exercised in the test suite.

### 1.2 `ls.sunlight_fraction`

- **Source**: `src/lunarscout/spice_geometry.py`
- **Exported**: `ls.__all__` line 240
- **Tests**: Never called directly in any test file. Only referenced indirectly
  through the `sunlight_fraction_threshold` parameter of other functions.
- **Risk**: The function may return incorrect values, raise unexpected
  exceptions, or fail silently on edge-case inputs (0%, 100%, NaN inputs).
- **Recommendation**: Add tests for 0.0, 0.5, 1.0 fractions and invalid inputs.

______________________________________________________________________

## 2. Public API Names Not Validated in `__all__`

- **Source**: `src/lunarscout/__init__.py`
- **Problem**: No test verifies that every name in `__all__` resolves to an
  importable object, or that every imported public name appears in `__all__`.
  Stale entries accumulate without detection.
- **Risk**: `__all__` can drift from the actual import set; linting tools that
  respect `__all__` will have incorrect completions.
- **Recommendation**: Add a test that iterates `ls.__all__`, checks each name
  exists as an attribute of `ls`, and verifies the attribute is not `None` or
  a module stub. Also compare `set(ls.__all__)` against the set of names
  imported in `__init__.py` to flag stale entries.

______________________________________________________________________

## 3. ProgressEvent — No Unit Tests

- **Source**: `src/lunarscout/progress.py` (27 lines)
- **Tests**: `ProgressEvent` appears in 7 test files but only as a callback
  argument type, never constructed or inspected directly.
- **Missing**:
  - Construction with valid and invalid field values
  - `fraction` invariant (0.0 \<= fraction \<= 1.0)
  - `backend` None vs "cpu" vs "cuda"
  - Dataclass immutability / frozen contract
  - Equality and hashing behavior
- **Risk**: Low (small dataclass), but the frozen/slots contract is untested.

______________________________________________________________________

## 4. Internal Modules with No Dedicated Tests

These modules contain significant logic (combined ~1,500 lines) but are only
exercised indirectly through public wrappers. When public function behavior
changes, regressions in these internals may go undetected.

| Module                          | Lines | What it does                                            | How it is tested                                              |
| ------------------------------- | ----- | ------------------------------------------------------- | ------------------------------------------------------------- |
| `map_algebra/_dtypes.py`        | 797   | Overflow promotion, dtype result tables, safe casting   | Only through `normalize_overflow` dispatch in `__init__.py`   |
| `map_algebra/_eager.py`         | 227   | Eager evaluation dispatch for operations                | Only through public `ma.compute()`                            |
| `map_algebra/_kernels.py`       | 192   | Focal/convolve kernel implementations                   | Only through `ma.focal_*()` public API                        |
| `map_algebra/_normalization.py` | 94    | Data normalization helpers (`compute_data_range`, etc.) | Only `normalize_canonical` tested directly                    |
| `map_algebra/_units.py`         | 133   | Unit metadata propagation, cross-unit validation        | Only through `add()` / `multiply()` unit checks               |
| `map_algebra/_validity.py`      | 76    | Numeric error handling (`normalize_numeric_errors`)     | Only through `__init__.py` dispatch                           |
| `map_algebra/_validation.py`    | 103   | Argument validation, grid checking                      | Only `_as_raster_operand` and `_as_expression_operand` tested |

**Recommendation**: The `_dtypes.py` module at 797 lines is the most concerning.
Its overflow promotion tables should have focused tests for:

- Integer overflow promotion chains (uint8→uint16, int8→int16, etc.)
- Float→int promotion policy
- Safe vs unsafe casting boundaries
- Built-in vs extended integer coverage (uint64, int64 interactions)

For the other modules, adding at least a smoke test per public function would
provide regression detection.

______________________________________________________________________

## 5. `_cuda_runtime.py` — Minimal Coverage

- **Source**: `src/lunarscout/_cuda_runtime.py` (25 lines)
- **Tested**: Only `CUDA_INSTALL_HINT` string content (in `test_cuda_status.py`)
- **Not tested**: `import_numba_cuda()` function (imports `numba` and
  `numba.cuda`)
- **Risk**: Low (25-line shim), but the import logic could break on Numba
  version upgrades.

______________________________________________________________________

## 6. Missing Error-Recovery Tests

### 6.1 File I/O Failure Paths

- **Scenario**: Disk full during `write_geotiff`, `write_temporal_cube`,
  `TemporalGeoTiffSeriesWriter`, or `write_max_pyramid_cache`.
- **Coverage**: None. No tests inject `OSError` or `IOError` during writes.
- **Recommendation**: The `map_algebra/_writer.py` tests already cover
  overwrite-protection and restart-id logic. Extend with filesystem-
  error-injection tests via `monkeypatch` on `rasterio.open`/`np.tofile`.

### 6.2 Network Failure Paths

- **Scenario**: SPICE kernel download timeouts, HTTP errors.
- **Coverage**: None. `download_map_product` has zero test coverage.
- **Recommendation**: Mock `urllib.request` or `requests` to inject timeouts,
  partial responses, and HTTP error codes.

### 6.3 Pipeline Cancellation

- **Scenario**: User-provided callback raises `OperationCancelledError` during
  horizon generation or lightmap computation.
- **Coverage**: Only `test_public_lightmap.py` has a `cancel_after_first`
  callback test (line 1200). No cancellation test for horizon generation
  pipeline or for temporal cube writing.
- **Recommendation**: Add cancellation tests for `generate_horizons`,
  `write_temporal_cube`, and `TemporalGeoTiffSeriesWriter.write_cube`.

______________________________________________________________________

## 7. Input Validation Edge Cases — Untested

### 7.1 `Raster` Constructor

- **Missing**: Non-finite values in valid mask, mismatched valid/value shapes
  where one dimension matches but the other doesn't, units=None propagation,
  zero-size arrays, arrays with >2 dimensions.
- **Current**: `map_algebra/test_raster.py` tests basic construction and
  validation. Does not test:
  - Zero-width or zero-height raster
  - valid mask with dtype other than bool
  - valid mask containing NaN or None values
  - GeoReference with negative width/height

### 7.2 `GeoReference` Constructor

- **Missing**: WKT/proj4 mismatch detection, degenerate affine transforms
  (zero pixel size), coordinate bounds for rasters crossing the ±180°
  longitude line.
- **Current**: `test_georeference.py` tests non-invertible affine and
  inconsistent pixel size. Missing:
  - Affine with zero determinant but non-obvious linear dependence
  - Navigation at extreme latitudes (±90°)
  - `pixel_to_lonlat` at numerically challenging points

### 7.3 `TemporalCube` / `TemporalRaster`

- **Missing**: Unsorted times, duplicate times, empty time dimension,
  non-UTC timezone-aware datetimes, times spanning DST transitions.
- **Current**: `test_temporal.py` tests basic construction and arithmetic.
  Missing:
  - `times` array with dtype `datetime64[ns]` (currently uses `datetime64[D]`)
  - Time ranges that span a leap second

______________________________________________________________________

## 8. Platform and Environment Gaps

- **Windows**: No CI or test matrix for Windows. Path handling uses `Path`
  throughout (good), but no tests verify UNC paths or drive-letter handling
  in scenario operations.
- **Python version matrix**: CI seems to target a single Python version
  (`.github/workflows/ci.yml`).
- **Big-endian**: No tests for big-endian platforms. Bit-level file format
  tests (`_numba_horizon/file_format.py`) assume little-endian.
- **Numba JIT cache**: No tests for cached vs uncached JIT compilation paths.
  The `_numba_horizon` modules use `@njit` extensively; stale cache behavior
  is not tested.

______________________________________________________________________

## 9. Test Specificity Observations

### Strengths

- **Reference oracle tests**: The `numba_horizon` tests use SHA-256 hashes
  to validate pre-captured C# reference data. Phase-by-phase contracts
  ensure each processing stage matches the C# reference at high precision
  (atol ~1e-7 for floats).
- **Eager/expression parity**: `map_algebra` tests systematically compare
  eager and expression evaluation paths for most operations, ensuring the
  lazy execution engine produces identical results to direct computation.
- **Dependency boundary**: `test_dependency_boundary.py` uses a fresh
  subprocess to verify that `import lunarscout` does not load Numba, SpiceyPy,
  or .NET runtimes, and does not write to the working directory.
- **Error code taxonomy**: Every exception class has a stable `code=` field,
  and tests verify the correct code string is raised for each error condition.
- **Example scripts as integration tests**: `test_examples.py` runs selected
  example scripts in subprocesses and validates their outputs.

### Weaknesses

- **Parametrized breadth vs depth**: Some operations are tested with only two
  or three parameter combinations. For example, `reclassify_values` tests
  `uint8` input but not `uint16`, `uint32`, `uint64`, or `int64` inputs.
  `align` tests `nearest` resampling extensively but `bilinear`, `cubic`,
  `cubicspline`, `lanczos`, `average`, and `rms` are only tested via the
  smoke "is usable" parametrized test.
- **Single-pixel raster edge cases**: Many tests use 2×2 or 3×3 rasters.
  Single-pixel (1×1) and single-row/single-column rasters are rarely tested.
- **All-valid / all-invalid / all-NaN**: Only `normalize_minmax` and
  `standardize` test the all-invalid case. Other operations (addition,
  multiplication, focal, zonal) do not test all-invalid or all-NaN inputs.
- **Floating-point edge cases**: ±inf, ±denorm, and signaling NaN values
  are not systematically tested across operations. Only a few pyramid tests
  inject NaN/±inf into elevation arrays.
- **Integer boundary values**: `int64` min/max and `uint64` max are only
  tested in `test_alignment.py`. Other operations (`add`, `multiply`,
  `reclassify_values`) lack tests at these extremes.

______________________________________________________________________

## 10. Test File-to-Module Mapping

### Top-Level

| Test File                      | Primary Target Module(s)             | Test Functions |
| ------------------------------ | ------------------------------------ | -------------- |
| `conftest.py`                  | Shared fixtures                      | —              |
| `test_alignment.py`            | `alignment.py`                       | 12             |
| `test_cuda_status.py`          | `cuda.py`, `_cuda_runtime.py`        | 5              |
| `test_dependency_boundary.py`  | `__init__.py`, `pyproject.toml`      | 5              |
| `test_examples.py`             | Example scripts (integration)        | 3              |
| `test_georeference.py`         | `georeference.py`                    | 7              |
| `test_geotiff_io.py`           | `geotiff.py`                         | 14             |
| `test_map_products.py`         | `map_products.py`                    | 14             |
| `test_product_errors.py`       | `errors.py`                          | 4              |
| `test_public_horizon.py`       | `horizon.py`                         | 11             |
| `test_public_lightmap.py`      | `products.py`                        | 34             |
| `test_public_m2_validation.py` | `products.py` (M2 site)              | 12             |
| `test_regions.py`              | `regions.py`                         | 18             |
| `test_release_artifacts.py`    | `scripts/build_release_artifacts.py` | 4              |
| `test_scenario.py`             | `scenario.py`                        | 29             |
| `test_spice.py`                | `spice.py`, `spice_geometry.py`      | 21             |
| `test_temporal.py`             | `temporal.py`                        | 12             |
| `test_temporal_store.py`       | `temporal_store.py`                  | 17             |
| `test_terrain.py`              | `terrain.py`                         | 8              |

### Map Algebra

| Test File                        | Primary Target Module(s)            | Test Functions |
| -------------------------------- | ----------------------------------- | -------------- |
| `test_classify_coords.py`        | `coordinates.py`, `_validation.py`  | ~40            |
| `test_distance.py`               | `distance.py`                       | ~20            |
| `test_example_surface_public.py` | Integration (examples 18-21)        | 5              |
| `test_expression.py`             | `expression.py`, `_model.py`        | ~25            |
| `test_focal.py`                  | `focal.py`                          | ~30            |
| `test_local_ops.py`              | `local.py`                          | ~30            |
| `test_numeric_policy.py`         | `local.py` (overflow/saturate)      | ~25            |
| `test_planner_windows.py`        | `_planner.py`, `_windows.py`        | ~20            |
| `test_public_resample.py`        | `_spatial.py`                       | ~20            |
| `test_public_terrain.py`         | `_model.py` (terrain ops)           | ~15            |
| `test_raster.py`                 | `raster.py`                         | ~20            |
| `test_reductions_zonal.py`       | `reductions.py`, `zonal.py`         | ~30            |
| `test_regions.py`                | `regions.py` (map_algebra)          | ~15            |
| `test_registry_review.py`        | `_registry.py`                      | ~15            |
| `test_spatial_windows.py`        | `_spatial.py`, `_windows.py`        | ~20            |
| `test_stack_layers.py`           | `local.py` (stack)                  | ~15            |
| `test_temporal.py`               | `temporal.py`, `_temporal_model.py` | ~30            |
| `test_writer.py`                 | `_writer.py`                        | ~25            |
| `test_writer_lifecycle.py`       | `_writer.py`                        | ~15            |

### Numba Horizon

| Test File                              | Primary Target Module(s)                                                     | Test Functions |
| -------------------------------------- | ---------------------------------------------------------------------------- | -------------- |
| `test_phase1_real_terrain_fixtures.py` | External data validation                                                     | 3              |
| `test_phase1_reference_oracles.py`     | Reference data contract                                                      | 9              |
| `test_phase2_contract.py`              | `contract.py`                                                                | 10             |
| `test_phase3_geometry.py`              | `geometry.py`, `geometry_numba.py`                                           | 8              |
| `test_phase4_cuda_mechanics.py`        | `cuda_backend.py`, `kernel_math.py`, `fixed_step.py`                         | 5              |
| `test_phase4_hierarchy_safety.py`      | `hierarchy.py` (external script)                                             | 2              |
| `test_phase4_pyramid.py`               | `pyramid.py`, `hierarchy.py`                                                 | 12             |
| `test_phase4_subpatch.py`              | `subpatch.py`, `generator.py`                                                | 8              |
| `test_phase6_pipeline.py`              | `pipeline.py`, `file_format.py`                                              | 14             |
| `test_phase6b_elevation.py`            | `elevation_pipeline.py`                                                      | 6              |
| `test_phase6b_lightmap.py`             | `lightmap.py`, `lightmap_cpu.py`, `lightmap_cuda.py`, `lightmap_pipeline.py` | 7              |
| `test_phase6b_mission_duration.py`     | `mission_duration.py`, `mission_duration_pipeline.py`                        | 6              |
| `test_phase6b_product_store.py`        | `product_store.py`                                                           | 8              |
| `test_phase6b_psr.py`                  | `psr.py`, `psr_cuda.py`, `psr_pipeline.py`, `product_vectors.py`             | 9              |
| `test_phase6b_safe_haven.py`           | `safe_haven.py`, `safe_haven_pipeline.py`                                    | 6              |

## 11. Summary of Recommendations

### High Priority

1. Add tests for `ls.download_map_product` (mocked HTTP).
1. Add tests for `ls.sunlight_fraction` (direct calls with edge values).
1. Add a `__all__` validation test for `ls.__init__`.
1. Add focused tests for `map_algebra/_dtypes.py` overflow promotion tables.

### Medium Priority

5. Add constructor validation edge-case tests for `Raster` (zero-size, >2D
   arrays, non-`bool` valid mask).
1. Add constructor edge-case tests for `GeoReference` (zero pixel size, extreme
   latitudes).
1. Add unit tests for `ProgressEvent` dataclass invariants.
1. Add pipeline cancellation tests for `generate_horizons` and
   `write_temporal_cube`.
1. Test floating-point ±inf/NaN propagation across all map_algebra operations
   (not just pyramid).
1. Test integer boundary values (int64 min/max, uint64 max) across operations
   beyond alignment.

### Low Priority

11. Add filesystem-failure-injection tests (disk full, permission denied).
01. Add network-failure tests for SPICE kernel downloads.
01. Test big-endian byte order in `_numba_horizon/file_format.py`.
01. Add single-pixel and single-row/column raster edge-case tests.

______________________________________________________________________

## 12. Test Execution Notes

Run the full ordinary CPU suite:

```bash
.venv/bin/python -m pytest -q
```

Run specific domain tests during development:

```bash
.venv/bin/python -m pytest tests/test_public_horizon.py tests/test_public_lightmap.py -q
.venv/bin/python -m pytest tests/map_algebra/test_local_ops.py -q
.venv/bin/python -m pytest tests/numba_horizon/ -q
```

Real CUDA tests require `LUNARSCOUT_REQUIRE_NUMBA_CUDA=1` and a visible NVIDIA
device:

```bash
LUNARSCOUT_REQUIRE_NUMBA_CUDA=1 .venv/bin/python -m pytest \
    tests/numba_horizon/test_phase4_cuda_mechanics.py -q
```

Example integration tests:

```bash
.venv/bin/python -m pytest tests/test_examples.py -q
```
