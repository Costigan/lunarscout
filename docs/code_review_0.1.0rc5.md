# Lunarscout 0.1.0rc5: Pre-Release Code Review

> **Update 2026-10-09.** Part 2 horizon findings H-A, H-B, H-C, H-D, M-G and M-I are fixed on branch `fix/hierarchy-cull-bound` (not yet merged). See "Fix status" at the start of Part 2.

Date: 2026-10-08. Mode: read-only. Scope: public API and root namespace, map algebra and writer lifecycle, raster/temporal/SPICE/trajectory, and (second pass) the horizon calculation, CUDA kernels, tile pipeline, worker and deployment files. USRA Proprietary unless stated otherwise.

## Verdict

**Not ready to publish as-is.** Part 1 found one confirmed silent-wrong-result bug (integer overflow), several high-impact correctness defects and two data-loss paths. Part 2 (horizon) adds that the C# parity evidence does not establish correctness: the horizon kernel has algorithmic errors it inherits from C#, plus a pipeline hang and a resume that can silently reuse bad or foreign tiles. Crash consistency of `ma.write()` under hard kills held up well, and the lazy-import and dependency-boundary rules checked out. Part 2 starts at the heading "Part 2: Horizon calculation" below.

## Evidence and limits

- Part 1: three parallel reviewers read staged source and tests. Part 2: three more reviewers covered horizon geometry, CUDA kernels, and the pipeline/deploy files. I personally re-checked the highest-ranked findings against the code (marked **[verified by me]**), and ran `review_scripts/s1_cpu_repros.py` in my sandbox, where all 8 checks reproduced (marked **[reproduced by me]**). Items from a reviewer's own experiment are marked **[reproduced]**; all others are static reads.
- **No GPU was available.** No CUDA kernel was executed on hardware. GPU-related "reproduced" results used CPU emulation of the same algorithm or the Numba CUDA simulator, so they establish logic, not hardware behavior (races, FMA contraction, timing).
- **Part 1 staging gap, now closed:** I originally did not stage `_numba_horizon/`, `data/`, `LICENSE` and `deploy/`, and wrongly described them as absent from your folder. They exist, and Part 2 reviewed them. Consequently the Part 1 concern that the wheel might lack `_numba_horizon/` is largely resolved, except that the wheel inspector would not detect such a loss (H8).
- **Still not available to me:** `docs/USER_GUIDE.md`, `docs/ARCHITECTURE.md`, `docs/numba-horizon-phase-1-real-terrain-fixtures.json`, `docs/numba-horizon-phase-4-cuda-memcheck.json`, `examples/`, `tests/data/`, the trajectory tests, the C# capture project (`CSharpPhase4RealTerrainCapture.csproj`, which the phase-4 report tells you to run but which is not in `scripts/numba_horizon/`), and some of the other tests. A CPU-only run of `tests/numba_horizon` in the sandbox gave 107 passed, 23 failed, 14 errors; every failure I could check was a missing fixture file, not a code defect.
- Not run: GPU code, the full test suite, `python -m build`.

## Ranked findings

### BLOCKER

**B1. Signed-integer multiply overflow is missed (silent wrong result).** **[FIXED on `fix/b1-integer-multiply-overflow`, 2026-10-09: ceiling bounds in the three affected branches; exhaustive int8/uint8 and int16-int64 frontier tests. The int8 power path was not affected.]** `map_algebra/_dtypes.py:467-469`. The checks use floor division (`lower // aa`), which is off by one for inexact quotients when operand signs differ or both are negative. **[verified by me]** I enumerated all int8 pairs: 366 of 65,536 overflowing products go undetected (e.g. 3 × −43 = −129, and −1 × −128). **[reproduced]** `multiply(int8 raster [3], -43)` returned 127, valid=True, under the default `overflow="raise"`. The same mask feeds `checked_integer_power` and `square`, for every signed width. Fix: use ceil-based bounds or compute in a wider type; add exhaustive int8/int16 tests. Existing `test_numeric_policy.py` only checks max/min boundaries.

### HIGH

**H1. `align()` masks cross-CRS output with wrong coordinates.** **[FIXED, 2026-10-09: destination pixel centres are transformed into the source CRS before the coverage test; untransformable points count as outside]** `alignment.py:259-277`. **[verified by me]** The "outside source" mask converts the target grid's projected coordinates with `source_georef.projected_to_pixel` and no CRS transform. When source and target CRS differ (the docstring promises reproject/resample), valid pixels are overwritten with nodata. Example: south-polar-stereographic DEM aligned onto an equirectangular grid. No cross-CRS test exists. Fix: derive the mask from the `reproject` output or transform coordinates first.

**H2. Terrain functions are not Horn/GDAL-compatible.** **[FIXED, 2026-10-09: Horn 3x3 kernel with gdaldem's compute_edges rule, gdaldem hillshade mapping, signed pixel sizes, rotated and unscaled geographic grids rejected; parity tests against gdaldem on random relief]** `terrain.py:123-132`. **[verified by me]** `_gradient` uses `np.gradient` (2-point central difference), while docstrings say "GDAL-compatible Horn" and tests reference gdal. Tests use only planar surfaces, so the difference is invisible. On random relief the reviewer measured up to 3.5 difference in interior dz/dx (z std 5 m, 1 m pixels). Edge handling also differs (one-sided differences vs GDAL edge replication). Also: `abs()` on pixel sizes mirrors aspect on south-up grids, and no check that the CRS is projected (geographic grids give garbage slope). Fix: implement the Horn 3×3 kernel, or relabel as central-difference; add a non-planar parity test against `gdaldem`. This matters because slope drives landing-site screening.

**H3. Windowed non-nearest downsampling disagrees with `compute()`.** **[FIXED, 2026-10-09: bilinear/cubic/cubicspline/lanczos source windows are padded by support x ceil(downsampling ratio) + 1; parity tests for 14 methods, factors 4 and 7, windows 4-64]** `_spatial.py:344-407`, `_windowed.py:228-250`. **[reproduced]** GDAL widens bilinear/cubic/lanczos kernels on downsampling, but the window padding is a fixed 1-3 pixels. For 20 m → 80 m, `write()` differs from `compute()` by up to 2.5 (0-100 data) with seams at window edges for 4-128 px windows. Safe only for nearest, average, upsampling, or a single large window. Fix: pad by `support × ceil(scale)` or reject non-nearest downsampling in windowed mode; add a parity test.

**H4. Temporal local ops bypass the numeric policy.** **[FIXED, 2026-10-09: each temporal layer is evaluated by the spatial numeric core (`_eval_unary` / `_eval_binary`), with spatial dtype and unit inference at build time and `overflow=` / `numeric_errors=` accepted; parity tests across 5 dtypes and 35 operations]** `map_algebra/_temporal_model.py:1035-1064`. **[reproduced]** On uint8 temporal rasters: `t*100` wraps; `t-t` gives 255; `250+10` gives 4; `1.5 - t` truncates the scalar to 1; `t/0` gives inf and is marked valid; `sqrt(-1)` and `log(0)` give valid NaN/-inf. Spatial ops mark these invalid or raise. Fix: route through the same checked-integer and numeric-domain code.

**H5. `start_fresh=True` deletes the existing output without `overwrite=True`.** **[FIXED, 2026-10-09: start_fresh discards restart state only and restores an interrupted publication; replacing an output needs overwrite; an output that is also an input is rejected in preflight]** `map_algebra/_writer.py:438-445`. **[verified by me]** The code unlinks `output_path` and the manifest unconditionally. **[reproduced]** A cancellation immediately after leaves the user with nothing; with `output == source`, the input file is deleted before any window is read. Fix: delete only staging/journal artifacts; require `overwrite` before touching the output; reject output equal to any source.

**H6. `overwrite=True` on a temporal series can delete an unrelated directory.** **[FIXED, 2026-10-09: overwrite replaces only an empty directory or an existing Lunarscout series, checked before writing and again at commit]** `temporal_store.py:346-381`. **[verified by me]** `_commit_staging_directory` checks only that the destination is a directory, renames it to a backup, and `rmtree`s it after publish. `write_temporal_cube("/data/project", cube, overwrite=True)` would destroy `/data/project`. Fix: replace only an empty directory or one whose `manifest.json` has the expected format.

**H7. No `fsync` of staged TIFFs / series layers.** **[FIXED, 2026-10-09: ma.write fsyncs the staged TIFF before every journal checkpoint and the output directory after publishing; temporal series fsync each layer, the VRT, manifest and layers directory before COMPLETE, and the parent directory after commit]** `_writer.py:561-611`; `temporal_store.py:162-178, 510-520`. Only the JSON journal is fsynced. After power loss or on NFS/Ceph (the stated NRP target), a journal or COMPLETE marker can be durable while tile data is missing, and resume would publish missing tiles. Static finding (not reproduced). Fix: fsync data files, then the journal/marker, then the directory.

**H8 (downgraded to LOW after measurement). The wheel inspector cannot detect a wheel that lacks the implementation packages.** `scripts/build_release_artifacts.py` (`inspect_wheel`, required set around lines 120-128) requires only `__init__.py`, the SPICE TOML and dist-info files. **[measured on your machine with `s3_wheel_check.py`]** The wheel built from tracked files does contain `_numba_horizon/`, `map_algebra/`, `trajectory/`, `data/spice/default_kernels.toml` and `LICENSE`, has no junk files, installs into a clean venv, imports, passes `pip check`, and the `lunarscout-horizon-worker` entry point starts. So there is no current packaging defect from this cause; the inspector's required set is just weaker than it should be. Suggested fix unchanged: require those packages in `inspect_wheel`.

**H9. Undeclared runtime dependencies.** **[FIXED, 2026-10-09: `analysis` (pandas) and `plot` (matplotlib) extras; missing modules raise OptionalDependencyError with the install command]** `pandas` (`spice_geometry.py:306, 433`) and `matplotlib` (`spice_geometry.py:459, 485`; `scenario.py:1243, 1442, 1670`) are imported lazily but not declared in dependencies or extras. Root-exported `body_azimuth_elevation_dataframe`, `body_vectors_ned_dataframe`, and `plot_body_elevation(s)` raise raw `ModuleNotFoundError` after a plain `pip install`. Fix: add a `plot`/`analysis` extra and raise a structured error with an install hint.

**H10. Focal ops allocate a full `(H, W, fh, fw)` array.** **[FIXED, 2026-10-09: sum/mean/std/min/max/count/median/range reduce over horizontal strips with a 32 MiB temporary budget; results bit-identical across 1,200 configurations; 800x800 9x9 peak 472 -> <96 MiB]** `focal.py:257, 333, 345`. **[reproduced]** `focal_mean` on a 2000×2000 float32 raster with a 9×9 footprint peaked at 1.79 GB RSS (~100× the raster). A 10k×10k raster would need ~40 GB. Fix: strip processing or scipy/cumulative-sum implementations; document memory.

### MEDIUM

- **M1. Existing output with a matching manifest is returned unverified, with no progress callback.** `_writer.py:451-454`. **[reproduced]** A truncated 1000-byte `out.tif` was returned as valid; callbacks were never called, contradicting the exactly-once completion contract; `overwrite` is ignored.
- **M2. No locking on staging/journal/publish.** **[reproduced]** Two processes writing the same destination both failed with `RasterioIOError`, leaving shared partial state, and one runs rollback against the other's in-flight backups. Fix: exclusive lock file.
- **M3. Lazy and eager map-algebra paths disagree.** Lazy drops units on scalar multiply/divide (eager keeps `m`); grid-mismatch error classes and codes differ; `clip(uint8_expr, lower=0.5)` declares uint8 but computes float64 and fails at window 0; out-of-range clip bounds raise bare `OverflowError`; `lower > upper` silently returns `upper`; lazy `hypot`/`arctan2` with a scalar passes construction then crashes with `AttributeError` (`local.py:1225, 1252`). **[reproduced]**
- **M4. `scientific_identity()` is exponential on shared sub-expressions.** `_model.py:165-183`. **[reproduced]** `e = e + e` ten times (11 nodes) takes 40 s per call; `write()` and `plan()` call it several times. Fix: memoize by node id; cache CRS normalization.
- **M5. Other scale cliffs.** Python-loop taxicab/chessboard distance (`distance.py:189`), O(Z·N) zonal scans (`zonal.py:284`), Python-object integer paths, `compute()` keeps all intermediates alive.
- **M6. SPICE meta-kernel written to a fixed shared path** `/tmp/lunarscout_spice/lunarscout_default.tm`, non-atomically (`spice.py:131-156`). Races across processes or users; symlink pre-planting risk on multi-user hosts.
- **M7. SPICE kernel download integrity is weak.** `spice.py:159-233`. Checksum verified after replacing the destination (bad file stays); `sha256` is optional in the code, though every entry in the shipped `default_kernels.toml` that I read carries one (so the default path is verified after download, just after the file is already in place); any URL scheme accepted; `entry["filename"]` unsanitized. Same weaknesses in `map_products.download_map_product` (`map_products.py:226-247`: no timeout, no scheme check, no `Content-Length` check, raw `URLError`).
- **M8. No locking around SPICE global state** (`spice.py:267-285`); CSPICE via spiceypy is not thread-safe.
- **M9. UNVERIFIED: SPICE meta-kernel `PATH_VALUES` is a single string** (`spice.py:144-147`); strings over ~80 characters may fail without `+` continuation. Check with a long kernel directory.
- **M10. Inconsistent aberration correction and units.** `body_vectors_moon_me` uses `"NONE"`; NED, azimuth/elevation, and `sunlight_fraction` use `"LT+S"` (~20″ difference). `body_vectors_ned` returns kilometres while `body_vectors_moon_me` returns metres, undocumented.
- **M11. Regions:** a float mask with NaN counts NaN as True when nodata is None/georef is None (`regions.py:132-135`); `label_regions`/`region_sizes` raise on the common NaN-nodata georef case (`regions.py:179-208`).
- **M12. Callback exceptions re-wrapped.** `products.py` around lines 528-545, 710-727, 856-873, 1388-1405, 1583-1600: `OSError`/store-error branches do not check `adapter.callback_error is exc`, contradicting the "propagate unchanged" docstring. Five near-identical `except` ladders should be one helper.
- **M13. `open_temporal_cube(validate_layers=True)` fully decompresses every layer** to read metadata (`temporal_store.py:232-241, 1262-1270`). Float32 series means accumulate in float32 (`temporal_store.py:1067-1069`).
- **M14. Pure-Python dynamic planners with large state caps** (`_dynamic_gridrunner.py:307-379`: 5M states; SOC planners 100k labels). Hours-long runtimes plausible; untimed.
- **M15. Test/packaging hygiene.** `tests/test_release_artifacts.py:10` imports `scripts.*` but there is no `pythonpath` setting, so plain `pytest` can fail while `python -m pytest` works. README/AGENTS/CONTRIBUTING link to files that are not shipped (`docs/USER_GUIDE.md`, `examples/README.md`); relative links will 404 on PyPI.
- **M16. `map_products.map_product_download_directory`** does not constrain `scenario` to stay under the root (`map_products.py:202-205`); catalog parsing leaks raw `KeyError`/`TypeError`.

### LOW / NIT (selected)

- README typo "lunarscount" (lines 8 and 17, the latter in install instructions).
- `Scenario.generate_horizons` omits `patch_offset` and `patch_stride` (`scenario.py:940-995`).
- Error taxonomy: same `code=` reused across classes; `OutputExistsError` vs `ProductStorageError` for overwrite refusals; ~17 raises without `code=`; raw `ValueError`/`TypeError` in `temporal_store.py`, `georeference.py:139`, `map_algebra/coordinates.py:213`.
- Root namespace leaks (`from_temporal_cube_to_raster`, `PackageNotFoundError`, `_distribution_version`); `map_algebra/__init__.py` has no `__all__` (star-import shadows builtin `round`, leaks `np`, `Path`).
- `spice.py:36-41` fallback path resolves outside the package in an installed wheel; use `importlib.resources`.
- `raster == np.int64(3)` returns plain `False` (`raster.py:314, 333`); `//`, `%`, `**` do not support expression operands.
- `contains_pixel` is corner-anchored only while sibling methods default to center (`georeference.py:292-302`).
- `geotiff.py` `overwrite=False` has a check-then-replace race.
- Misaligned window sizes inflate output up to ~4.5× (**[reproduced]**); orphaned `.tmp` files after a kill; `os.O_DIRECTORY` raises `AttributeError` on Windows (package is Linux-only).
- Time handling: naive datetimes rejected by trajectory but treated as UTC elsewhere; `np.datetime64` ignores `source_timezone`.
- Float32-vs-Python-float threshold comparison in `EarthElevationThresholdProvider` (`providers.py:984`); two slip-interpolation implementations (`_block_cpu.py` vs `static.py`); duplicated `_span_is_allowed`.
- SOC planner: default `algorithm="greedy"` can report `reachable=False` for feasible problems; "exact" is exact only over event-time departures. Worth documenting.
- Stray repo files seen in the directory listing: `docs/#p1#`, `scripts/threshold_safe_havens.py~`, `.ipynb_checkpoints`, `examples/.#notebook_example.py`; `docs/API_DOCUMENTATION.md` and `docs/API_REFERENCE.md` have identical sizes (likely duplicates). `dist/` holds rc3 artifacts while `pyproject.toml` says rc5.
- `tests/test_dependency_boundary.py` hardcodes `version == "0.1.0rc5"`, which breaks on the next bump.

## What held up

- **Writer crash consistency:** 12 hard-kill (`os._exit`) injections (after window write, before/after journal write, mid-run, last window) all resumed to byte-identical output. Publication rollback logic reads correctly. **[reproduced]**
- **Lazy-import rule:** no top-level `spiceypy` or `numba` import; no import-time file, network, or dataset activity in modules read.
- **Dependency boundary:** no `lunar_analyst` import anywhere in `src`.
- **`__all__`:** no missing, undefined, or duplicate names; all classes in `errors.py` exported.
- **Geometry core:** GeoReference affine/inverse and pixel-anchor math match GDAL; aspect sign convention matches GDAL; SPICE UTC→ET handling, MOON_ME conversion, and NED basis check out.
- **Trajectory:** power accounting conserves energy at 100% efficiency (efficiencies other than 1.0 are rejected, so no sign or double-counting errors today); block-kernel indexing and A* admissibility look sound; safe-interval timing rules are mutually consistent.
- **Registry:** windowed operation ids and `file_backed_available` flags agree exactly.

## Suggested fix order for a release candidate

1. B1 (overflow) with exhaustive tests; H5 and H6 (data-loss guards); H1 (align) or a documented "same CRS only" restriction.
2. H2: implement Horn or relabel; H3: padding or rejection; H4: temporal numeric policy.
3. Build the wheel from the real repo, strengthen `inspect_wheel`, install into a clean venv, and run the CPU suite there (H8, H9, M15).
4. H7 fsync, M1-M2 writer hardening, M6-M8 SPICE and download hardening.
5. Document remaining caveats (memory of focal ops, aberration/unit conventions, SOC greedy default) in the User Guide before release.

## Coverage disclosure

- **Read fully:** `__init__.py`, `errors.py`, `cuda.py`, `horizon.py`, `map_products.py`, `products.py`, `georeference.py`, `raster.py`, `geotiff.py`, `alignment.py`, `terrain.py`, `regions.py`, `temporal.py`, `temporal_store.py`, `spice.py`, `spice_geometry.py`, most of `trajectory/`, most of `map_algebra/` (`_writer`, `_windowed`, `_planner`, `_windows`, `_model`, `_spatial`, `_dtypes`, `_eager`, `expression`, `distance`, `zonal`, `reductions`, `regions`, `temporal`), `pyproject.toml`, `MANIFEST.in`, `.gitignore`, `README.md`, `AGENTS.md`, and the release build script.
- **Partly read:** `scenario.py` (plotting code not read), `local.py`, `focal.py` (lines 1-560 of 929), `_temporal_model.py`, `_registry.py` (programmatic consistency check only), `CONTRIBUTING.md`.
- **Not read or not available:** `_numba_horizon/`, `data/`, `LICENSE`, `docs/USER_GUIDE.md`, `docs/ARCHITECTURE.md`, `examples/`, trajectory tests, `test_spice.py`, `test_geotiff_io.py`, `test_dem_loading.py`, and several map-algebra tests.


---

# Part 2: Horizon calculation, CUDA kernels, pipeline, deployment

Basis: C# parity (phase-4 report) shows the Numba/CUDA code reproduces the earlier C# implementation (6e-8 deg on one 16×16 LOLA patch; 4.04e-5 deg on a two-DEM fixture), including its bugs. It is not independent evidence of correctness. The reviewers therefore compared against a separate float64 brute-force ray march. Reproduction caveat: the brute-force comparisons ran on a CPU emulation of the production kernel (`traverse_hierarchy`, with a variant mirroring the production no-clamp behavior), not on a GPU.

## Verdict for Part 2

- **No kernel-level out-of-bounds, race, or uninitialized-memory defect was found** in the CUDA code by reading and by simulator runs. Index math is safe for 128-pixel tiles, there is no shared memory or `syncthreads`, and the safe-haven reducer matched its CPU reference in 26 randomized trials.
- **Algorithm-level errors exist** (H-A to H-D below) that C# parity cannot reveal. **Measured on your RTX 5090 (see "Measured results" below):** production GPU horizons on the LOLA fixture differ from the independent brute force by 0.30° on average and up to 2.03°, with 74% of 815 samples above the 0.005° tolerance. A single NaN node produced 89.43° horizons on the real GPU (H-C). Correcting only the far-field slope formula (H-A) in a scratch copy cut the mean error to 0.033° and removed the far-field error (2 to 5 km max 2.03° → 0.07°); the residual is near-field terrain below the observer that production misses (s5; consistent with H-B, not yet isolated).
- **Operational hazards** (a deadlock, silent resume reuse, silent invalid patches) matter for NRP production runs.

## Fix status, 2026-10-09 (branch `fix/hierarchy-cull-bound`, not merged)

All measurements in this section were run by Claude Code directly on the RTX 5090 or its CPU, against the 512 px LOLA fixture and the s2 float64 brute force unless stated. Full CPU suite after the last commit: 2131 passed, 26 skipped. GPU opt-in tests: 268 passed, 0 failed.

| Finding | Status | Commit | Evidence |
|---|---|---|---|
| H-B cull bound | **Fixed.** Blocks below the observer are bounded at the block's far side (flat model) or at the curved-slope peak d\* clamped to the block, in both CUDA hierarchy kernels and the CPU emulation. | `2a5d8b2` | Unit tests fail on the old bound. Accuracy equals a cull-disabled run on real terrain. 2048 px GPU time 545 s vs 582 s before. |
| H-A far-field slope | **Fixed.** One float32 `far_field_slope` (chord to the ground-sphere point, never subtracting two lunar radii) at all six CUDA sites and four CPU sites; the cull bound uses its exact peak. | `051ded8` | Within 4e-6° of exact geometry. Real terrain mean \|d\| 0.300° → 0.0153° (with H-B). Independent C# spherical reference 0.24107 on an obstacle ray: old formula > 0.25, new 0.24822. GPU time unchanged (the s4 float64 experiment doubled it). |
| M-B float32 radial cancellation | **Fixed for the far field** by the same `far_field_slope`. The 0.5 km flat/curved switch is unchanged. | `051ded8` | |
| M-G no iteration cap | **Fixed.** `hierarchy_maximum` caps each ray at 16 steps per pixel of range + 1000; a capped ray returns NaN and `generate_horizons` raises `HorizonGenerationError(code="horizon_step_limit_exceeded")`. | `3e20291` | Forced limit of 10 steps: raised after 7.8 s, GPU idle. Results unchanged; A/B timing within noise. The hazard was real: an unguarded experiment of mine (s6 "clamp" on a 2048 px DEM, a float32 step stall beyond about 32 km) ran the GPU at 100% overnight, and killing the host process did not stop the kernel. |
| H-C NaN / nodata | **Fixed.** Loader masks declared nodata (raw units) and non-finite values to NaN, detects radius encoding from valid values only, and no longer mutates the caller's array. Segment centres use `segment_center_terrain` (9x9 valid mean fallback). The kernel reads the observer height from its own node, so a NaN neighbour no longer corrupts valid pixels. | `e141ee5` | s2 `--synthetic nan`: max \|d\| 89.43° → 0.436°, mean signed +16.03° → −0.006°. |
| H-D `-inf` bins | **Fixed.** Raw tiles are read with the compressed format's −50° floor, so raw and compressed decode alike and products see finite horizons. | `a5c6b95` | Public CPU test (all `-inf` raw tile: lightmap 255, PSR 0) fails without the fix. |
| M-I observer on invalid node | **Fixed at the product level.** Horizons there stay invalid (about 90°). Every product marks pixels whose DEM node is invalid as nodata, checking each patch's DEM window just before it is written; `skip_dem_nodata_check=True` skips the check on all nine public product functions and Scenario methods. | `c310a0f` | Store, public API and Scenario forwarding tests. |
| QuadTreeHorizonGenerator parity | **Retired.** Lunarscout no longer compares generated horizons, traces or per-DEM buffers with the C# captures; non-horizon comparisons (pyramids, segment geometry, buffer merge) and all ReferenceRayEmulator comparisons remain. The C# far-field fix (H-A) is applied, uncommitted and untested, in `QuadTreeHorizonGenerator.cs` and `QuadTreeRayEmulator.cs`; the C# still has H-B and probably H-C. | `22d57d5` | |
| N-1 oracle / production mismatch (new) | **Fixed.** `d543e6e` removed the level-0 cell-exit clamp from production for outer-DEM speed but not from the CPU oracle or the traced kernel, so the CPU-oracle GPU test failed from that commit on. Both now step like production. | `36738ea` | GPU opt-in tests 268/268. |

**N-2 (new). Skipped nodes without the cell-exit clamp: large on synthetic spikes, small on real terrain.** Without the clamp, a 1-pixel level-0 step can straddle a single raised node: in the safety matrix (a 150 m node 850 m away on flat ground, ray through the node) production samples 75.5 m and 51.1 m either side and reports 5.21° instead of 10.12°. Worst cases (CPU, dense minus hierarchy): 4.77° without the clamp, 0.56° with a stall-safe clamp, 1.18° with quarter-pixel steps, 0.14° with both. On the 512 px LOLA fixture the clamp changes horizons by about 0.005° mean and made the worst sun-relevant ray worse (0.29° → 0.45°). The clamp is specific to bilinear terrain (it forces samples where bilinear peaks lie), so it was not restored; the matrix values are re-pinned as diagnostics.

**N-3 (new). The terrain interpolation model dominates near-field accuracy.** The same rays traced through an interpolating cubic spline instead of bilinear terrain (CPU, quarter-pixel steps):

| Limiting terrain | Sun-relevant rays: cubic − bilinear, mean \|d\| | > 0.05° |
|---|---|---|
| 0 to 20 m | 0.37° | 79% |
| 20 to 100 m | 0.35° | 78% |
| 100 to 500 m | 0.08° | 28% |
| 500 m to 2 km | 0.03° (p95 0.02°) | 3% |
| beyond 2 km | 0.002° (p95 0.007°) | 0% |

All 815 real-terrain rays: mean \|d\| 0.119°, mean signed +0.110° (cubic higher), max 2.85°. The bilinear brute force itself is converged to about 0.004° (1/16 vs 1/4 pixel steps). Within 500 m, where over half of sun-relevant rays are limited, the choice of terrain surface matters 10 to 30 times more than production's whole difference from the bilinear reference (about 0.015°). One tile; indicative only. Every accuracy figure in this review is relative to bilinear terrain, which the DEM does not establish.

## Measured results on your hardware (RTX 5090 Laptop, numba 0.66.0, CUDA 12.9 toolkit)

Run date 2026-10-08, from the output you pasted. The brute force is an independent float64 great-circle ray march over the same bilinear DEM, using production's pixel-to-CRS convention; it samples 12 random pixels of tile (0,0) × up to 72 azimuths.

| Run | Samples (comparable) | Mean abs diff | Max abs diff | > 0.005° | Pattern |
|---|---|---|---|---|---|
| CPU self-test (script plumbing) | 72 | 0.0040° | 0.0204° | 33% | Conventions agree; residual is algorithm difference on 0 to 5° horizons |
| Real terrain, LOLA 512 px fixture, production GPU | 852 (815) | 0.300° | **2.027°** | **74%** | Error grows with horizon elevation |
| Synthetic 200 m wall, production GPU | 426 (426) | 0.088° | 1.677° | 35% | +1.34° at 22.9°; one miss of −1.68° |
| Synthetic NaN scenario, first version | 355 (284) | 0.0001° | 0.0007° | 0% | Badly designed; not informative (my error) |
| **Synthetic NaN scenario, corrected, production GPU** | 497 | n/a (mean signed **+16.03°**) | **89.43°** | n/a | **Confirms H-C: production reports 89.4271° at pixel (149,147), az 270°, where the reference gives −0.0010°; limiting terrain 60 m away** |
| **Real terrain, far-field formula patched (s4, scratch copy)** | 852 (815) | **0.033°** | 0.996° | 39% | Mean signed −0.027°; all bins ≥ 0.5 km within 0.19°; remaining worst are near-field negatives at (9,2) and (39,5) |
| **Synthetic wall, far-field formula patched (s4)** | 426 (426) | 0.060° | 1.677° | 37% | Over-estimate gone; now **under**-estimates by up to 0.61° at 500 to 550 m (just past the 0.5 km switch); (74,64) az 75° miss unchanged |

Real-terrain error by distance to the limiting terrain point (re-run with distance binning; DEM 512×512 px at 20 m; production ray cap 12,288 m):

| Distance | Result |
|---|---|
| 0 to 0.5 km | mean signed about −0.06°, worst −1.0° (negative: production lower) |
| 2 to 5 km | n=216, mean signed **+0.72304°**, max abs **2.02731°** |
| 5 km and above | mean signed about +0.41° |

The error is concentrated beyond 2 km, which is where H-A's far-field formula applies (it is used beyond 0.5 km). The range cap (12,288 m) does not explain most misses: nearly all limiting distances are under 8 km.

Real-terrain error by brute-force horizon elevation (signed, production minus brute):

| Horizon | n | Mean signed | Max abs |
|---|---|---|---|
| below 0° | 189 | −0.107° | 0.996° |
| 0 to 2° | 97 | −0.008° | 0.168° |
| 2 to 5° | 119 | −0.001° | 0.213° |
| 5 to 10° | 116 | +0.029° | 0.274° |
| 10° and above | 294 | **+0.730°** | **2.027°** |

**What this establishes**

1. **C# parity is not evidence of correctness.** The phase-4 report shows 6e-8° agreement with C# on one 16×16 patch; against an independent reference the same fixture shows errors up to 2°. (Caveat: this run used 12 random pixels of tile (0,0), not the same 16×16 patch.)
2. **H-A is isolated on the GPU (s4).** `s4_farfield_patch_test.py` replaced only the far-field slope arithmetic (six sites in `cuda_backend.py`, evaluated in float64; cull logic, near-field formula, stepping and NaN handling unchanged) in a scratch copy and re-ran `s2` with the same seeded samples. Its CPU check puts the old formula +1.36° high at a 19.9° wall and the new one exact. On real terrain:

   | Bin | Unpatched mean signed / max abs | Patched mean signed / max abs |
   |---|---|---|
   | 10° and above (n=294) | +0.730° / 2.027° | −0.003° / 0.106° |
   | 5 to 10° (n=116) | +0.029° / 0.274° | −0.005° / 0.168° |
   | 2 to 5 km (n=216) | +0.723° / 2.027° | −0.003° / 0.071° |
   | 5 km and above (n=136) | +0.406° / 0.955° | +0.003° / 0.192° |
   | 0 to 0.5 km (n=297) | −0.060° / 0.996° | −0.069° / 0.996° |
   | All (n=815) | mean abs 0.300°, 74% > 0.005° | mean abs **0.033°**, 39% > 0.005° |

   H-A accounts for essentially all of the positive, elevation-dependent error. It does not touch the near-field negatives. On the wall, the patch overshoots in the other direction: rays limited at 500 to 550 m (just beyond the 0.5 km far-field switch) go from up to +1.34° to up to −0.61°. That points to a second far-field effect at a sharp edge (where in the segment the distance is taken), which the smooth LOLA terrain does not show. Not yet explained.
3. **Near the sun's elevation range the mean error is small but the tails are not.** Means for 0 to 5° are within ±0.01°, but the maxima are 0.17° and 0.21°. Under the repo's own uniform-disk model (2.546 fraction per degree), a 0.17° error on one ray is about 43% sunlight error for that ray. s5 has now measured how many rays are affected: 6.5% of sun-relevant rays exceed 0.05° (see "Residual diagnostics" below).
4. **Production is lower than the reference in some cases** (negative bins; mean −0.107° and as low as −1.0° where the horizon is below 0°; and in the wall test a ray that production reports as 0.0000° while the reference sees 1.68°). That sign is consistent with under-sampling or culling (M-A, H-B) but this run does not prove the cause. One more candidate: production's ray range is `min(1e6 m, 1.2 × DEM size)` (`geometry.py:473`), which could truncate a diagonal ray on a large DEM. On this fixture (20 m, cap 12,288 m) it does not apply: the negative errors are within 120 m of the observer (s5). The wall's 0.0000° ray is a far-field miss at the wall's corner, not a fallback value (s5).

**What it does not establish**

- The first NaN run did not test H-C (my plateau covered the observers; test-design error). The corrected run does, and confirms it (table above). It does not isolate the mechanism: code reading points to the −32000 sentinel added to the observer height and to all-zero fallback segments blended into neighbors, but this run cannot tell them apart. Some negative differences in that run (−0.68° at (152,152) az 120°, where production gave 0.0 and the reference 0.678°) show production also missing terrain near the NaN.
- The near-field negatives are isolated to H-B by s6 (below). About half of the sun-relevant rays still exceed 0.005° after the H-A and H-B experiments; that residual is not explained.
- It does not separate the algorithm's error from the brute force's own limits. The brute force uses a dense 0.25-pixel step on the same bilinear terrain model, so it is a fair reference for that model, but it is still my code.
- The run used one LOLA tile and one GPU. Polar, mixed-resolution and high-latitude cases are untested.

**Residual diagnostics (s5, unpatched repo, run 2026-10-08 on the RTX 5090 by Claude Code directly)**

`s5_residual_diagnostics.py` targets what s4 left.

1. **Production misses low near-field terrain.** At (39,5) az 45°, production gives −7.380°, which equals the reference truncated at 40 m (−7.380°); the reference horizon is −6.606°, set by terrain 80 to 120 m away inside the DEM. Production does not see that terrain. At (9,2) az 0°, production −6.946° matches the reference up to 40 m (−6.964°), and the reference's −5.949° comes from 40 to 60 m, where the ray reaches row 0 and the DEM edge. That case may be an edge-convention difference between the two codes (whether terrain beyond the last node row is sampled), not a production miss; s5 cannot tell.
2. **Not an edge or subpatch-phase effect.** Sweeps toward the top and left borders show no error that grows toward row or column 0, and no period-8 pattern shared with the interior control. Near-field mean signed errors are −0.20° to +0.04° per pixel everywhere.
3. **The common feature is terrain below the observer.** Every large near-field miss has a negative horizon, and production is lower than the reference. That is the case H-B describes (cull bound too low for blocks below the observer). Consistent with H-B; not proven.
4. **Exact-zero horizons are genuine, not a fallback.** Real terrain: 275 of 23.6 M values are exactly 0.0, and 130 of 150 sampled match the reference within 0.005°. Wall: 66.7% are exactly 0.0 and all 150 sampled match (flat ground). The (74,64) az 75° miss is a ray grazing the wall's corner at 1,500 m (reference cumulative maximum 0 → 0.68° → 1.68° between 1,400 and 1,600 m; the ray at 75.5° is also low, 4.08° vs 6.88°). It is a far-field miss at a sharp edge, unaffected by the H-A patch.
5. **Sun-relevant tails (reference horizon −2° to 5°, 463 rays, unpatched).** Mean signed −0.014°, mean abs 0.020°, max 0.848° ((11,92) az 340°, limiting terrain 40 m). 51% exceed 0.005°, 6.5% exceed 0.05°, 3.5% exceed 0.1°. Largest within 500 m (0 to 100 m: max 0.85°; 100 to 500 m: 73% above 0.005°); 2 to 5 km max 0.064°. Production runs low, so it predicts slightly too much sunlight. By the repo's uniform-disk model, 0.005° is 1.27% sunlight error on that ray.

**Cull vs step isolation (s6, run 2026-10-08 on the RTX 5090 by Claude Code directly)**

`s6_cull_step_isolation.py` built five scratch copies: unpatched, the s4 far-field patch (HA), HA with the cull test disabled in both hierarchy kernels (HA+nocull), HA with `margin_step` set to 0 (HA+dense), and both. Each generated horizons for the LOLA fixture and the wall and was compared on the same rays against one shared brute-force reference. The unpatched and HA variants reproduce s4 exactly (mean abs 0.3001° and 0.0333°), so the harness matches.

| Sample | Variant | Mean signed | Mean abs | Max abs | > 0.005° | Rays < −0.05° |
|---|---|---|---|---|---|---|
| s2 real terrain (815) | HA | −0.027° | 0.033° | 0.996° | 39.4% | 74 |
| | **HA+nocull** | −0.008° | **0.015°** | **0.366°** | 37.9% | 43 |
| Sun-relevant, −2° to 5° (463) | HA | −0.015° | 0.019° | 0.848° | 45.4% | 30 |
| | **HA+nocull** | −0.011° | 0.015° | **0.289°** | 45.1% | 27 |
| Wall (426) | HA | −0.059° | 0.060° | 1.677° | 36.6% | 108 |
| | **HA+nocull** | −0.036° | 0.039° | **0.604°** | 35.4% | 79 |

Named rays (reference / HA / HA+nocull): (9,2) az 0° −5.949 / −6.946 / **−5.960**; (39,5) az 45° −6.606 / −7.380 / **−6.620**; (11,92) az 340° −1.590 / −2.438 / **−1.578**; wall (74,64) az 75° +1.677 / 0.000 / **+1.851**.

1. **H-B is confirmed on the GPU.** Disabling the cull removes every named near-field miss to within about 0.02°, including (9,2). s5's suggestion that (9,2) might be an edge-convention artifact is wrong: it is the cull. The wall's 0.0000° ray is also the cull (it now reads 0.17° high instead of 1.68° low).
2. **M-A did not act in this run.** HA+dense is identical to HA on every ray. The reason is in the code, not the data: `advance = max(pixel_step, min(margin_step, angular_step))`, and `angular_step` (true_m × 1.51e-6 km) is below one pixel (20 m) until about 13 km, past this DEM's 12.3 km ray cap. So level 0 always advances one pixel (a quarter pixel within 0.5 km) and `margin_step` never binds. M-A may still apply on coarser or larger DEMs where rays run beyond about 13 km at 20 m, or about 0.66 × resolution km in general. A step finer than one pixel was not tested.
3. **Still unexplained:** after H-A and H-B, 45% of sun-relevant rays exceed 0.005° (mean abs 0.015°, max 0.29°), and the wall still reads up to 0.60° low at 500 to 550 m. Candidates: one-pixel level-0 steps versus the reference's quarter-pixel steps beyond 0.5 km, the 0.8-pixel `primary_far_step` floor, and subpatch-segment interpolation. None is tested.
4. **Cost.** Single timings (real + wall): unpatched 23 s, HA 43 s, HA+nocull 34 s. HA is slower because s4's experiment computes in float64, which runs at a small fraction of float32 speed on consumer GPUs; a production fix should stay in float32. Disabling the cull did not slow this 512 px DEM, but that says nothing about large DEMs, where the cull exists for speed. A conservative bound (evaluate below-observer blocks at their far edge) is the fix to test, not removing the cull.

**Other measurements**

- `acquire_phase1_real_fixture.py` printed three GDAL "`</head>` doesn't have matching `<head>`" errors but produced a file whose hash matched the manifest (the script raises on mismatch). My guess is GDAL probed sidecar files over `/vsicurl` and got an HTML error page; I have not verified that.
- `s3_wheel_check.py` reported no problems: see H8 above. Its pandas probe was too weak: it only tested that the attribute exists, not that calling it fails, so H9 is neither confirmed nor refuted by that run. The clean install did not pull in pandas or matplotlib, which is consistent with H9.
- The sdist includes `examples/` but not `docs/`, so README links to `docs/USER_GUIDE.md` will not resolve for sdist users; PyPI will show relative links as broken.

## Ranked findings, Part 2

### HIGH

**H-A. Far-field slope treats a ground chord as slant range.** **[FIXED, `051ded8`]** `cuda_backend.py:791-833` (production), `hierarchy.py` (~202, 232), `geometry.py:380-420`. **[formula structure verified by me; numbers reproduced by reviewer on CPU emulation; isolated on your GPU by s4: patching only this formula cut real-terrain mean abs error from 0.300° to 0.033° and the 10°-and-above bin from +0.730° to −0.003° mean signed]** For rays beyond 0.5 km the code computes `local_x = sqrt(true_m² − local_z²)`, treating `true_m` as slant distance, but `true_m` is a chord on the observer-radius sphere. The slope is overestimated by about u³/2 with u = tan(elevation). Brute-force comparison for a step wall: +0.019° at 4.9° elevation, +0.15° at 9.8°, +1.27° at 19.6°; replacing `local_x` with the horizontal range brings the 20° case within 0.007°. Impact: negligible near the poles, where relevant horizons are 0 to 2°, but errors exceed 0.005° above roughly 3° elevation, which matters for steep walls and non-polar use. C# has the same formula, so parity could not reveal it.

**H-B. The hierarchy cull is not conservative for blocks below the observer.** **[FIXED, `2a5d8b2`]** `cuda_backend.py:768`, `hierarchy.py:190`. **[reproduced by reviewer, CPU emulation; isolated on your GPU by s6: disabling the cull fixed every named near-field miss, e.g. (39,5) az 45° from −7.380° to −6.620° against a reference of −6.606°, and cut real-terrain max error from 0.996° to 0.366°]** The bound is evaluated at the nearest distance in the block. When the block maximum lies below the observer, the slope rises with distance, so the nearest-distance bound is too low and the block can be culled even though it holds terrain above the current horizon. On a 600 m summit, 36 of 12,639 culls hid terrain above the current horizon (worst +0.075 slope, about 4°); using a conservative bound changed final horizons by up to −0.22° and moved 17% of rays by more than 0.001°. Same bound in C#.

**H-C. NaN or nodata in a DEM corrupts a neighborhood, and declared nodata is never masked.** **[FIXED, `e141ee5`]** **(Reproduced on the real GPU by your run: one NaN node gave 89.43° horizons where the reference gives about 0°; mean error +16° over 497 samples.)** `geometry.py:531`, `geometry_numba.py`, `products.py` (`_load_dem`, `_normalize_dem_elevations`), `cuda_backend.py:1120`. **[reproduced by reviewer; code reading also confirms declared nodata is unused]** A NaN at a subpatch center yields an all-zero fallback segment that is blended into neighbors: horizons near the NaN went from 3.1° to 25.1°. A NaN beside an observer gives a −32000 observer height (about 90° horizons). A −9999 nodata is above the −20000 cutoff and is treated as real terrain (a 10 km pit); a nodata of 0 on a radius-from-center DEM disables the radius shift and puts elevations near 1.7375e6 m. Also, `np.asarray(float32)` returns the caller's array, so normalization mutates it in place. Fix: mask declared nodata to NaN first, reject NaN under observers and subpatch centers, make segment builders NaN-safe.

**H-D. A `-inf` horizon bin makes a lit pixel dark in lightmap (and its derived products).** **[FIXED, `a5c6b95`]** `lightmap_cuda.py:146-162`, `lightmap_cpu.py:157-170`. **[verified by me in code and reproduced by me with s1: fill −50° gave sun fraction 1.0; −inf and NaN gave 0.0]** `-inf` is the legal "no obstruction" sentinel, but the interpolation computes `-inf − (−inf) = NaN`, the `horizon < sun_top` test is false, and no photons are counted. The worker CLI defaults to uncompressed output, which keeps `-inf`; compressed output clamps to −50. An edge pixel whose rays all leave a single-DEM raster would be dark. Safe-haven and mission-duration inherit the signal. `psr_cuda.py` raises on non-finite horizons instead, so behavior is inconsistent. Fix: clamp or validate horizons in every session and on raw write.

**H-E. The bounded PSR pipeline hangs forever if the writer fails or is cancelled.** **[FIXED, 2026-10-09: the writer sentinel loop stops when the writer thread is no longer alive, so the recorded writer error is raised; same guard added to the horizon pipeline]** `psr_pipeline.py:705-735` and `put_writer_sentinel` (`:626`). **[verified by me in code; reproduced by me with s1 (no return after 40 s) and by the reviewer]** The writer thread exits on its first exception, the writer queue (capacity 1) usually still holds an item, and `put_writer_sentinel` retries on `Full` forever with no check that the consumer is alive. The existing test uses one patch, so the queue is never occupied. Fix: keep the writer in a drain loop after an error, or check `writer_thread.is_alive()` in the sentinel loop.

**H-F. Horizon resume trusts file names only, and tiles carry no provenance.** **[DECLINED, 2026-10-09]** `pipeline.py:195-199`, `file_format.py:393-397` (docstring admits it does not validate completeness), `file_format.py:308-313`. **[verified by me in code; truncated-tile skip and name collision reproduced by me with s1]** (a) A tile truncated to half its size is counted as done (skipped=1, computed=0) and later fails to read. (b) A raw `.bin` of the right size is accepted even if all NaN. (c) The only tile identity is `horizon_YYYYY_XXXXX_EEE`, with observer height truncated to decimeters, so 1.5, 1.55 and 1.59 m collide. (d) No header, version, DEM hash or parameter record exists, so a run with a different DEM, DEM order, or code version pointed at the same output directory silently skips every existing tile. The production-pipeline doc claims "structurally validated skip/resume". Fix: run manifest (DEM sha256, ordered DEM list, exact observer height, kernel parameters, library version), refuse to resume on mismatch, validate completeness on skip.

**H-G. Stale pyramid cache.** **[FIXED, 2026-10-09: cache removed; the GPU builds each pyramid per run in 0.28 s for a 30,400 px DEM]** `pyramid.py:34-48`, used from `horizon.py:180-186`. **[verified in code by two reviewers; not reproduced]** The `.pyr.bin` cache is validated only by float count and named from the DEM path with its suffix swapped. A DEM re-exported at the same size reuses the old max-pyramid, so culling skips blocks that are now higher and horizons come out too low, with no error. Two DEMs with the same stem collide. A read-only DEM directory makes `write_max_pyramid_cache` raise `OSError` and kill the run.

**H-H. Product pipelines convert any horizon read error into a permanent "invalid" patch.** **[FIXED, 2026-10-09: only a missing tile is an invalid patch; an existing tile that is truncated, undecodable or contains NaN raises `HorizonFormatError` (`horizon_tile_unreadable`) in all five products; the final progress event reports the invalid-patch count]** `psr_pipeline.py:356,385`, `lightmap_pipeline.py:174`, `elevation_pipeline.py:182`, `mission_duration_pipeline.py:275`, `safe_haven_pipeline.py:230`. **[static]** `(OSError, ValueError)` is caught and the patch is journaled invalid. A transient `EIO` or `ESTALE`, or a corrupt tile, becomes a masked hole, and the run reports success with no invalid-patch count. Resume does not retry, because the inventory identity is stat-based. Fix: retry transient errors, distinguish missing from corrupt, report an invalid-patch summary, offer `fail_on_invalid`.

### MEDIUM

- **M-A. Adaptive level-0 stepping under-samples.** `cuda_backend.py:841-850`. **[reproduced by reviewer, CPU emulation; not active in s6 on the 20 m fixture: `angular_step` stays below one pixel inside the 12.3 km ray cap, so `margin_step` never binds and zeroing it changed nothing]** `margin_step = margin·d·1.732` only guarantees no miss for terrain shallower than about 30° (tan 30° = 1/1.732), not the 60° the constant suggests. On rough 20 m terrain 57% of rays fell below the brute force by more than 0.005° (worst 0.72°); smooth terrain 45% (worst 0.22°). The bias is systematic (underestimate). The repo's own test asserts a 0.03 to 0.04 slope miss versus fixed step, and the hierarchy-safety JSON records a 0.57° dense-minus-hierarchy gap marked non-gating. (Since the oracle was aligned with production in `36738ea`, that diagnostic is 4.77°; see N-2.)
- **M-B. float32 radial cancellation in the production level-0 slope.** **[FIXED for the far field, `051ded8`; 0.5 km switch unchanged]** `cuda_backend.py:586-604, 812-833`. **[reproduced by two reviewers, float32 numpy emulation]** `point_radius − observer_radius` at about 1.74e6 m is quantized to 0.125 m. Errors above the 0.005° budget occur below about 1.4 km (0.0142° at 500 m; about 0.07° at 100 m and 0.35° at 20 m in one reviewer's emulation of the curved formula). The diagnostic kernels already use the cancellation-free `(h − oz)(2R + h + oz)` form; the production kernel does not. The switch at 0.5 km is also discontinuous.
- **M-C. Near the pole, grid convergence is unapplied and rays are translated, not rotated.** **[FIXED, 2026-10-10: patches with a pixel within 160 px of a geographic pole use subpatch size 1, so every pixel gets its own exact rays. `review_scripts/s10_polar_rays.py` on the 20 m LOLA fixture (pole at pixel 256,256), mean / max error against the s2 brute force: production 6.05° / 11.9° at the pole, 1.4 to 1.6° / 6 to 8° at 20 to 60 m, 0.72 to 0.77° at 80 to 160 m, about 0.03° at 1 to 1.3 km; subpatch 1 about 0.017° mean (max ≤ 0.25°) at every distance. After the fix, production matches the subpatch-1 numbers at every distance (mean 0.013 to 0.021°). Cost per pole patch: 1.63 GiB of segments per DEM, about 10 s CPU build, 0.7 s GPU; the whole 512 px fixture (12 of 16 patches near the pole) takes 128 s instead of 8 to 12 s, while a full-size polar DEM has about the same 12 pole patches. The CUDA session now keeps one work-buffer set per segment shape, so pole and ordinary patches can share a run. A GPU-gated public test on a pole-centred bowl reads 0° at the pole without the fix and the analytic 11.29° with it.]** `geometry.py`, `cuda_backend.py:1073-1120`. **[reproduced by reviewer]** At lat_0 = −90, errors of 4.5° at 40 m from the pole, 0.55° at 160 m, 0.084° at 400 m, 0.030° at 800 m; at lat_0 = −89 or 0 at most 0.02°. No polar test exists. The pole pixel is ambiguous in azimuth.
- **M-D. Mixed-resolution DEMs may be misregistered by half a pixel.** `products.py:~216`, `geometry.py`. **[reproduced by reviewer; depends on whether LOLA tiles are area- or point-registered, which is UNVERIFIED]** Up to 0.18° error against an analytic bump on 20 m and 100 m DEMs.
- **M-E. The accepted tolerance corresponds to 1.27% sunlight error, not 1%.** **[reproduced by me: dF/dh = 2/(πr) = 2.546 per degree for r = 0.25°; 0.005° gives 1.27%; about 0.0039° gives 1%]** The reported 1.0291e-4 equals 4.0412e-5° × 2.546, so the arithmetic is right and only the 0.005° proxy is mis-derived. The "124× margin" is a margin relative to C# only.
- **M-F. CPU and CUDA are not bit-identical, and parity tests are too small to show it.** **[static plus simulator]** libNVVM applies FMA contraction by default while x86 LLVM does not; GPU `sin/cos/atan2` differ from libm; only the subpatch kernel calls `libdevice.fmaf` explicitly. Simulator runs showed 2.2e-5 sun-fraction differences on random data. Tests for lightmap, mission-duration and safe-haven assert exact equality on 1 to 35 hand-picked samples. The CPU PSR backend is a float64 numpy reference while CUDA mirrors C# float32, and no CPU-vs-CUDA PSR test exists. Expect occasional ±1 byte or threshold-edge flips.
- **M-G. No iteration cap in the production hierarchy loop.** **[FIXED, `3e20291`]** `cuda_backend.py:672` (`hierarchy_maximum`, `while s <= s_end`), also `psr_cuda.py:150`. **[static]** Normal inputs terminate, but a corrupt `s_end` or non-finite segment could hang a GPU thread or trip a display watchdog. Fix: iteration guard and host-side segment validation.
- **M-H. A CUDA fault leaves a poisoned context and possibly a misleading first error; no GPU-memory pre-check.** `cuda_backend.py:1560`, `pipeline.py:521`, `:549`, `_prepare_production_pyramids` (`:1246`). **[static]** With two workers, A's illegal address makes B's next call fail first, and `errors[0]` is reported. `backend="auto"` falls back to CPU only on `CudaBackendError`, so a numba `CudaAPIError` from OOM in a constructor propagates.
- **M-I. Observer on invalid terrain is not detected.** **[FIXED, `c310a0f` (product nodata; horizons stay invalid)]** `cuda_backend.py:1120-1123`. **[static]** `bilinear` returns −32000 when any corner is nodata and it is added straight into `observer_z`.
- **M-J. Multi-DEM consistency not validated** (`cuda_backend.py:1130`, `horizon.py:438-447`): different sphere radii, vertical datums, extents or resolution ordering are accepted silently. **[static; effect UNVERIFIED]**
- **M-K. Tile extension and compress flag.** **[FIXED, 2026-10-10: after a tile is published, `HorizonTileStore.write` deletes the same tile's other-format file, so a fresh `.bin` is never shadowed by a stale `.cbin`. A failed write leaves the existing file in place. Store tests cover both directions and the failure case, and fail without the fix. The deployed Job sets `COMPRESS=1`; the worker's uncompressed default is unchanged.]** **[reproduced by me with s1]** Re-writing a tile with `compress=False` after `.cbin` exists leaves the old `.cbin` preferred on read (read back 5.0, new data 20.0). The worker defaults to uncompressed while `generate_horizons` defaults to compressed, so a restarted pod can mix both.
- **M-L. Worker and Job.** `worker.py:121-208`, `deploy/horizon-job.yaml`. **[static]** No SIGTERM handler (as PID 1 Python may ignore it, UNVERIFIED), orphan `*.tmp.*` files never cleaned; `POD_COUNT` is not checked against `completions` (a larger value reports Succeeded with whole stripes ungenerated); `backoffLimit` unset (default 6, shared) so preemption churn can fail the Job; one deterministic bad tile fails the whole partition on every retry.
- **M-M. Dockerfile and image provenance.** `deploy/Dockerfile`. **[static]** Floating base tag `python:3.12-slim`, unpinned `numpy`/`scipy`, floating `cuda-toolkit`, root user, no `.dockerignore`. RTX 5090 (sm_120) needs CUDA ≥ 12.8; a floating toolkit minor can produce PTX that older node drivers reject. `build.sh` tags the short SHA of `HEAD` even from a dirty tree, and `--tag-latest` creates a mutable tag.
- **M-N. Product stores.** `product_store.py:233-302`. **[static]** No cross-process lock; the manifest fingerprint covers georef, config and horizon inventory but not DEM contents, backend or code beyond the constant `phase6b-v1`; stat-based inventory identity breaks resume after rsync or copy of the horizons directory.
- **M-O. Test portability.** `test_phase4_cuda_mechanics.py` (~line 229) asserts compute capability (12, 0) and fails on every other GPU; GPU tests are gated by an env var, not by GPU presence.

### LOW / NIT (selected)

- Last row and column of a DEM are never sampled (`pixel_x >= width − 1`, `hierarchy.py:125`, `cuda_backend.py:683`); a 200 m wall in column 299 of 300 gave 42.2° vs 46.9° **[reproduced by reviewer]**.
- 0.25° azimuth point sampling: spacing 87 m at 20 km and 436 m at 100 km, so narrow ridges can fall between rays.
- The CPU hierarchy oracle used by tests (`hierarchy.py:258-262`) clamps the level-0 advance to the cell exit, while the production kernel does not; they differ by up to 0.7° on random rays **[reproduced by reviewer]**, so the oracle cannot stand in for production.
- `is_complete` accepts a block length of 1 but the decoder rejects it **[reproduced]**; no file-format header or version; values beyond ±50° are silently clamped on compressed write **[reproduced by me]**.
- Two `local_x`/slope conventions: the production far-field uses chord distance; `direct_reference.py` is exact-spherical but is not compared against production angles anywhere.
- Lightmap iterators hold a lock across `yield` (deadlock if two iterators interleave); `select_device` is thread-local while `horizon.py:177` hardcodes device 0; tile geometry assumed square; only subpatch, lightmap and PSR kernels use `cache=True`.
- Progress-callback exceptions abort a horizon run after the tile is written; `KeyboardInterrupt` can leave non-daemon threads running (`pipeline.py:511-520`).
- `acquire_phase1_real_fixture.py`: hash is computed on a re-converted ENVI copy, a mismatch raises but leaves the file in place, fixed predictable `/tmp` output path, no timeouts. The URL comes from the fixture manifest, which I could not read.
- Evidence gap: the phase-4 memcheck covers one 16×16, one-pass fixture only; `initcheck` was not run (racecheck and synccheck would be vacuous since there is no shared memory or barrier); the lightmap, PSR and reducer kernels are not covered.

## What held up in Part 2

- Projection: forward and inverse stereographic match PROJ to 4e-10 m at lat_0 of −90, −89.5, −80, 0 and 45 **[reproduced]**. Azimuth convention (0 north, clockwise, `index × 0.25°`) agrees with brute force. Ray fit position error at most 4e-4 px.
- Max-pyramid: exact block-max equality on odd shapes (37×53, 1×9, 130×7, 65×65) with NaNs **[reproduced]**; the four-cell height bound is conservative in height.
- CUDA kernels: no out-of-bounds read or write found; all `device_array` buffers are written before use; same-stream ordering protects slot reuse; PSR pinned buffers are released only after a synchronous `compute_patch`.
- File writes: tiles are written to a temp file, fsynced, and renamed atomically, so a killed run leaves orphan temp files rather than partial tiles.
- Safe-haven reducer: 26 randomized simulator trials matched the CPU reference with 0 mismatches.

## How to settle the open questions

`review_scripts/` (saved beside this report) contains scripts for your machine; see its README. `s1_cpu_repros.py` (CPU, run by me; not yet run on your venv) reproduces eight of the findings above. `s2_horizon_vs_bruteforce.py` has been run on your GPU for real terrain and the wall and corrected NaN cases (results above); `--synthetic bowl` has not been run. `s4_farfield_patch_test.py` (real terrain and wall) isolated H-A. `s5_residual_diagnostics.py` characterized the residual. `s6_cull_step_isolation.py` isolated H-B and showed M-A inactive on this fixture. Still open: the residual after H-A and H-B (45% of sun-relevant rays above 0.005°), the wall's under-estimate at 500 to 550 m, and a test of a conservative cull bound as the actual fix. `s3_wheel_check.py` has been run: no packaging problems. The phase-4 report's C# capture project is not in the repo; if it is needed to reproduce the parity numbers it should be added.

Open after 2026-10-09:

- Terrain interpolation model (N-3): decide what surface horizons represent (bilinear, a low-curvature interpolant, or bilinear with stated uncertainty) before tuning level-0 stepping. Quarter-pixel steps cut sun-relevant rays above 0.05° from 6.0% to 1.3% against the bilinear reference for 36% to 46% more GPU time on the 512 px fixture (two runs); not timed on a large DEM.
- Wall residual: after H-A and H-B the synthetic wall still reads up to 0.6° low at 500 to 550 m, and the wall-corner ray at (74,64) az 75° is missed unless the cull is disabled; a near-distance diagonal bound did not fix it. Cause unknown.
- M-D to M-F, M-H, M-J, M-L, M-O: not started. M-C and M-K fixed 2026-10-10. M-M and M-N: declined 2026-10-10. H-F (resume provenance): declined 2026-10-09. (A note here that a PSR validation error surfaced as `psr_storage_failed` was a misreading of pytest's source context; it surfaces correctly as `psr_calculation_failed`.)
- C# (`QuadTreeHorizonGenerator.cs`): H-A applied but not built or tested; H-B and H-C not ported.
- Merge `fix/hierarchy-cull-bound`.

## Part 2 coverage

- **Read fully:** `direct_reference.py`, `fixed_step.py`, `kernel_math.py`, `hierarchy.py`, `pyramid.py`, `subpatch.py`, `generator.py`, `geometry.py`, `geometry_numba.py`, `contract.py`, `cuda_backend.py`, `lightmap_cuda.py`, `lightmap_cpu.py`, `lightmap.py`, `psr_cuda.py`, `psr.py`, `safe_haven_cuda.py`, `safe_haven.py`, `worker.py`, `file_format.py`, `pipeline.py`, `product_store.py`, `product_vectors.py`, `horizon.py`, `lightmap_pipeline.py`, `elevation_pipeline.py`, deploy files, the acquire script, most of `psr_pipeline.py` and `mission_duration_pipeline.py`.
- **Skimmed:** `safe_haven_pipeline.py`, parts of `cuda_backend.py` kernels (hierarchy sections), `products.py`/`scenario.py` horizon parts, the phase 1 to 6 docs, and the phase 1 to 4 tests (mostly by name).
- **Not read:** `test_phase6b_*` bodies beyond sampling, `scripts/numba_horizon` benchmark and capture scripts, `benchmarks/`, and the missing files listed above.
