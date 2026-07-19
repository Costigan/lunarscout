# PLAN1: Python/Numba API and TestPyPI 0.1.0

Status: governing execution plan for the first Python-only Lunarscout package
candidate

Last updated: 2026-07-19

This plan converts the successful Python/Numba horizon and downstream-product
evaluation into a supported Lunarscout API, an installable wheel, and a
verified `0.1.0` publication on TestPyPI. It uses checkboxes as the progress
record; a checked item means the evidence exists, not merely that code has
been drafted.

`docs/ARCHITECTURE.md` is the normative target architecture. This plan
supersedes the moonlib distribution and managed-runtime work in older plans.
Older prototype documents remain useful as implementation history and
scientific evidence, but they do not preserve a production dependency on C#,
Python.NET, CLR, or `moonlib`.

## 1. Decisions and release principles

- [x] Adopt Python plus Numba as the production implementation.
- [x] Do not ship or require the former C# `moonlib` implementation.
- [x] Keep the package installable and useful on machines without an NVIDIA
  GPU.
- [x] Keep production horizon generation NVIDIA-CUDA-only.
- [x] Provide useful CPU implementations for every public downstream
  horizon-consuming product.
- [x] Permit explicit CUDA acceleration, and automatic CPU fallback, for
  downstream products.
- [x] Keep `import lunarscout` lightweight: importing the package must not
  import Numba CUDA, initialize a CUDA context, load SPICE, open GDAL data, or
  write files.
- [x] Preserve bounded-memory, patch-oriented processing instead of retaining
  a regional horizon cube.
- [x] Preserve resumable staged output, atomic final publication, validity
  masks, and completed-output protection.
- [x] Treat TestPyPI as a real installation and API evaluation, not as a claim
  that the API is already stable for `1.0`.
- [x] Confirm the initial supported Python versions. The proposed minimum
  matrix is Python 3.11 and 3.12 on Linux x86-64.
- [x] Decide whether the first TestPyPI upload is named `0.1.0rc1` followed by
  `0.1.0`, or whether `0.1.0` itself is the first upload. Using an RC first is
  preferred because package-index versions are immutable.
- [ ] Record the supported Numba, NumPy, Rasterio/GDAL, SpiceyPy/CSPICE, and
  NVIDIA driver ranges from clean-environment tests.

## 2. Intended 0.1.0 product scope

The first TestPyPI evaluation is intended to exercise the complete useful
lighting workflow. Safe havens and mission duration are part of that scope,
not unspecified future additions.

### Existing public foundations

- [x] NumPy raster values with explicit `GeoReference` metadata.
- [x] Grid comparison and validation helpers.
- [x] UTC temporal axes and time-series data conventions.
- [x] Scenario and DEM discovery helpers.
- [x] Reading and plotting existing `.bin` and `.cbin` horizon tiles.
- [x] SPICE-backed Sun and Earth vector and position helpers.
- [x] Verify these APIs from an installed wheel with no source checkout on
  `PYTHONPATH`.

### Product APIs to promote

- [x] Horizon generation from one or more DEMs, using NVIDIA CUDA.
- [x] Time-series solar lightmaps, using CPU, CUDA, or automatic fallback.
- [x] Permanent-shadow-region (PSR) classification, using CPU, CUDA, or
  automatic fallback.
- [x] Sun-center terrain-relative elevation time series.
- [x] Earth-center terrain-relative elevation time series.
- [x] Safe-haven products derived from Earth-outage intervals and contiguous
  low-Sun duration.
- [x] Landed mission duration using a sunlight-fraction threshold.
- [x] Landed mission duration using a Sun-center terrain-relative elevation
  threshold.
- [x] Landed mission duration using sunlight fraction plus Earth-center
  terrain-relative elevation thresholds.
- [x] Landed mission duration using Sun-center plus Earth-center
  terrain-relative elevation thresholds.

### Explicitly outside the initial product scope

- [x] Document that battery state-of-charge, power-system simulation, thermal
  modeling, traverse planning, and arbitrary user-defined raster reducers are
  not `0.1.0` features.
- [x] Document that landing-slope masks may be combined with generated
  products by callers, but are not silently folded into the scientific
  meaning of PSR, safe-haven, or mission-duration outputs.
- [x] Decide whether physical TIFF-block recovery without a completion journal
  is a later enhancement. Journal-based restart remains required for `0.1.0`.

## 3. Preserve the accepted scientific and file contracts

### Horizons

- [x] Preserve 128 by 128 horizon patches with 1,440 float32 azimuth samples
  per pixel at 0.25-degree spacing.
- [x] Preserve compatible `.bin` and `.cbin` reading and writing.
- [x] Preserve cumulative multi-DEM obstruction handling.
- [x] Preserve bounded resident DEM pyramids and reusable GPU buffers.
- [x] Preserve validation and skipping of already-complete horizon files.
- [ ] Publish the horizon format, coordinate, observer-height, naming, and
  partial-edge contracts in user documentation rather than relying only on
  prototype notes.

### Lightmaps and PSR

- [x] Preserve uint8 time-series lightmaps with
  `trunc(255 * visible_solar_fraction)`.
- [x] Preserve the accepted 16-slice solar-disk and 0.27-degree solar
  half-angle model.
- [x] Preserve PSR value `255` for permanent shadow and `0` where the Sun
  clears the terrain horizon.
- [x] Preserve the separate validity mask; value zero is valid science data
  and is not nodata.
- [x] Preserve the five-viewpoint vector-reduction heuristic and accepted
  upper-solar-limb semantics.
- [ ] Give these scientific choices stable algorithm identifiers and versions
  in staged-product manifests and public metadata.

### Elevation, safe haven, and mission duration

- [x] Keep Sun-center and Earth-center terrain-relative elevation as separate
  float32 products.
- [x] Define an Earth outage as a maximal half-open interval during which the
  Earth-center elevation relative to the terrain horizon is strictly below
  the configured threshold.
- [x] Define safe-haven output as the longest contiguous low-Sun duration for
  each Earth-outage interval.
- [x] Keep the four landed mission-duration calculations as distinct
  operations rather than a public mode string.
- [x] Preserve inclusive mission-duration thresholds and separate evaluation
  and candidate-start intervals.
- [x] Preserve bounded streaming reducers for long time series.
- [ ] Put threshold units, interval endpoint rules, time-step assumptions,
  invalid-pixel behavior, and output units in every public docstring and user
  guide section.

### Common GeoTIFF products

- [x] Preserve full-grid, 128 by 128 tiled, compressed BigTIFF output.
- [x] Preserve integer predictor 2 and a suitable floating-point predictor.
- [x] Preserve deterministic configurable invalid payloads and a separate
  dataset validity mask.
- [x] Preserve per-band UTC metadata and ordered timestamp metadata for time
  series.
- [x] Preserve staged output, manifest binding, durable completion journals,
  restart, and atomic final publication.
- [x] Preserve an existing completed output when an overwrite attempt fails.
- [ ] Define which product metadata fields are public compatibility promises
  and test them from an independent Rasterio/GDAL reader.

## 4. Public API design and promotion

No private `_numba_horizon` function should be re-exported unchanged merely to
make it public. The public surface must be small, domain-named, typed, and free
of prototype terminology.

### Naming and facade

- [x] Approve a concise public naming table for root functions, focused
  modules, and `Scenario` convenience methods.
- [x] Provide a simple root-level or `Scenario` path for the operations users
  normally call in scripts and notebooks.
- [x] Keep detailed backend status, diagnostics, and tuning in a focused
  namespace rather than crowding the package root.
- [x] Avoid `native`, `managed`, `moonlib`, and implementation-specific
  `numba` names in the scientific API.
- [x] Use four plainly named mission-duration functions rather than one
  function with a fragile string-valued mode.
- [x] Decide whether product creation returns `Path` alone or a small immutable
  result object containing the path and summary metadata.
- [ ] Add complete type hints and docstrings for every promoted symbol.
- [x] Curate `src/lunarscout/__init__.py` without violating the lightweight
  import boundary.

The naming review should cover at least these conceptual operations:

```text
generate_horizons
generate_lightmap
generate_psr
generate_sun_elevation
generate_earth_elevation
generate_safe_havens
mission_duration_from_sunlight
mission_duration_from_sun_elevation
mission_duration_from_sunlight_and_earth
mission_duration_from_sun_and_earth_elevation
```

These are planning names, not accepted API until the naming checkbox is
complete.

### Shared arguments and behavior

- [x] Standardize `backend="auto" | "cpu" | "cuda"` for downstream products.
- [x] Make `backend="cpu"` avoid importing or probing CUDA.
- [x] Make explicit `backend="cuda"` fail clearly if CUDA cannot be used; it
  must never silently switch to CPU.
- [x] Make `backend="auto"` expose which backend was actually selected through
  diagnostics, progress metadata, or the result object.
- [ ] Standardize `overwrite`, `start_fresh`, resume, invalid-payload,
  compression, and output-path arguments.
- [ ] Standardize explicit Moon-ME vector inputs and generated SPICE vector
  inputs, including precedence and shape validation.
- [ ] Preserve exact UTC-to-ET conversion as the default when generating
  vectors.
- [x] Standardize the simple monotonic fraction progress callback for normal
  callers.
- [x] Retain a structured event callback for applications that need stage and
  patch detail.
- [x] Define callback exception behavior and document that GUI callers may
  need to marshal updates onto their UI thread.
- [x] Expose cooperative cancellation without requiring an application job
  framework.
- [ ] Validate cheap Python inputs and output paths before expensive SPICE,
  CUDA, or file initialization.

### Structured exceptions

- [x] Add domain-based public exceptions under `LunarscoutError` for invalid
  inputs, grids, vectors/times, horizon formats, horizon generation, compute
  backends, product calculation, product storage, and cancellation.
- [x] Give actionable failures stable `code` values and machine-readable
  `details`.
- [x] Replace or temporarily alias transitional `Native*` exception names.
- [x] Ensure no public error suggests installing .NET, locating a DLL, or
  initializing a managed runtime.
- [ ] Test exact exception class, stable code, important details, and output
  preservation for representative failure paths.

### Promotion gates for each product

For horizon generation, lightmaps, PSR, both elevation products, safe havens,
and all four mission-duration products:

- [ ] Freeze a reviewed signature and docstring.
- [ ] Add root/module and `Scenario` access where appropriate.
- [ ] Add synthetic scientific tests independent of the former C# runtime.
- [ ] Add CPU tests when the product has a CPU backend.
- [ ] Add explicitly gated real-CUDA comparison tests when CUDA is supported.
- [ ] Add missing/corrupt horizon, invalid vector/time, partial-edge, mask,
  restart, cancellation, and failed-overwrite tests as applicable.
- [ ] Add an executable example that uses only public API.
- [ ] Verify the example from an installed wheel outside the checkout.

## 5. Remaining product validation

### Horizon generation

- [x] Complete the accepted CUDA correctness evaluation on synthetic and real
  terrain.
- [x] Validate reusable buffers, resident DEM pyramids, bounded memory, and
  compatible files.
- [ ] Run the promoted API on a representative multi-DEM scenario and record
  input hashes, output inventory hashes, cold/warm timing, host RSS, and GPU
  memory.
- [ ] Verify no-GPU and incompatible-driver failures from the installed wheel.
- [ ] Verify cancellation and restart across a process interruption using the
  promoted API.

### PSR

- [x] Validate CPU and CUDA scientific identity on focused fixtures.
- [x] Validate the complete 1,599-patch Mons Mouton product in QGIS,
  Rasterio, and GDAL.
- [x] Improve sustained processing with four reader workers, five bounded
  decoded-horizon slots, 16-patch durable checkpoints, and reusable pinned
  decode buffers.
- [x] Record the optimized full-run result: 62.690 seconds and 25.5064
  patches/second for 1,599 patches.
- [x] Record exact output identity with SHA-256
  `e246ac369b36d3e5f67f9c6c1f64284f0ddbc26448c17358b69cdd69c9ffed5d`.
- [ ] Re-run a representative installed-wheel PSR job through the public API
  and capture the final reproducible benchmark record.
- [ ] Treat asynchronous double buffering, multiple CUDA streams, and
  multi-patch kernel submissions as optional unless installed-wheel evidence
  reveals a release-blocking regression.

### Lightmaps and body elevation

- [x] Implement CPU and CUDA backends for lightmaps.
- [x] Implement separate CPU and CUDA Sun- and Earth-center elevation
  products.
- [ ] Record separated vector-generation, horizon-read/decompression,
  transfer, kernel/CPU calculation, TIFF write, flush/checkpoint, and total
  timings for short and long representative time series.
- [ ] Record CPU and GPU utilization, host RSS, GPU memory, output size, and
  throughput.
- [ ] Verify the TIFF 65,535-band limit is rejected before product creation.
- [ ] Verify explicit-vector runs do not load SPICE.
- [ ] Verify a deliberately disabled-CUDA `auto` run completes on CPU.

### Safe havens

- [x] Implement the safe-haven calculation and tiled resumable product.
- [x] Implement CPU and CUDA downstream backends.
- [ ] Add public tests for Earth-outage detection at exact threshold and time
  boundaries.
- [ ] Add public tests for no-outage, whole-interval outage, adjacent outages,
  missing patches, and partial edge tiles.
- [ ] Benchmark representative CPU and CUDA safe-haven products, recording
  separated stage timings, throughput, CPU/GPU utilization, host RSS, GPU
  memory, output identity, and band count.
- [ ] Run a deliberately disabled-CUDA end-to-end `auto` fallback product.
- [ ] Exercise restart, cancellation, failed overwrite, and invalid-tile
  journaling through the public API.

### Landed mission duration

- [x] Implement all four duration reducers and product pipelines.
- [x] Implement bounded streaming rather than a full time-by-region cube.
- [ ] Add public scientific tests for inclusive thresholds, candidate-start
  windows, evaluation endpoints, no feasible start, and duration-unit
  conversion.
- [ ] Compare CPU and CUDA near threshold boundaries and document any allowed
  floating-point tolerance without changing classifications silently.
- [ ] Benchmark short and long representative series for all four operations.
- [ ] Record separated stage timing, throughput, CPU/GPU utilization, host
  RSS, GPU memory, output identity, restart, and cancellation evidence.
- [ ] Run deliberately disabled-CUDA end-to-end fallback cases.

## 6. Operational and failure semantics

- [x] Keep queues and decoded horizons explicitly bounded.
- [x] Ensure a completion journal never advances beyond flushed TIFF data.
- [x] Keep the staged TIFF open during normal processing while using bounded
  durable checkpoints; do not reopen it for every patch.
- [x] Quantify PSR checkpoint recomputation as at most the unjournaled bounded
  checkpoint batch.
- [ ] State the recomputation bound for every promoted product.
- [ ] Test cancellation before initialization, during vector generation,
  during horizon preparation, after calculation, during a checkpoint batch,
  and before publication.
- [ ] Test corrupt `.bin` and `.cbin`, truncated compressed data, missing
  patches, incompatible staged manifests, and malformed journals.
- [ ] Test disk-full behavior during TIFF data write, TIFF synchronization,
  journal serialization/write/fsync, metadata finalization, and publication.
- [ ] Test forced process termination at representative stages and prove that
  restart never trusts unflushed data.
- [ ] Test failed overwrites and prove the previous complete output remains
  byte-for-byte unchanged.
- [ ] Test progress monotonicity, resume-aware initial progress, invalid-patch
  completion, callback failures, and final `1.0` behavior.
- [ ] Decide whether to implement physical TIFF block recovery when the
  journal is missing or explicitly defer it with a documented limitation.

## 7. Remove the managed-runtime product path

The former implementation may remain temporarily as migration evidence, but
it must not be part of the installed product or required verification.

- [x] Remove `pythonnet` from package dependencies.
- [x] Remove or replace public `native` initialization and status APIs.
- [x] Route all promoted product calls directly to the Python implementation.
- [x] Remove .NET and DLL configuration from normal installation and examples.
- [x] Ensure no wheel extra installs Python.NET or a managed assembly.
- [x] Exclude C# build outputs and managed assemblies from wheel and sdist.
- [x] Convert any essential C# parity cases into immutable fixtures or
  independently specified Python scientific tests.
- [x] Remove C# execution from release acceptance.
- [x] Decide whether to delete `native/moonlib` and native tests before
  `0.1.0`, or retain them temporarily as clearly historical, undistributed
  migration evidence.
- [x] Mark the managed-runtime sections of `docs/FRESH_PLAN.md`, old roadmaps,
  and API sketches as superseded where readers could mistake them for current
  installation guidance.
- [x] Verify `import lunarscout` and representative product runs never load
  `clr`, `pythonnet`, or `moonlib` modules.

## 8. Packaging the test library

### Metadata and dependencies

- [ ] Review `pyproject.toml` metadata: version, description, license, authors,
  maintainers, URLs, classifiers, keywords, and supported Python versions.
- [ ] Choose core dependencies and optional extras from actual product scope
  and import boundaries. SpiceyPy is a lazy core dependency because generated
  vectors are in scope; do not advertise HDF5 unless a supported public product
  requires it. CUDA must not require the toolkit or driver at installation.
- [x] Add the validated Numba dependency and remove `pythonnet`.
- [x] Ensure CPU-only installation does not require the CUDA toolkit or an
  NVIDIA driver merely to resolve dependencies or import Lunarscout.
- [x] Confirm the base-install placement of Rasterio, PyProj, SciPy, and
  SpiceyPy, then test every advertised installation combination. Add HDF5 only
  if a supported public product requires it.
- [x] Define `lunarscout[cuda]` as the supported NVIDIA installation profile,
  using the validated Numba-CUDA CUDA 12 user-space stack while keeping the
  base installation CPU-only. Both profiles import as `lunarscout`; neither
  installs an NVIDIA driver.
- [x] Read the version from installed package metadata so source and wheel
  report the same value.
- [ ] Add development-only build, test, lint, and package-validation tools to
  the development extra or documented development environment.

### Wheel and sdist contents

- [x] Build both wheel and source distribution in a clean build environment.
- [x] Run `python -m twine check` on both artifacts.
- [x] Inspect artifact contents against an allowlist.
- [x] Include required SPICE manifests and other small static package data.
- [x] Exclude generated GeoTIFFs, benchmark artifacts, caches, local DEMs,
  journals, staged files, native binaries, and repository virtual
  environments.
- [x] Ensure the sdist can build a wheel without access to the repository
  checkout or Git history.
- [x] Record artifact filenames, sizes, and SHA-256 hashes.
- [ ] Verify license and third-party notices are present where required.

### Installed runtime behavior

- [ ] Ensure Numba caches use a writable user or configured cache location
  when site-packages is read-only.
- [ ] Verify cache unavailability causes an actionable diagnostic or ordinary
  JIT fallback, not an import failure.
- [ ] Verify data caches include source identity, dimensions, dtype, algorithm
  version, and integrity metadata.
- [ ] Verify all file creation honors explicit caller paths and does not write
  during import.
- [x] Verify package behavior from a current working directory unrelated to
  the source tree.

## 9. Verification matrix

### Ordinary CPU suite

- [x] Maintain an ordinary test suite that does not require a GPU.
- [x] Last recorded prototype result: 114 passed and 14 skipped.
- [x] Add public API, exception, import-boundary, and installed-wheel tests.
- [x] Run the complete suite with CUDA deliberately hidden or disabled.
- [x] Prove all downstream product families complete with `backend="cpu"`.
- [x] Prove all downstream product families complete with `backend="auto"`
  falling back to CPU.

### Explicitly gated CUDA suite

- [x] Keep real GPU tests explicitly gated.
- [x] Last recorded focused PSR result: 19 passed.
- [x] Prove that every CUDA integration test executed a real kernel rather
  than merely observing GPU visibility or selecting a backend name.
- [x] Compare public CPU and CUDA results for lightmap, PSR, both elevation
  products, safe haven, and all mission-duration variants.
- [x] Run public horizon generation on the supported NVIDIA environment.
- [x] Record device, driver, Numba, CUDA-stack, and GPU-memory information.

### Clean environments

- [x] Create clean Python 3.11 CPU-only environment and install the wheel.
- [x] Create clean Python 3.12 CPU-only environment and install the wheel.
- [x] Create a clean supported NVIDIA environment and install the same wheel.
- [x] Run `pip check` in every clean environment.
- [x] Run import and core raster/temporal smoke tests outside the checkout.
- [x] Run one small public product from every downstream family on CPU.
- [x] Run one small horizon generation and one product from every CUDA-backed
  downstream family on the NVIDIA host.
- [x] Reproduce the package and key scientific outputs in a second environment
  rather than relying only on the development checkout.
- [ ] Record expected behavior for no GPU, hidden GPU, missing driver,
  incompatible driver, CUDA initialization failure, and CUDA JIT failure.

### Independent file validation

- [ ] Read generated `.bin` and `.cbin` horizons independently and verify
  dimensions, azimuth ordering, dtype, and hashes.
- [ ] Inspect every GeoTIFF product family with Rasterio and GDAL.
- [ ] Verify CRS, transform, dimensions, dtype, band count, tiling,
  compression, predictor, timestamps, metadata, mask, valid values, and
  invalid payload.
- [ ] Compare complete pixel arrays and masks, not only summary statistics,
  for accepted regression fixtures.

## 10. Documentation and examples

- [x] Rewrite the README installation section around `pip install
  lunarscout`; remove the promised moonlib installation path.
- [x] Explain CPU and optional CUDA execution, lazy core SPICE support, and any
  dependency groups the candidate actually advertises.
- [ ] Update `docs/USER_GUIDE.md` from private/provisional product descriptions
  to the accepted public API.
- [x] Keep `docs/ARCHITECTURE.md` synchronized with any approved API or
  packaging changes.
- [x] Add a public horizon-generation example with editable local paths.
- [ ] Convert the PSR example to public API and preserve explicit CUDA failure
  behavior.
- [ ] Add public lightmap and Sun/Earth elevation examples.
- [ ] Add a public safe-haven example that explains outage bands and duration
  units.
- [ ] Add examples for all four mission-duration calculations, sharing setup
  where practical.
- [ ] Add a CPU-only example that proves useful work without CUDA.
- [ ] Document QGIS rendering for sparse classification products: both zero
  and 255 are valid PSR classes.
- [ ] Document progress, cancellation, resume, overwrite protection, staging,
  checkpoint recomputation bounds, and cleanup of abandoned staged jobs.
- [ ] Document structured exceptions and troubleshooting for SPICE, file
  formats, GDAL/Rasterio, CUDA visibility, driver compatibility, and memory.
- [ ] State tested hardware/software matrices and known limitations without
  implying support for untested platforms.
- [ ] Update `CHANGELOG.md` with the Python-only architecture, promoted
  products, managed-runtime removal, packaging, and TestPyPI status.

## 11. Automation and release preparation

- [ ] Add CI for supported Python versions, ordinary CPU tests, package build,
  artifact inspection, `twine check`, and clean-wheel smoke tests.
- [ ] Ensure CI never requires .NET or the former native projects.
- [ ] Keep real GPU acceptance explicitly gated on a suitable NVIDIA runner or
  documented manual release procedure.
- [ ] Add a release checklist/script that records Git commit, dirty state,
  versions, test results, artifact hashes, and upload target.
- [ ] Require a clean worktree and an annotated release commit or tag for the
  final candidate build.
- [ ] Verify that no credentials, local paths, generated products, staging
  files, or benchmark data are present in artifacts.
- [ ] Prepare TestPyPI project ownership and a scoped API token using a trusted
  publishing mechanism or local credential store; never commit credentials.

## 12. TestPyPI publication and installation

- [ ] Select an unused immutable candidate version.
- [ ] Build wheel and sdist from the exact reviewed commit.
- [ ] Complete all release gates in Section 14.
- [ ] Upload only the reviewed artifacts to TestPyPI.
- [ ] Verify the TestPyPI project page, metadata, README rendering, license,
  Python requirement, and artifact list.
- [ ] Install from TestPyPI into a new CPU-only environment. Use PyPI as the
  dependency source when dependencies are not mirrored on TestPyPI.
- [ ] Run the installed CPU smoke suite and one product from every downstream
  family outside the checkout.
- [ ] Install the same artifact into a clean NVIDIA environment.
- [ ] Run a small real horizon generation and one CUDA product from every
  CUDA-backed downstream family.
- [ ] Verify explicit CUDA failure and automatic CPU fallback using the
  installed package.
- [ ] Record commands, environment manifests, logs, output hashes, host RSS,
  GPU memory, and file-inspection results in a TestPyPI candidate report.
- [ ] Ask external test users to report installation, import, API, scientific
  output, error-message, and performance feedback against the exact candidate
  version.
- [ ] Fix candidate defects under a new immutable version; never overwrite an
  existing index artifact.

Publication to production PyPI is a later decision and requires a separate
review of TestPyPI feedback. This plan does not authorize a production-PyPI
upload.

## 13. Milestones

### M0: Scope and API decisions

- [x] Confirm Python/platform support and candidate version strategy.
- [x] Approve public names, signatures, result type, backend semantics,
  progress, cancellation, and exception taxonomy.
- [x] Confirm that every product listed in Section 2 is in the first candidate
  or explicitly document a justified deferral.

### M1: Public Python-only product surface

- [x] Promote horizon generation.
- [x] Promote lightmap and PSR.
- [x] Promote Sun- and Earth-center terrain-relative elevation.
- [x] Promote safe havens.
- [x] Promote all four mission-duration products.
- [x] Remove the managed-runtime path from public execution and dependencies.

### M2: Scientific, operational, and performance acceptance

- [ ] Complete public CPU and gated CUDA correctness matrices.
- [ ] Complete representative safe-haven and mission-duration benchmarks.
- [ ] Complete deliberately disabled-CUDA fallback runs.
- [ ] Complete restart, cancellation, disk-full/process-kill, journal, and
  failed-overwrite tests at the agreed release depth.
- [ ] Record output identity, memory, utilization, and separated stage timings.

### M3: Package candidate

- [ ] Finalize dependencies, extras, metadata, package data, and cache
  behavior.
- [x] Build and inspect clean wheel and sdist.
- [ ] Complete documentation, public examples, CI, and changelog.

### M4: Clean-wheel reproduction

- [ ] Pass CPU-only installation and all downstream smoke tests.
- [ ] Pass NVIDIA installation, horizon generation, and CUDA product tests.
- [ ] Pass second-environment reproduction with no source checkout or .NET.

### M5: TestPyPI evaluation

- [ ] Upload the reviewed candidate.
- [ ] Reinstall from TestPyPI and repeat CPU and GPU smoke tests.
- [ ] Publish the candidate evidence report and collect tester feedback.

## 14. TestPyPI 0.1.0 release gates

All of these boxes must be checked before representing the candidate as ready:

- [ ] The public API is reviewed and documented for horizon, lightmap, PSR,
  Sun/Earth elevation, safe haven, and all four mission-duration products.
- [x] Package installation and use require no .NET, Python.NET, CLR, DLL, or
  `moonlib` artifact.
- [x] `import lunarscout` is side-effect-light on CPU-only and NVIDIA systems.
- [x] Ordinary CPU tests pass on every supported Python version.
- [x] Explicitly gated real-CUDA tests pass on the supported NVIDIA stack and
  prove actual kernel execution.
- [x] Every downstream product completes in a deliberately disabled-CUDA
  environment.
- [x] Explicit CUDA requests fail truthfully and never silently fall back.
- [x] Scientific values, masks, metadata, and accepted fixture hashes match.
- [ ] Restart, cancellation, durable journal ordering, atomic publication, and
  failed-overwrite protection pass the agreed failure matrix.
- [ ] Resource use is bounded and representative host RSS and GPU memory are
  recorded.
- [ ] Safe-haven and mission-duration performance is measured and adequate for
  evaluation users.
- [ ] Wheel and sdist pass build, content inspection, `twine check`, and
  clean-install tests.
- [ ] Installed-wheel tests run outside the checkout with no source-tree
  `PYTHONPATH`.
- [ ] README, user guide, examples, architecture, changelog, limitations, and
  troubleshooting reflect the shipped artifact.
- [ ] The exact commit, artifact hashes, environment versions, commands, test
  counts, benchmark results, and known limitations are recorded.
- [ ] No user-generated product or credential is overwritten, deleted, or
  included in the release artifacts.

## 15. Post-0.1 roadmap and sequencing

The first release remains the coherent terrain, horizon, lighting, elevation,
safe-haven, and landed mission-duration surface defined above. Map algebra,
distance fields, and path planning must not expand the `0.1.0rc1` acceptance
scope or delay TestPyPI feedback on that surface.

Document the architectural boundaries for later work now, but do not freeze
exact public function names and signatures until the installed `0.1` candidate
has been exercised. TestPyPI feedback may reveal conventions that should be
shared by later raster operations.

The intended release sequence is:

```text
0.1.0rc1, rc2 if needed  TestPyPI installation and lighting API evaluation
0.1.0                    stabilized first lighting release
0.2.0rc1                 map-algebra and distance-field evaluation
0.2.0                    validated map-algebra release
0.3.0rc1                 path-planning evaluation
```

Every uploaded candidate is immutable. Candidate defects use a new RC version.
Publication of `0.1.0` or any later version to production PyPI remains a
separate user decision after TestPyPI evidence and is not authorized by this
plan.

### M6: Stabilize the 0.1 product

- [ ] Complete the M2 through M5 gates before implementing later public
  product families.
- [ ] Collect TestPyPI installation and API feedback for both `lunarscout` and
  `lunarscout[cuda]`.
- [ ] Fix release-candidate defects under a new immutable RC version.
- [ ] Decide whether the evidence supports a production-PyPI `0.1.0` release.
- [ ] Record reusable public conventions that map algebra and path planning
  must follow.

### M7: Map algebra and distance fields for 0.2

Map algebra is a reusable scientific raster layer, not an application-policy
layer. Before implementation, specify and review:

- [ ] The boundary between in-memory NumPy operations and file-producing,
  patch-oriented operations.
- [ ] Whether each operation accepts arrays, paths, or separate plainly named
  array and file APIs; avoid ambiguous inputs that change return type.
- [ ] `GeoReference` compatibility, alignment, pixel orientation, physical
  units, and explicit rejection of shape-only grid matching.
- [ ] Validity-mask and nodata propagation, deterministic invalid payloads,
  dtype promotion, overflow, scalar/raster broadcasting, and floating-point
  behavior.
- [ ] Memory bounds for large rasters. An operation must not silently load a
  regional raster into memory merely because its small-array implementation
  uses NumPy.
- [ ] Backend behavior consistent with the `0.1` contract: `backend="auto"`,
  `"cpu"`, and `"cuda"` where both implementations are useful; CPU must not
  probe CUDA, explicit CUDA must not fall back, and selected backends must be
  recorded in progress and file metadata.
- [ ] Progress, cancellation, staging, restart, overwrite protection, and
  structured exceptions for file-producing operations.
- [ ] A focused module or namespace that keeps the curated package root usable
  as the operation count grows.

Distance fields are part of the `0.2` map-algebra family. Before freezing their
API, define:

- [ ] Raster seed representation and validity rules.
- [ ] Connectivity and distance metric.
- [ ] Pixel-distance versus physical-distance units, including anisotropic or
  rotated grids.
- [ ] Treatment of invalid areas, barriers, raster edges, empty seeds, and
  all-seed rasters.
- [ ] Output dtype, precision, maximum-distance behavior, and deterministic
  invalid payload.
- [ ] A useful CPU implementation where scientifically appropriate, with CUDA
  acceleration following the shared backend contract. Any genuinely
  CUDA-only algorithm must say so explicitly, as horizon generation does.
- [ ] CPU/CUDA correctness comparisons, independent reference cases, bounded
  memory evidence, and clean `lunarscout[cuda]` installed-wheel tests.

Do not represent the map-algebra API as accepted until a `0.2.0rc1` TestPyPI
candidate has been installed and evaluated independently of the checkout.

### M8: Path planning for 0.3

Path planning builds on accepted raster and distance-field primitives. Design
it only after the `0.2` data, grid, mask, unit, and backend contracts are
settled.

- [ ] Define explicit inputs for traversability, cost, distance, slope,
  illumination, communication, start, destination, and other constraints.
- [ ] Keep scientific products separate from mission policy. Do not silently
  choose or combine slope suitability, illumination risk, battery state,
  thermal state, traverse objectives, or communication weighting.
- [ ] Decide how callers provide policies and weights without making one
  application policy the scientific meaning of a Lunarscout product.
- [ ] Define coordinate, grid, mask, barrier, route-validity, path-cost, and
  unreachable-destination semantics.
- [ ] Define deterministic tie-breaking, reproducibility, progress,
  cancellation, memory bounds, and structured failures.
- [ ] Separate reusable path-search algorithms from scenario mutation, job
  orchestration, UI behavior, and Lunar Analyst application state.
- [ ] Add independently specified synthetic cases and realistic-terrain
  validation before preparing a `0.3.0rc1` TestPyPI candidate.

### Other post-candidate engineering

These items remain valuable but are not automatic `0.1` blockers when the
limitation is documented:

- [ ] Evaluate asynchronous transfers and double-buffered CUDA patch slots.
- [ ] Compare multiple CUDA streams with multi-patch kernel submissions.
- [ ] Tune bounded decompression and checkpoint defaults on additional storage
  and CPU configurations.
- [ ] Implement physical TIFF block recovery when a journal is missing, if its
  complexity is justified by real recovery needs.
- [ ] Expand supported operating systems, Python versions, and GPU stacks from
  clean-environment evidence.
- [ ] Refine provisional `0.x` API names using TestPyPI user feedback.

## 16. Progress reporting template

At the end of each implementation unit, update the relevant checkboxes and
record:

- [ ] Exact input identities and source commit.
- [ ] What was implemented, promoted, packaged, or optimized.
- [ ] Exact verification commands and pass/skip/fail counts.
- [ ] CPU and CUDA backend selections and proof of actual kernel execution.
- [ ] Scientific value, mask, metadata, and hash comparisons.
- [ ] Separated stage and end-to-end timings and patch/time-slice throughput.
- [ ] CPU/GPU utilization, host RSS, GPU memory, queue bounds, and maximum
  decoded-horizon memory.
- [ ] Restart, cancellation, interruption, disk-full, and failed-overwrite
  results relevant to the unit.
- [ ] Clean-wheel or source-tree environment, dependency versions, and cache
  state.
- [ ] Remaining blockers for the next milestone and for TestPyPI publication.
