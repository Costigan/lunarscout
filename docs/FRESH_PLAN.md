# Plan for a Truthful and Reproducible Package

> **Superseded:** This managed-runtime packaging plan is retained only as
> historical evidence. `docs/PLAN1.md` and `docs/ARCHITECTURE.md` govern the
> Python/Numba product; its .NET, Python.NET, `moonlib`, and native-package
> directions are obsolete.

**Plan date:** 2026-07-13

**Source review:** `docs/FRESH_REVIEW.md`

**Milestone:** A clean Lunarscout checkout can build, install, test, and explain
the complete user installation and its Linux platform runtime without relying
on undeclared packages, former Lunar Analyst paths, or undocumented machine
state.

## 1. Milestone Definition

This milestone is about establishing a reliable package baseline. It does not
require every experimental API or native feature to be production-ready.

The package is truthful when:

- Project metadata, installation documentation, runtime error messages, and
  actual imports describe the same dependencies and optional capabilities.
- Every documented command uses the current standalone repository layout.
- Supported and unsupported platforms are stated explicitly.
- Experimental features and examples are labeled and do not masquerade as
  completed parts of the `0.1.0` contract.
- Native availability reporting does not claim that a configuration is usable
  unless its required components are discoverable.
- Test and release documentation distinguish ordinary unit coverage from real
  native/GPU validation.

The package is reproducible when:

- A clean checkout can build an sdist and wheel using documented commands.
- A fresh environment can install those artifacts without importing the source
  checkout.
- The complete user installation has an automated clean-install test.
- The installed Python and Linux runtime packages run their supported workflows
  without undeclared dependencies or a separate source checkout.
- The Linux platform runtime package can be assembled from a clean checkout
  using documented inputs and produces a passing native build/test result.
- CI executes the same commands that contributors and release operators are
  instructed to run.
- A release candidate records source commit, tool versions, artifact hashes,
  test results, and known skipped integration tiers.

### Milestone Exit Criteria

- [ ] A clean wheel and sdist build succeeds from the current repository.
- [ ] The wheel contains the expected Python modules and package data, and no
      unintended examples, tests, native build outputs, or local artifacts.
- [ ] The sdist contains all files required to rebuild the Python and Linux
      runtime wheels.
- [ ] The Python and Linux runtime wheels install together in fresh Python 3.11
      and 3.12 environments.
- [ ] `import lunarscout` and supported workflows pass using the declared runtime
      dependencies and automatically installed platform package.
- [ ] Experimental work is explicitly documented and does not alter the
      published dependency contract prematurely.
- [ ] Fresh-install smoke tests cover the complete installation.
- [ ] The committed native tree builds from a clean checkout on its supported
      Linux/.NET platform runtime.
- [ ] CPU-compatible native tests pass in CI.
- [ ] Primary documentation contains no former `packages/lunarscout` or
      `native/new_horizon` paths.
- [ ] Primary documentation, metadata, and runtime errors agree on dependencies,
      runtime acquisition, and prerequisites.
- [ ] Python and native CI checks are required before integration.
- [ ] Warning expectations are explicit and new warnings fail the appropriate
      CI job.
- [ ] A release-candidate verification report is generated and retained.

## 2. Scope Boundaries

### Included

- Python runtime and developer dependency design.
- Package metadata and package-data correctness.
- Wheel and sdist construction.
- Clean-environment installation and smoke testing.
- Python version support verification.
- A documented Linux platform-runtime build and installation path.
- Native build and CPU-compatible native test automation.
- CI and contributor verification commands.
- Installation, examples, platform, and troubleshooting documentation.
- Warning cleanup needed to keep supported dependency upgrades viable.
- Minimum asset inventory needed to know what is shipped in artifacts.
- Release-candidate evidence and artifact hashing.

### Deferred

- [ ] Native binary wheels or bundled cross-platform native payloads.
- [ ] macOS or Windows native support unless explicitly selected below.
- [ ] Broad moonlib source trimming or large C# refactors.
- [ ] Full scenario database ownership and mutation APIs.
- [ ] Production adoption of the large time-series raster format selected in D10.
- [ ] Complete scientific provenance redesign for all output products.
- [ ] API stabilization of every exported root name.
- [ ] Full static typing of the repository.
- [ ] Reproducible GPU numerical results across different GPU models.
- [ ] Bit-for-bit identical wheels across arbitrary operating systems.

Deferred work may be referenced by the package as a known limitation, but it
must not block this milestone unless it prevents the documented installation from
being built or used truthfully.

## 3. Decisions Required Before Implementation

The following decisions should be recorded in this file or in short ADRs before
dependent tasks begin. Recommended defaults prioritize a small, supportable
first baseline.

### Decision D1: One User Installation

**Status:** accepted on 2026-07-15.

**Decision:** Lunarscout has one supported user-facing installation rather than
minimal and optional feature profiles:

```bash
pip install lunarscout
```

That installation provides the Python raster/temporal APIs, moonlib-backed
calculations, SpiceyPy geometry, pandas DataFrames, and opinionated Matplotlib
plots. Moonlib and SPICE are essential capabilities, not optional extras.
Matplotlib and pandas are normal runtime dependencies. HDF5 runtime dependencies
will be included only if required by the implementation selected in D10.

Developer-only tools such as pytest, build, pip-tools, and Ruff remain separate
from runtime dependencies. This is a contributor toolchain, not a second user
profile.

One user installation does not require one physical distribution artifact.
The `lunarscout` package may depend on an automatically selected platform
runtime package as decided in D4 while preserving one user command.

**Decision checklist:**

- [x] Use one supported user-facing installation.
- [x] Include moonlib-backed capability in the normal product contract.
- [x] Include SpiceyPy in normal runtime dependencies.
- [x] Include pandas and Matplotlib in normal runtime dependencies.
- [x] Keep developer tools separate from user runtime dependencies.
- [x] Defer final HDF5 dependency treatment to D10.
- [ ] Record the final dependency list in `requirements.in` and package metadata.

**Unblocks:** runtime dependency metadata, clean-install testing, documentation,
and runtime packaging design.

### Decision D2: Supported Platform Matrix

**Status:** accepted on 2026-07-15.

**Decision:** support Linux x86-64 on Python 3.11 and 3.12 for the initial
milestone. Continue targeting .NET 10 (`net10.0`). GPU and real-scenario
validation remain an explicit integration tier rather than a requirement for
ordinary local commits. Windows and macOS are intended future targets, not
current claims.

Current Linux assumptions to revisit for later operating-system support:

- `RuntimeIdentifier` is fixed to `linux-x64` in moonlib and its tests.
- moonlib references `MaxRev.Gdal.LinuxRuntime.Minimal`.
- native bootstrap discovery contains a Linux runtime identifier and Linux
  loader-path handling.
- the current runtime setup uses `LD_LIBRARY_PATH` and sometimes `LD_PRELOAD`.
- Linux CSPICE uses an ELF `libcspice.so`; Windows requires a PE `cspice.dll`,
  and macOS requires a Mach-O dynamic library, normally named `.dylib`.
- platform runtime wheels/packages need distinct OS and CPU tags.
- MaxRev GDAL, HDF5, ILGPU, CSPICE, and GPU-driver behavior need validation on
  every added OS/architecture.
- macOS x86-64 and arm64 are distinct runtime targets.

**Decision checklist:**

- [x] Support Python 3.11 and 3.12 initially.
- [x] Support Linux x86-64 initially.
- [x] Continue targeting .NET 10.
- [x] Treat Windows and macOS as future support targets.
- [x] Keep GPU/real-scenario validation outside the ordinary commit loop.
- [ ] Decide whether Python 3.13 is added before or after `0.1.0`.
- [ ] Record unsupported combinations in primary user documentation.
- [ ] Add an OS-portability issue/roadmap section containing the assumptions above.

**Blocks:** CI matrix, classifiers, native documentation, and release criteria.

### Decision D3: Dependency Constraint Policy

**Status:** accepted on 2026-07-15.

**Decision:**

- Use tested lower bounds plus justified upper bounds in `pyproject.toml`.
- Keep `requirements.in` authoritative for direct runtime dependencies.
- Keep a separate authoritative development input, such as
  `requirements-dev.in`, for contributor-only tools.
- Use pip-tools to generate exact, transitive development/test lock files.
- Generate locks under Python 3.11 and 3.12 because dependency markers, wheels,
  and compatible transitive versions can differ by Python minor version.
- Test minimum supported dependencies in one CI job and current compatible
  dependencies in another.
- Do not exact-pin transitive dependencies in published wheel metadata.

The generated lock files are build/test inputs and are not edited manually.
`requirements.in` and `requirements-dev.in` remain the human-edited sources.

**Decision checklist:**

- [x] Select pip-tools as the lock/resolution tool.
- [x] Keep `requirements.in` authoritative for runtime direct dependencies.
- [x] Generate exact development/test locks rather than publishing transitive pins.
- [ ] Decide whether runtime upper bounds are allowed and when they are required.
- [ ] Choose final lock filenames and platform-marker policy.
- [ ] Define the cadence for dependency refreshes.
- [ ] Define ownership for vulnerability updates.

**Blocks:** metadata edits, CI environment construction, and contributor setup.

### Decision D4: Native Distribution Promise

**Status:** platform-package direction accepted on 2026-07-15; .NET hosting
mechanism requires a prototype.

**Decision:** `pip install lunarscout` automatically installs a matching
platform runtime package, initially `lunarscout-runtime-linux-x64`. That runtime
package contains moonlib and its required platform-native payload. There is no
user-visible `native` extra.

The preferred user experience does not require a separate system .NET
installation. However, Python.NET's documented CoreCLR host explicitly states
that [self-contained deployment is not supported](https://pythonnet.github.io/pythonnet/python.html).
Therefore the Gemini proposal
to run `dotnet publish --self-contained true` and load the result directly is a
hypothesis to test, not an accepted implementation.

Microsoft's self-contained deployment model publishes the runtime and framework
files alongside an OS/architecture-specific application; it does not compile
CoreCLR into an ordinary reusable class-library DLL. See the
[.NET deployment documentation](https://learn.microsoft.com/en-us/dotnet/core/tools/dotnet-publish).

The required evaluation is:

1. Test whether a private Microsoft .NET 10 runtime directory can be placed in
   `lunarscout-runtime-linux-x64` and passed to `clr_loader.get_coreclr()` using
   `dotnet_root` plus the moonlib runtime configuration.
2. If packaging it is too large, test downloading the exact runtime archive on
   first native use, verifying its checksum, and configuring `dotnet_root`.
3. If neither in-process approach is reliable, evaluate a self-contained .NET
   worker executable with process isolation and an IPC boundary. This changes
   the execution architecture and is a fallback, not the first choice.

The platform package should include the moonlib assembly, MaxRev GDAL/native
dependencies, and CSPICE shared library. Large SPICE kernels follow D8 and are
downloaded on first use rather than bundled.

**Decision checklist:**

- [x] Use an automatically installed platform runtime package.
- [x] Name the initial target conceptually `lunarscout-runtime-linux-x64`.
- [x] Include moonlib and its platform-native dependencies in that package.
- [x] Avoid a separate user-visible native extra.
- [x] Prefer not to require a separately installed system .NET runtime.
- [ ] Prototype private-runtime hosting with `clr-loader` and `dotnet_root`.
- [ ] Select an exact .NET 10 runtime patch from Microsoft's versioned
      `release-metadata/10.0/releases.json`; pin the archive URL and SHA-512
      rather than following a mutable "latest" URL.
- [ ] Measure compressed and installed platform-package sizes.
- [ ] Verify Microsoft .NET runtime redistribution terms and required notices.
- [ ] Decide package-bundled versus verified first-use download for CoreCLR.
- [ ] Define security-update behavior for a privately distributed .NET runtime.
- [ ] Define CSPICE/GDAL discovery within the installed platform package.
- [ ] Decide whether direct `GenerateHorizons` access outside `MoonlibBridge` is
      accepted temporarily or must be bridged before the milestone.
- [ ] Decide whether the known Python.NET unload problem requires a documented
      "do not unload; exit the process" policy.

**Blocks:** final runtime package layout, native status semantics, bootstrap
implementation, artifact size policy, and native installation documentation.

### Decision D5: Unfinished Threshold Feature

**Status:** open. This is separate from the general storage-format decision D10.

**Question:** Should the current uncommitted threshold/HDF5 work be completed
before packaging work or isolated from the milestone branch?

**Recommended decision:** fix its compile error and make its focused tests pass,
but keep the feature experimental and outside the supported `0.1.0` contract.
If that cannot be done quickly, move it to a separate branch before establishing
the package baseline.

**Decision checklist:**

- [ ] Choose complete-now or isolate-now.
- [ ] Define the minimum focused native tests required if completed now.
- [ ] Confirm that it does not add HDF5 to the supported in-process native contract.
- [ ] Confirm whether its assets belong in release artifacts.

**Blocks:** clean native build baseline and native CI.

### Decision D6: CI and Release Infrastructure

**Status:** partially accepted on 2026-07-15.

**Decision:** use GitHub Actions for automated tests and release-candidate
verification. During rapid single-developer work, checks are informative and do
not block ordinary checkpoint commits or require branch protection. Full
verification is mandatory only when intentionally preparing a release
candidate. Release publication is manual. Dependency vulnerability scanning is
report-only initially.

Upload wheel, sdist, test reports, and the release verification report for
release-candidate workflows. Do not upload real scenario data.

**Decision checklist:**

- [x] Use GitHub Actions.
- [x] Do not require branch checks during the current rapid-development phase.
- [x] Require the complete selected check set before a release candidate is promoted.
- [x] Publish releases manually.
- [x] Keep dependency vulnerability scanning report-only initially.
- [ ] Decide artifact retention periods.
- [ ] Decide whether releases target PyPI, TestPyPI, GitHub Releases, or internal
      distribution first.
- [ ] Decide when branch protection becomes worthwhile.

**Blocks:** workflow implementation and release procedure.

### Decision D7: Warning and Static-Quality Policy

**Status:** accepted in principle on 2026-07-15; formatting and type-checker scope
remain open.

**Decision:**

- Fix NumPy deprecations immediately.
- Treat unexpected Python warnings as errors in focused jobs.
- Prefer `pytest.warns` when a test intentionally triggers a warning. Allow a
  narrowly filtered warning only when it comes from understood third-party
  behavior and include a test comment explaining the exact reason.
- First fix all straightforward C# nullable warnings, then baseline only the
  genuinely difficult remainder and do not allow that count to increase.
- Adopt Ruff for syntax/import/basic quality checks.
- Defer repository-wide type checking; start with package metadata and selected
  mature Python modules.

**Decision checklist:**

- [x] Adopt Ruff for linting.
- [x] Fix NumPy deprecations immediately.
- [x] Treat unexpected warnings as errors in focused jobs.
- [x] Fix straightforward C# nullable warnings before baselining any remainder.
- [ ] Decide whether formatting is enforced now.
- [ ] Select mypy, pyright, or no type checker for this milestone.
- [ ] Set the initial Python warning policy.
- [ ] Set the initial C# warning budget.
- [x] Keep dependency/vulnerability findings report-only initially.

**Blocks:** quality CI and acceptance thresholds.

### Decision D8: Artifact Content and Large Assets

**Question:** Which data and native assets belong in the Python sdist, wheel,
and repository?

**Status:** kernel strategy accepted on 2026-07-15; other artifact contents
remain open.

**Decision:** do not bundle the large SPICE kernel set in every wheel or runtime
package. Download versioned kernels from NAIF on first use, verify cryptographic
checksums, and cache them in a Lunarscout data directory. Offline installation
is not a requirement. Python SpiceyPy and moonlib may retain separate native
CSPICE libraries, but should use compatible, explicitly versioned kernel data.

Package installation itself should not download kernels. First-use loading or
an explicit preload command owns the network operation so artifact construction
remains reproducible.

**Decision checklist:**

- [x] Download large SPICE kernels on first use rather than bundling them.
- [x] Do not require offline installation.
- [x] Verify downloaded kernel checksums and cache successful downloads.
- [ ] Decide whether `native/` is included in the sdist.
- [ ] Remove or exclude redundant static kernel copies after runtime requirements
      are inventoried.
- [ ] Decide whether `native/moonlib/cspice.dll` remains in a Linux-first source
      distribution.
- [ ] Decide whether example notebooks ship in the sdist and whether outputs are
      stripped.
- [ ] Decide whether `data/product_overview.tif` ships in the sdist.
- [ ] Approve a third-party asset inventory requirement before public release.

**Blocks:** manifest configuration, artifact inspection tests, and license review.

### Decision D9: Version and Release Target

**Question:** Is this milestone a corrected `0.1.0`, a new `0.1.1`, or a new
minor development release?

**Status:** version and manual ownership accepted on 2026-07-15; distribution
channel remains open.

**Decision:** `0.1.0` is the first published version because no artifacts have
been published. Release publication is manual and the sole human developer is
the release owner.

A GitHub Release is not a special source commit. It is a Git tag, normally
`v0.1.0`, plus release notes and optional attached artifacts that identify an
existing commit as a release. TestPyPI is a separate staging package index used
to test upload, metadata, and installation before publishing the same candidate
to PyPI.

**Decision checklist:**

- [x] Confirm no prior published artifacts.
- [x] Use `0.1.0` as the first published version.
- [x] Publish manually.
- [x] Assign release approval to the sole human developer.
- [ ] Select the initial distribution channel.
- [ ] Decide whether artifacts must be signed or accompanied by attestations.

**Blocks:** final metadata, changelog, and release workflow.

### Decision D10: Large Time-Series Storage Format

**Status:** preferred direction accepted on 2026-07-15; validation is required
before `0.1.0`.

**Decision:** use one multiband, tiled BigTIFF as the canonical durable format
if it passes the acceptance tests below. It provides a single-file product,
avoids high-file-count directories, and is directly usable through GDAL,
Rasterio, and QGIS. HDF5 is the fallback if BigTIFF cannot meet required access
or generation performance. It may also be used as a regenerable intermediate
or cache, but not as a second authoritative copy of every product.

There is currently no repository evidence that either format performs poorly
on CephFS. Performance must be measured with representative data rather than
inferred from stripe size alone. BigTIFF is naturally suited to spatial frame
reads, while a time-major or suitably chunked HDF5 layout may be better for
point and neighborhood light curves. The acceptance tests must expose that
tradeoff rather than assume one layout is efficient for both.

Required evidence:

- [ ] Define representative dataset dimensions, datatypes, and access patterns.
- [ ] Define quantitative acceptance thresholds for generation throughput,
      frame reads, point light curves, neighborhood light curves, file size,
      and QGIS usability.
- [ ] Record the CephFS stripe/object layout used for each benchmark.
- [ ] Benchmark full 128 x 128 x time patch writes into a tiled, multiband
      BigTIFF using the native generator's actual access order.
- [ ] Benchmark BigTIFF frame, point-history, and neighborhood-history reads.
- [ ] Confirm that a representative large product opens reliably in QGIS and
      that users can identify and select time bands without private metadata.
- [ ] Define UTC time-coordinate metadata using standard TIFF/GDAL facilities
      where possible, with a documented sidecar manifest only if necessary.
- [ ] Confirm BigTIFF offset handling, compression, finalization, interrupted
      write recovery, staged replacement, and concurrent reader behavior.
- [ ] Measure frame reads, point light curves, neighborhood light curves,
      sequential generation, compression, metadata cost, and recovery behavior.
- [ ] Run the same benchmark locally and on representative CephFS storage.
- [ ] If BigTIFF fails a threshold, benchmark HDF5 with representative chunk
      shapes, compression, cache settings, and the same access patterns.
- [ ] If an HDF5 intermediate/cache is retained, prove that it is disposable and
      reproducibly derived from the authoritative BigTIFF.
- [ ] Record the final validated format decision, schema, and migration policy.

**Blocks:** final runtime dependencies, durable format documentation, and the
`0.1.0` release.

## 4. Workstream A: Governance and Baseline Inventory

**Objective:** establish one agreed package contract before editing dependency
metadata or CI.

**Prerequisites:** Decisions D1-D10 have owners; D1-D3, the accepted part of D4,
D6, D7, the accepted part of D8, and the accepted part of D9 are recorded.

### Tasks

- [ ] Create a decision log table in this document or an `docs/adr/` directory.
- [ ] Record the single user installation and separate developer toolchain.
- [ ] Record the supported Python/OS/native matrix.
- [ ] Record the dependency and lock policy.
- [ ] Record the Linux platform-runtime distribution promise.
- [ ] Record the CI/release service and target channel.
- [ ] Inventory public modules and map each to its runtime requirements.
- [ ] Inventory direct runtime imports, including lazy imports.
- [ ] Inventory imports used only by tests and examples.
- [ ] Inventory package data currently configured in `pyproject.toml`.
- [ ] Inventory candidate sdist content under `native/`, `data/`, `examples/`,
      `scripts/`, and `docs/`.
- [ ] Record the current verification baseline:
      Python test count, native test count, warnings, build commands, and skips.
- [ ] Identify commands that depend on the active editable installation.
- [ ] Identify commands that rely on the repository being on `PYTHONPATH`.
- [ ] Identify references to former Lunar Analyst paths and environment variables.

### Deliverables

- [ ] Approved decision records.
- [ ] Runtime-requirement/import matrix.
- [ ] Artifact-content inventory.
- [ ] Baseline verification report tied to a commit and worktree state.

### Completion Criteria

- [ ] Every runtime import belongs to the declared user dependency set or Linux
      platform runtime.
- [ ] Every documented feature has explicit runtime/data prerequisites.
- [ ] There are no unresolved blocking decisions for metadata and CI work.

## 5. Workstream B: Restore a Green Source Tree

**Objective:** ensure packaging work begins from a source tree that builds and
passes its existing tests.

**Prerequisites:** Decision D5.

### Tasks

- [ ] Resolve `CS1106` in `native/moonlib/pipeline/LightmapThresholds.cs` or
      isolate the unfinished feature from the milestone branch.
- [ ] Run `dotnet build native/moonlib/moonlib.csproj -v minimal`.
- [ ] Run focused threshold tests if the feature remains in the branch.
- [ ] Run `dotnet test native/tests/HorizonGen.Tests/HorizonGen.Tests.csproj`.
- [ ] Run `.venv/bin/python -m pytest -q`.
- [ ] Confirm pure-Python import does not load Python.NET, CLR, moonlib, GDAL
      native bindings, or SPICE.
- [ ] Record expected skipped tests and why they are skipped.
- [ ] Remove or move editor lock files from tracked source.
- [ ] Ensure generated `bin`, `obj`, `TestResults`, cache, and notebook checkpoint
      content remains ignored.
- [ ] Add a non-destructive command that reports large local test artifact paths
      and sizes, or document equivalent commands.

### Verification

```bash
.venv/bin/python -m pytest -q
dotnet build native/moonlib/moonlib.csproj -v minimal
dotnet test native/tests/HorizonGen.Tests/HorizonGen.Tests.csproj -v minimal
git status --short
```

### Completion Criteria

- [ ] Current Python suite passes.
- [ ] Current native project builds.
- [ ] Current CPU-compatible native suite passes.
- [ ] Skips and warnings are recorded rather than silently ignored.
- [ ] No generated build/test artifact is staged for release.

## 6. Workstream C: Runtime and Developer Dependency Alignment

**Objective:** make the single user installation and separate contributor
toolchain match actual supported behavior.

**Prerequisites:** Decisions D1-D4 and D7.

### Tasks

- [ ] Make `requirements.in` the authoritative direct runtime dependency list.
- [ ] Add `spiceypy`, pandas, and Matplotlib as normal runtime dependencies.
- [ ] Keep NumPy, SciPy, Rasterio, pyproj, and Python.NET as normal runtime
      dependencies.
- [ ] Apply the HDF5 dependency result selected in D10.
- [ ] Create `requirements-dev.in` for pytest, build, pip-tools, Ruff, artifact
      inspection, and other contributor-only tools.
- [ ] Generate and commit Python 3.11 and 3.12 development/test lock files.
- [ ] Reflect the authoritative direct runtime requirements in `pyproject.toml`
      without hand-maintained semantic disagreement.
- [ ] Add dependency comments only where a native/platform boundary is non-obvious.
- [ ] Ensure missing external runtime/data failures raise structured Lunarscout
      errors with accurate remediation.
- [ ] Add tests that the complete runtime dependency set supplies every public
      Python API.
- [ ] Test kernel-download failures separately from missing-package failures.
- [ ] Run a resolver check for every supported Python version.

### Suggested Installation Test Cases

- [ ] Import the root package and all documented public modules.
- [ ] Exercise GeoTIFF, terrain, regions, alignment, temporal arrays/series,
      scenario paths, and catalog behavior.
- [ ] Exercise controlled SPICE kernel state and fake-kernel geometry.
- [ ] Exercise pandas and Matplotlib helpers with a non-interactive backend.
- [ ] Exercise native status without initializing CLR.
- [ ] Exercise native initialization using the installed Linux runtime package.
- [ ] Exercise the selected D10 storage implementation.

### Completion Criteria

- [ ] `pip check` succeeds in every fresh supported Python environment.
- [ ] Normal installation includes all accepted D1 capabilities.
- [ ] Contributor tools are not runtime dependencies.
- [ ] Missing external data/runtime conditions produce structured errors.
- [ ] Metadata and human-edited dependency sources agree.

## 7. Workstream D: Package Metadata and Artifact Construction

**Objective:** produce complete, minimal, inspectable wheel and sdist artifacts.

**Prerequisites:** Workstream C and Decisions D8-D9.

### Metadata Tasks

- [ ] Set the approved package version.
- [ ] Confirm name, description, license expression, and Python requirement.
- [ ] Add project URLs for source, issues, and documentation.
- [ ] Add authors/maintainers if desired for publication.
- [ ] Add classifiers only for verified Python versions and operating systems.
- [ ] Add keywords only if they improve package discovery.
- [ ] Confirm package discovery includes `lunarscout._native_runtime`.
- [ ] Confirm `src/lunarscout/data/spice/default_kernels.toml` is included.
- [ ] Decide whether to expose a supported runtime `__version__` value using
      `importlib.metadata`.
- [ ] Define whether a `py.typed` marker is appropriate now or deferred.

### Artifact Configuration Tasks

- [ ] Add explicit sdist inclusion/exclusion configuration.
- [ ] Exclude virtual environments, caches, build outputs, local test artifacts,
      editor files, scratch outputs, and untracked notebooks.
- [ ] Include source files required to build the supported Linux runtime package.
- [ ] Include native project files and required static inputs if D8 requires them.
- [ ] Exclude platform binaries that are not part of the support contract.
- [ ] Ensure Git LFS pointer handling is correct for any artifact-included LFS file.
- [ ] Strip notebook execution output or exclude notebooks from release artifacts.
- [ ] Ensure the license and required third-party notices are included.
- [ ] Build wheel and sdist without relying on the editable install.
- [ ] List artifact contents and compare them to an approved allowlist/policy.
- [ ] Record artifact sizes and fail on unexpected large growth.
- [ ] Check wheel metadata for requirements, platform dependencies, Python
      version, and license.
- [ ] Validate the sdist can rebuild the same functional wheel in a fresh directory.

### Verification

```bash
.venv/bin/python -m build
.venv/bin/python -m twine check dist/*
unzip -l dist/*.whl
tar -tf dist/*.tar.gz
```

The exact tool commands should be updated if D3 selects a different build
frontend. Artifact inspection should be automated rather than left solely to
manual review.

### Completion Criteria

- [ ] Wheel and sdist build from a clean checkout.
- [ ] Artifact validation reports no metadata errors.
- [ ] Wheel content is minimal and approved.
- [ ] Sdist content is sufficient and approved.
- [ ] Artifact sizes are recorded and within the agreed budget.
- [ ] Artifact SHA-256 hashes are generated.

## 8. Workstream E: Clean-Install and Wheel-First Testing

**Objective:** prove tests use installed artifacts rather than accidental source
checkout access.

**Prerequisites:** Workstream D.

### Test Harness Tasks

- [ ] Create fresh environments outside the repository for artifact tests.
- [ ] Install the built wheel rather than `pip install -e .`.
- [ ] Run imports with the repository root removed from `PYTHONPATH`.
- [ ] Run tests from a working directory outside the source checkout where
      practical.
- [ ] Assert `lunarscout.__file__` resolves into the fresh environment.
- [ ] Add an installed-package smoke script that imports and exercises supported
      operations using generated temporary data.
- [ ] Verify package data lookup works from the installed wheel.
- [ ] Verify SPICE manifest lookup does not depend on repository `data/`.
- [ ] Verify native status and initialization using the automatically installed
      Linux runtime package.
- [ ] Verify kernel first-use download and cached reuse with a controlled server
      or fixture, not live NAIF traffic in ordinary CI.
- [ ] Verify examples selected as supported can run against the wheel.
- [ ] Separate repository-only development examples from installed-package
      acceptance tests.
- [ ] Test the sdist installation path independently of the wheel path.
- [ ] Run `pip check` after every supported installation.
- [ ] Capture installed dependency versions in test artifacts.

### Matrix

| Installation | Python 3.11 | Python 3.12 | Clean wheel | Smoke workflows |
| --- | --- | --- | --- | --- |
| Lunarscout plus Linux runtime | [ ] | [ ] | [ ] | [ ] |
| Lunarscout sdist/source build | [ ] | [ ] | [ ] | [ ] |

Add Windows and macOS rows only when D2 is deliberately expanded.

### Completion Criteria

- [ ] No supported smoke test imports from the checkout accidentally.
- [ ] The complete product installs and passes on every claimed Python version.
- [ ] Installed package data resolves correctly.
- [ ] Source distribution installation passes its designated smoke test.

## 9. Workstream F: Linux Platform Runtime Package

**Objective:** build and install moonlib, CoreCLR hosting support, and native
dependencies through the automatically selected Linux x86-64 runtime package.

**Prerequisites:** Decisions D2, D4, D5, and D8; Workstream B.

### Build and Package Contract Tasks

- [ ] Document the required build OS, architecture, .NET SDK, and native tool
      prerequisites for producing the runtime package.
- [ ] Document exact moonlib build commands from repository root.
- [ ] Document where `moonlib.dll`, runtimeconfig, CSPICE, GDAL, PROJ, and other
      runtime files are expected after build.
- [ ] Prototype `clr_loader.get_coreclr(dotnet_root=..., runtime_config=...)`
      against a private .NET 10 runtime directory.
- [ ] Confirm Python.NET can load and repeatedly call moonlib through that private
      runtime without a system .NET installation.
- [ ] Do not describe `dotnet publish --self-contained true` as supported unless
      the prototype overcomes Python.NET's documented limitation and tests prove it.
- [ ] If private-runtime packaging fails, prototype verified first-use download
      of the official .NET runtime archive.
- [ ] If in-process private hosting fails, document the architectural cost of a
      self-contained out-of-process worker before choosing it.
- [ ] Define the `lunarscout-runtime-linux-x64` package name, version coupling,
      dependency metadata, platform wheel tag, and import/resource layout.
- [ ] Make `lunarscout` depend automatically on the matching runtime package for
      supported Linux x86-64 installations.
- [ ] Include moonlib, MaxRev GDAL/native libraries, CSPICE, runtime configuration,
      and any approved private CoreCLR payload.
- [ ] Exclude large SPICE kernels and use D8 first-use acquisition.
- [ ] Remove obsolete `native/new_horizon` paths from examples and documentation.
- [ ] Preserve repository-root discovery as a developer override.
- [ ] Preserve explicit `LUNARSCOUT_MOONLIB_DLL` as a diagnostic/developer override.
- [ ] Make installed runtime-package discovery the default user path.
- [ ] Define and test any required runtime-directory environment variables.
- [ ] Decide whether `LD_LIBRARY_PATH` and `LD_PRELOAD` requirements are supported
      contract, temporary workaround, or unsupported host-specific guidance.
- [ ] Ensure `ls.native.status()` reports each missing component accurately.
- [ ] Ensure `status()` does not initialize Python.NET or CLR.
- [ ] Test initialization success and failure in fresh subprocesses.
- [ ] Test repeated supported native operations in one process.
- [ ] Document process shutdown/unload limitations.
- [ ] Confirm runtime-package construction does not require files from Lunar Analyst.
- [ ] Confirm native tests do not depend on a developer-specific scenario path.
- [ ] Categorize native tests as CPU-required, GPU-required, or real-data-required.
- [ ] Keep CPU-compatible tests in ordinary native CI.
- [ ] Put GPU and real-data tests behind explicit opt-in markers/workflows.

### Native Dependency Evidence

- [ ] Record NuGet package versions from the resolved assets/lock file.
- [ ] Decide whether NuGet lock files are committed and locked mode is used in CI.
- [ ] Record .NET SDK and runtime versions.
- [ ] Record the exact private .NET runtime archive/package version and checksum
      if CoreCLR is redistributed.
- [ ] Record MaxRev GDAL runtime version.
- [ ] Record CSPICE source/version/checksum and redistribution status.
- [ ] Record HDF5 involvement and confirm it is not loaded in-process unless the
      selected D10 design requires it.
- [ ] Generate a native build manifest or verification report.

### Verification

```bash
dotnet restore native/moonlib/moonlib.csproj
dotnet build native/moonlib/moonlib.csproj --no-restore -v minimal
dotnet test native/tests/HorizonGen.Tests/HorizonGen.Tests.csproj -v minimal
LUNARSCOUT_MOONLIB_DLL=/absolute/path/to/moonlib.dll \
  .venv/bin/python -c "import lunarscout as ls; print(ls.native.status())"
```

### Completion Criteria

- [ ] A clean supported build host can produce the platform runtime wheel using
      only documented inputs.
- [ ] Native CPU-compatible tests pass from that build.
- [ ] `pip install lunarscout` installs and discovers the matching runtime package.
- [ ] Native initialization succeeds without a separately installed .NET runtime,
      or D4 is explicitly revised based on prototype evidence.
- [ ] Missing and incompatible runtime components produce accurate diagnostics.
- [ ] Native limitations and unsupported platforms are explicit.

## 10. Workstream G: CI and Required Checks

**Objective:** run reproducibility and truth checks automatically without
blocking rapid checkpoint commits; make the complete suite mandatory for release
candidates.

**Prerequisites:** Decisions D2, D3, D6, and D7; stable commands from Workstreams
B-F.

### Workflow Design

- [ ] Add a Python fast-test workflow for each supported Python version.
- [ ] Build artifacts once and use the wheel for clean-install jobs.
- [ ] Add complete-install clean-wheel tests.
- [ ] Add Linux platform-runtime package build and install tests.
- [ ] Add sdist rebuild/install verification.
- [ ] Add deterministic example smoke tests categorized by required capabilities
      and data.
- [ ] Add lazy-native-import subprocess tests.
- [ ] Add a Linux native build job.
- [ ] Add CPU-compatible native tests.
- [ ] Cache Python and NuGet downloads using lock-aware keys.
- [ ] Ensure CI also succeeds with empty caches.
- [ ] Add Ruff or the selected quality tool.
- [ ] Add the selected warning policy.
- [ ] Add documentation path/link checks.
- [ ] Add package metadata and artifact-content checks.
- [ ] Add dependency vulnerability reporting according to D6/D7.
- [ ] Upload test reports, artifacts, hashes, and environment manifests for
      release-candidate runs.
- [ ] Add a scheduled or manual real-native workflow placeholder with explicit
      secrets/data requirements.
- [ ] Prevent workflows from downloading or publishing real scenario data.
- [ ] Set timeouts for native and example jobs.
- [ ] Make failure messages identify the local reproduction command.

### Check Policy

- [x] Do not require checks before ordinary checkpoint commits during rapid
      single-developer development.
- [ ] Run fast Python and Ruff checks automatically on pushes where practical.
- [ ] Run native build/test automatically when native source changes where practical.
- [ ] Require Python tests, artifact construction, clean installation, native
      build/tests, quality checks, and documentation checks for release candidates.
- [ ] Keep real GPU/scenario validation manual or scheduled unless infrastructure
      exists to make it reliable.

### Reproducibility Safeguards

- [ ] Pin action versions by immutable commit or approved major version policy.
- [ ] Print Python, pip/build frontend, .NET, GDAL/Rasterio, pyproj, NumPy, and OS
      versions in job logs or manifests.
- [ ] Avoid hidden setup from a developer's prebuilt `.venv` or native `bin/obj`.
- [ ] Test build jobs after deleting generated outputs.
- [ ] Use fresh temporary directories for artifact installation and examples.
- [ ] Retain artifact SHA-256 hashes.

### Completion Criteria

- [ ] Informational checks run on configured pushes/manual workflows.
- [ ] Complete checks run and pass for every release candidate.
- [ ] CI starts from a clean checkout with no prebuilt package/native outputs.
- [ ] A contributor can reproduce each failing job with a documented command.
- [ ] Release-candidate workflows retain sufficient evidence for audit.

## 11. Workstream H: Warning and Quality Baseline

**Objective:** prevent dependency drift from turning known warnings into future
failures.

**Prerequisites:** Decision D7.

### Python Tasks

- [ ] Fix NumPy scalar-conversion deprecations in scenario coordinate conversion.
- [ ] Identify every pyproj CRS-to-PROJ warning source.
- [ ] Preserve WKT as authoritative and remove unnecessary PROJ.4 conversions.
- [ ] Keep intentional incomplete-georeference Rasterio warnings scoped to their
      fixture tests.
- [ ] Remove broad warning suppression from examples where possible.
- [ ] Add narrow filters only for warnings that are understood and unavoidable.
- [ ] Run focused supported-installation tests with unexpected deprecations
      treated as errors.
- [ ] Add Ruff configuration limited to agreed rules.
- [ ] Fix newly selected lint failures without unrelated formatting sweeps.
- [ ] Add scoped type checks if approved.

### Native Tasks

- [ ] Enumerate the current nullable-reference warnings in the committed build.
- [ ] Fix warnings that represent credible null dereferences.
- [ ] Document or narrowly suppress warnings that are proven false positives.
- [ ] Prevent the warning count from increasing.
- [ ] Decide whether release builds use warnings-as-errors for selected warning IDs.
- [ ] Separate restricted-network vulnerability lookup warnings from compiler
      warnings in CI reporting.

### Completion Criteria

- [ ] NumPy deprecation warnings are eliminated.
- [ ] Expected geospatial warnings are narrowly scoped and documented.
- [ ] CI rejects unexpected new Python warnings.
- [ ] Native warning count is zero or has an approved, non-growing baseline.
- [ ] Static-quality commands are documented and automated.

## 12. Workstream I: Documentation Alignment

**Objective:** ensure a user following primary documentation gets the tested
package behavior.

**Prerequisites:** Decisions D1-D4 and stable results from Workstreams C-F.

### README

- [ ] Replace `pip install -e packages/lunarscout` with the correct standalone
      command.
- [ ] Fix the examples link.
- [ ] Replace former source-layout paths.
- [ ] Describe the complete user package accurately.
- [ ] Describe the single complete installation and separate developer toolchain.
- [ ] State supported Python, operating system, and architecture combinations.
- [ ] State that the platform runtime package is installed automatically on
      supported Linux x86-64 systems.
- [ ] Link to the user guide and native setup section.

### User Guide

- [ ] Reconcile installation commands with the single runtime dependency set.
- [ ] Separate Rasterio's Python GDAL runtime from moonlib's MaxRev GDAL runtime.
- [ ] Add a user install section and a separate contributor/runtime-build section.
- [ ] Document every required native environment variable.
- [ ] Correct the description of `scenario.py` as a high-level scenario workflow
      facade rather than path helpers only.
- [ ] Mark provisional and experimental features clearly.
- [ ] Mark HDF5 storage examples experimental unless D10 selects HDF5 as the
      canonical format.
- [ ] Document current temporal-series integrity semantics accurately.
- [ ] Replace packaging TODOs with current limitations or tracked follow-up work.
- [ ] Update verification commands to use the repository virtual environment and
      the same commands as CI.

### Examples Documentation

- [ ] Replace the old `PYTHONPATH` setup.
- [ ] Replace `native/new_horizon` runtime paths.
- [ ] State runtime/data prerequisites for every example.
- [ ] State which examples are CI acceptance tests.
- [ ] State which examples require real scenarios, GPU/native runtime, network,
      or large storage.
- [ ] Keep performance numbers labeled with machine/runtime context.
- [ ] Ensure examples use only public APIs unless explicitly labeled internal
      validation tools.

### Contributor and Release Documentation

- [ ] Add a concise contributor verification section or file.
- [ ] Document how to build artifacts.
- [ ] Document wheel-first smoke testing.
- [ ] Document native CPU versus GPU/real-data test tiers.
- [ ] Document large local artifact reporting and cleanup choices.
- [ ] Add a release checklist linked to this milestone.
- [ ] Add a known-limitations section for unsupported native platforms and unload
      behavior.

### Documentation Checks

- [ ] Add a test/search that rejects `packages/lunarscout` in primary docs.
- [ ] Add a test/search that rejects `native/new_horizon` in primary docs.
- [ ] Add relative-link validation.
- [ ] Validate documented shell commands in CI where practical.
- [ ] Review every install snippet against a fresh environment.

### Completion Criteria

- [ ] README quick start works from a clean checkout.
- [ ] The documented user installation and developer setup both work.
- [ ] Native setup works on the declared supported host.
- [ ] No primary document contradicts package metadata.
- [ ] Known limitations are explicit rather than implied by TODOs.

## 13. Workstream J: Asset and License Minimum Baseline

**Objective:** know what source and binary artifacts are distributed before a
public package is published.

**Prerequisites:** Decision D8.

### Tasks

- [ ] List every tracked binary and large data file.
- [ ] Record file size and SHA-256 hash.
- [ ] Record source URL or origin.
- [ ] Record version/date and purpose.
- [ ] Record license or redistribution terms.
- [ ] Decide repository, sdist, wheel, download-on-demand, or remove for each file.
- [ ] Confirm `data/product_overview.tif` Git LFS behavior in source archives.
- [ ] Review bundled SPICE kernels against the Python checksummed download model.
- [ ] Review `native/moonlib/cspice.dll` for platform relevance and redistribution.
- [ ] Review the DSN Excel workbook and derived text files.
- [ ] Strip generated outputs from tracked notebooks or exclude them from artifacts.
- [ ] Add `THIRD_PARTY_NOTICES` or an equivalent inventory if required.
- [ ] Add artifact-content tests for prohibited binary types in the wheel.

### Completion Criteria

- [ ] Every shipped non-source asset has documented provenance.
- [ ] Wheel and sdist asset content matches D8.
- [ ] No unsupported platform binary is included accidentally.
- [ ] Required license and notice files are present in artifacts.

## 14. Workstream K: Unified Local Verification

**Objective:** provide one reliable contributor entry point that mirrors CI.

### Tasks

- [ ] Choose a script, Makefile, task runner, or documented command set.
- [ ] Add a fast Python verification target.
- [ ] Add an artifact build target.
- [ ] Add a clean-wheel test target.
- [ ] Add a native CPU target.
- [ ] Add an "all available checks" target that reports skipped capabilities.
- [ ] Add explicit timeouts to long-running targets.
- [ ] Print tool and runtime versions.
- [ ] Use temporary output directories outside the source tree by default.
- [ ] Never delete existing scenario or test data implicitly.
- [ ] Return a nonzero exit code for failed required checks.
- [ ] Keep commands identical to CI implementation where possible.

### Suggested Command Contract

The exact interface is a decision, but it should offer equivalents of:

```text
verify python
verify package
verify native
verify all
```

### Completion Criteria

- [ ] A new contributor can run the fast baseline with one documented command.
- [ ] Release operators can run the complete available baseline with one command.
- [ ] Output distinguishes passed, failed, skipped, and unavailable checks.
- [ ] CI invokes the same underlying commands.

## 15. Workstream L: Release-Candidate Acceptance

**Objective:** prove the milestone against built artifacts and retain evidence.

**Prerequisites:** Workstreams A-K complete.

### Candidate Preparation

- [ ] Start from a clean checkout of the candidate commit.
- [ ] Confirm no untracked source or generated build files influence the build.
- [ ] Build wheel and sdist using the approved toolchain.
- [ ] Generate SHA-256 hashes.
- [ ] Generate an artifact content report.
- [ ] Generate an environment/tool version report.
- [ ] Generate runtime and developer dependency lists.
- [ ] Confirm changelog and version agree.

### Candidate Verification

- [ ] Install the Lunarscout and Linux runtime wheels on every supported Python
      combination.
- [ ] Run complete smoke workflows.
- [ ] Install from sdist and repeat the designated smoke test.
- [ ] Build the Linux platform runtime package on the supported host.
- [ ] Run CPU-compatible native tests.
- [ ] Run native Python status/initialization smoke from the installed wheel.
- [ ] Confirm lazy import behavior.
- [ ] Run documentation link/path checks.
- [ ] Run warning and quality gates.
- [ ] Run vulnerability reporting and adjudicate findings.
- [ ] Record skipped GPU and real-scenario tests explicitly.
- [ ] If required by D2/D6, run one real-scenario validation and attach its report.

### Candidate Review

- [ ] Review all CI jobs and retained artifacts.
- [ ] Review known limitations.
- [ ] Review third-party asset inventory.
- [ ] Review package metadata as rendered by the target package index.
- [ ] Install using only the published candidate instructions.
- [ ] Obtain owner approval for promotion.

### Release Evidence

- [ ] Source commit SHA.
- [ ] Version and release channel.
- [ ] Wheel and sdist SHA-256 hashes.
- [ ] Python, OS, and .NET versions tested.
- [ ] Direct and resolved dependency versions.
- [ ] Python/native test totals and skips.
- [ ] Warning baseline.
- [ ] Artifact content and size report.
- [ ] Native runtime manifest.
- [ ] Known limitations and deferred test tiers.

### Completion Criteria

- [ ] All milestone exit criteria in Section 1 are checked.
- [ ] All blocking decisions are recorded.
- [ ] All required CI checks pass on the candidate commit.
- [ ] Candidate artifacts install using only documented commands.
- [ ] Evidence is retained at the approved release location.
- [ ] The release is promoted or the failure is recorded with a follow-up owner.

## 16. Recommended Execution Order

The work should proceed in this order to avoid rebuilding CI and documentation
around unsettled policy:

1. [ ] Resolve remaining open parts of Decisions D1-D10, prioritizing D4, D5,
       D6, D8, and D10.
2. [ ] Complete Workstream A: governance and baseline inventory.
3. [ ] Complete Workstream B: restore a green source tree.
4. [ ] Complete Workstream C: align runtime and developer dependencies.
5. [ ] Complete Workstream D: construct and inspect artifacts.
6. [ ] Complete Workstream E: add wheel-first clean-install testing.
7. [ ] Complete Workstream F: establish the Linux platform runtime package.
8. [ ] Complete Workstream H: establish warnings and static-quality baseline.
9. [ ] Complete Workstream G: automate the stable commands in CI.
10. [ ] Complete Workstream I: align user and contributor documentation.
11. [ ] Complete Workstream J: finish minimum asset/license inventory.
12. [ ] Complete Workstream K: unify local verification.
13. [ ] Complete Workstream L: run release-candidate acceptance.

Workstreams H, I, and J can begin earlier after their decisions are resolved,
but their final text and gates should be validated against the built artifacts,
not assumptions.

## 17. Progress Dashboard

| Area | Decision complete | Implementation complete | Verification complete |
| --- | --- | --- | --- |
| Governance and baseline | [ ] | [ ] | [ ] |
| Green source tree | [ ] | [ ] | [ ] |
| Runtime/developer dependencies | [ ] | [ ] | [ ] |
| Package metadata/artifacts | [ ] | [ ] | [ ] |
| Clean-install tests | [ ] | [ ] | [ ] |
| Linux platform runtime | [ ] | [ ] | [ ] |
| CI and release checks | [ ] | [ ] | [ ] |
| Warning/quality baseline | [ ] | [ ] | [ ] |
| Documentation alignment | [ ] | [ ] | [ ] |
| Asset/license baseline | [ ] | [ ] | [ ] |
| Unified local verification | [ ] | [ ] | [ ] |
| Release-candidate acceptance | [ ] | [ ] | [ ] |

## 18. Decision Record Template

Use this template for each decision so later implementation does not have to
infer intent from code changes:

```markdown
### Dn: Decision title

- Status: proposed | accepted | superseded
- Date:
- Owner:
- Decision:
- Supported alternatives considered:
- Rationale:
- Consequences:
- Workstreams unblocked:
- Follow-up date, if provisional:
```

## 19. Final Milestone Sign-Off

- [ ] Package metadata is truthful.
- [ ] Documentation is truthful.
- [ ] Runtime and developer dependencies are separated truthfully.
- [ ] Supported artifacts build from a clean checkout.
- [ ] Supported artifacts install into fresh environments.
- [ ] Tests execute against installed artifacts.
- [ ] Linux platform-runtime construction and installation are reproducible.
- [ ] CI enforces the documented baseline.
- [ ] Artifact content and third-party assets are understood.
- [ ] Release evidence is complete and retained.
- [ ] Deferred work is labeled and does not masquerade as supported behavior.

When every item above is complete, Lunarscout has reached the milestone of a
truthful and reproducible package. It may still be pre-1.0 and native support
may remain Linux-only, but a user and contributor can rely on the package
saying exactly what it supports and proving those claims from a clean
environment.
