# Fresh Project Review

**Review date:** 2026-07-13

**Repository state reviewed:** current working tree at commit `59bd42a`, including
the local uncommitted changes present during the review

**Primary references:** `docs/USER_GUIDE.md`, `docs/roadmap.md`, the public
Python package in `src/lunarscout`, Python tests in `tests`, and the native C#
implementation and tests under `native`

## Executive Assessment

Lunarscout is a promising and already useful analyst library, not merely a
package skeleton. Its strongest qualities are the explicit treatment of
geospatial metadata, unusually careful file publication behavior, lazy native
runtime initialization, structured errors, and extensive behavioral tests.
The core Python raster and temporal APIs are coherent enough for real scripts
and notebooks. The examples are substantive acceptance tests rather than
decorative snippets. The native layer also contains meaningful regression,
streaming, and output-contract coverage.

The project is nevertheless better described as a well-tested development
library than as a releasable standalone product. The largest gaps are around
packaging truth, automated verification, native distribution and ownership,
documentation consistency, and scientific product provenance. These are not
reasons to redesign the core API. They are reasons to consolidate what already
works and establish reliable release boundaries around it.

The immediate recommendation is to define a supported pure-Python installation
and make it installable and documented as such, then create a separately gated
Linux native profile. CI should enforce both profiles. Native source trimming,
storage-format selection, and scenario-state expansion should follow rather
than compete with that baseline.

### Readiness Summary

| Area                               | Assessment                     | Notes                                                                        |
| ---------------------------------- | ------------------------------ | ---------------------------------------------------------------------------- |
| Core Python API                    | Strong development quality     | Clear concepts, structured errors, broad tests                               |
| Raster/geospatial correctness      | Strong                         | Explicit grids and metadata; warning cleanup remains                         |
| Temporal arrays and current series | Strong but transitional        | Safe publication and streaming; target format is undecided                   |
| SPICE API                          | Useful, provisional            | Good lazy loading; dependencies and provenance need work                     |
| Native API                         | Capable, high operational risk | Strong tests, difficult runtime and distribution story                       |
| Documentation                      | Extensive but inconsistent     | User guide is valuable; README and examples retain extraction-era paths      |
| Packaging                          | Not release-ready              | Metadata contradicts installation and optional-feature claims                |
| CI/release engineering             | Missing                        | No automated build, test, wheel, or platform matrix                          |
| Scientific assurance               | Promising, incomplete          | Good synthetic/reference tests; product-level provenance needs strengthening |

## Verification Performed

The review used the repository-managed Python environment and isolated the
committed native tree where necessary because the working tree contained an
unfinished native feature.

- `.venv/bin/python -m pytest -q`: **256 passed, 1 skipped** in 7.03 seconds.
  The run emitted 117 warnings: mostly pyproj CRS-to-PROJ conversion warnings,
  four Rasterio non-georeferenced warnings from intentional fixtures, and ten
  NumPy deprecation warnings in scenario coordinate conversion tests.
- `dotnet build native/moonlib/moonlib.csproj`: **failed in the current working
  tree** with `CS1106` at `native/moonlib/pipeline/LightmapThresholds.cs:18`.
  This file is untracked, so this is an in-progress worktree blocker rather than
  a defect in commit `59bd42a`.
- An archived copy of commit `59bd42a` built successfully: **0 errors, 16 C#
  nullable-reference warnings**. NuGet vulnerability lookup also warned because
  the network service was unavailable.
- The archived native tests passed: **142 passed, 1 skipped** in approximately
  eight seconds.
- `pip check`: **no broken requirements** in the existing environment.
- A clean isolated wheel build could not be completed in this environment
  because build isolation could not reach PyPI and the active virtual
  environment does not contain an importable `setuptools.build_meta`. This is
  an environment limitation, but it also demonstrates why a CI packaging job
  is needed.

This review did not rerun the long real-scenario GPU validations documented in
the roadmap. Their recorded evidence was considered, but it is not equivalent
to a fresh independent reproduction.

## What Is Working Well

### 1. The data model makes important constraints explicit

`GeoReference` carries CRS, affine transform, dimensions, and nodata metadata
instead of hiding them behind array subclasses. `same_grid` and
`require_same_grid` reject the common mistake of treating matching array shapes
as matching spatial grids. `TemporalCube` similarly makes the `(time, y, x)`
contract and UTC time axis explicit. These are appropriate choices for a
scientific library because they make invalid combinations harder to express
accidentally.

The API also generally validates inexpensive Python-side inputs before native
initialization. That keeps errors deterministic and preserves the documented
lazy-runtime boundary.

### 2. File-producing operations are designed defensively

GeoTIFF writes, temporal-series publication, native temporal generation, and
PSR generation use staging and explicit overwrite behavior. The tests exercise
failure cleanup and preservation of an existing destination. This is one of
the strongest parts of the implementation. Large scientific products are
expensive to regenerate, so overwrite preservation and cancellation behavior
are correctness requirements, not optional polish.

The temporal-series reader also validates manifests, paths, layer metadata,
time ordering, and the completion marker. Its bounded dataset and layer caches
are practical and avoid pretending that a file-backed cube is an in-memory
array.

### 3. The native runtime is lazy and diagnosable

`import lunarscout` does not initialize Python.NET, CLR, GDAL native bindings,
SPICE, or moonlib. Tests enforce that boundary in subprocesses. `ls.native.status()`
separates Python.NET, .NET, moonlib, CSPICE, and GDAL status, and structured
exceptions carry stable codes and detail dictionaries. This is a sound basis
for a difficult cross-runtime integration.

### 4. Tests focus on behavior and lifecycle

The Python suite covers more than nominal numerical results. It tests path
containment, atomic overwrite behavior, cancellation, cleanup, native fakes,
manifest tampering, cache behavior, example execution, and the dependency
boundary from Lunar Analyst. The deterministic examples run as subprocesses,
which catches import and workflow failures that unit tests often miss.

The C# suite covers horizon formats, compression, grid conversion, streaming
bridges, progress contracts, validity masks, regression scenarios, and
reference implementations. The archived committed tree passing 142 native
tests is meaningful evidence that the extracted implementation is not wholly
dependent on the former application repository.

### 5. Examples and documentation expose real workflows

The ordered example suite covers terrain, regions, alignment, temporal data,
streaming reductions, native generation, QGIS inspection, end-to-end
validation, and performance measurement. The user guide is candid about
provisional areas and documents value semantics such as PSR bytes, masks,
temporal dimensions, and native storage choices. This is much better than an
API reference without workflow context.

## Prioritized Findings

### P0: The current worktree does not build natively

The untracked `LightmapThresholds` implementation declares an extension method
inside a non-static class, causing `CS1106`. Python tests still pass because
they do not compile the native project. The committed tree builds, so the
correct response is to finish or isolate this feature before treating the
working tree as an integration candidate.

**Recommendation:** require a local `dotnet build` and focused native test run
before committing the threshold feature. Once CI exists, native compilation
must be a required check so Python-only success cannot mask this class of
failure.

### P1: Packaging metadata and installation documentation contradict each other

This is the clearest release blocker.

- `pyproject.toml` makes `pythonnet`, `h5py`, and `hdf5plugin` mandatory.
- `docs/USER_GUIDE.md`, `README.md`, `requirements.in`, and bootstrap error text
  describe Python.NET as optional and instruct users to install a `native`
  extra, but `pyproject.toml` defines no `native` extra.
- HDF5 packages are used only by the experimental two-file example, yet every
  base installation receives them.
- Public SPICE functions require `spiceypy`, but it is not declared in project
  dependencies or an extra.
- Public DataFrame and plotting helpers require pandas and Matplotlib, also not
  declared in an extra.
- `README.md` and `examples/README.md` still use the former
  `packages/lunarscout` source layout. The README link to the examples also
  points outside the current repository.
- The guide says GDAL Python bindings are externally supplied, while the actual
  Python implementation imports Rasterio and declares it as a normal
  dependency. Native GDAL remains a separate concern, but the current wording
  conflates the two stacks.

**Recommendation:** choose and enforce an extras model, for example:

```toml
dependencies = ["numpy", "pyproj", "rasterio", "scipy"]

[project.optional-dependencies]
spice = ["spiceypy"]
plot = ["matplotlib", "pandas"]
native = ["pythonnet"]
hdf5 = ["h5py", "hdf5plugin"]
dev = ["pytest", "build", "ruff", "mypy"]
```

Exact grouping can differ, but code, metadata, errors, and documentation must
describe the same install profiles. Add a clean-wheel smoke test for the base
package and each supported extra.

### P1: There is no CI or release gate

No GitHub Actions or equivalent configuration is present. As a result, the
project has strong tests but no assurance that they run on every change, from a
clean checkout, against declared dependencies. There is also no wheel/sdist
build check, documentation link check, lint/type check, coverage report,
dependency audit, or native build matrix.

This gap explains several current inconsistencies: an unfinished C# file can
coexist with a green Python suite, stale installation paths remain in primary
documentation, and nullable/deprecation warnings accumulate without a policy.

**Recommendation:** add CI in layers:

1. Python 3.11 and 3.12: install base package from a built wheel, run tests and
   deterministic examples, and assert lazy native imports.
1. Native Linux: build `moonlib` and run the CPU-compatible native suite.
1. Packaging: build wheel and sdist, inspect metadata and wheel contents, and
   install into a fresh environment.
1. Quality: Ruff or equivalent, a scoped type checker configuration, warning
   budgets, documentation links, and dependency vulnerability reporting.
1. Scheduled/manual integration: real scenario, native runtime, and GPU tests
   with archived machine-readable evidence.

### P1: The native ownership and distribution boundary is still too broad

The guide correctly calls the current wholesale moonlib extraction temporary.
The evidence supports that assessment:

- `QuadTreeHorizonGenerator.cs` is about 5,000 lines and `Lightmaps.cs` is over
  1,600 lines in the current worktree.
- A simple source scan finds more than 2,000 public C# declarations across
  moonlib, far beyond the intended Python-facing surface.
- Legacy utilities, display helpers, embedded kernels, DSN files, old math
  types, and unfinished or `NotImplementedException` code remain in the native
  tree.
- The C# project targets `net10.0` with a fixed `linux-x64` runtime identifier.
  The repository also contains a Windows `cspice.dll`, while Linux CSPICE is
  expected from an external `third_party` path or runtime directory.
- The Python bootstrap searches repository build directories or an explicit
  `LUNARSCOUT_MOONLIB_DLL`; installed wheels do not contain a native payload.
- The roadmap records an intermittent segmentation fault during
  `pythonnet.unload()`, which is a material lifecycle risk even if ordinary
  operation is stable.

There is also a boundary inconsistency: documentation says production moonlib
access is behind `MoonlibBridge`, but `GenerateHorizons` imports moonlib horizon
types and invokes `QuadTreeHorizonGenerator` directly. Either this is an
intentional low-level exception or the bridge contract is not actually
enforced.

**Recommendation:** define a small supported native ABI/API inventory. Move
each production operation behind `MoonlibBridge` or explicitly document the
few justified exceptions. Mark or internalize unrelated public C# types, split
large classes by responsibility, and decide whether the first supported native
profile is source-built Linux only. Do not promise native wheels until CSPICE,
GDAL, HDF5, .NET, license, and runtime-loading behavior are reproducible in a
clean environment.

### P1: Project direction is inconsistent across governing documents

`AGENTS.md` and the current user guide prohibit scenario database mutation and
describe Lunarscout as independent of application state. `docs/roadmap.md`
instead says Lunarscout may create, mutate, and delete scenarios and write
`scenario.db`. The roadmap is acknowledged as old, but its progress snapshot
presents the changed direction as current. The user guide also describes
`scenario.py` as "filesystem-safe scenario path helpers only," even though the
class now contains native terrain generation, horizon parsing and caching,
SPICE overlays, plotting, temporal generation, and PSR production.

This is not merely editorial. Scenario ownership determines the dependency
boundary, concurrency model, protected paths, provenance rules, and what
downstream applications may assume.

**Recommendation:** record a short architecture decision stating whether
Lunarscout owns only scenario-contained files or also owns a shared scenario
database. Update `AGENTS.md`, the roadmap status, the guide, and examples to one
answer. Until that decision changes explicitly, keep the current code boundary:
no Lunar Analyst imports and no application database mutation.

### P1: Scientific product provenance is not yet strong enough for durable outputs

The library documents many scientific semantics well, but several are carried
primarily in prose or caller-supplied metadata rather than in the product:

- PSR output semantics include a fixed 1970-2044 six-hour sampling interval and
  zero observer elevation. These assumptions should be machine-readable output
  metadata, not only guide text.
- Native temporal provenance currently identifies the generator and signal,
  but should include Lunarscout/moonlib versions, operation parameters, input
  identities, kernel manifest/version, observer height, time range, and relevant
  algorithm version.
- The temporal `COMPLETE` digest protects the manifest bytes, not the layer
  contents. Layer metadata is revalidated on open, but silent pixel corruption
  with unchanged metadata will not be detected. Calling this a completion
  digest is accurate as a publication marker, but it is not full product
  integrity.
- Downloaded map products are staged but have no declared checksum, expected
  size, timeout, or content validation. SPICE kernel downloads are better
  because they use manifest checksums.

**Recommendation:** define a common provenance schema for durable products and
embed it in GeoTIFF tags and/or a sidecar manifest. Add catalog checksums and
sizes for map products. For temporal series, either add optional per-layer
hashes or clearly document the completion marker as structural rather than
content integrity; hashing policy should account for very large products.

### P2: Several modules have become ownership catch-alls

The public facade is approachable, but internal modules are becoming difficult
to maintain:

- `scenario.py` is about 1,300 lines and combines path security, native terrain
  generation, horizon binary decoding and file-handle caching, coordinate
  conversion, plotting, SPICE geometry, and temporal/native product orchestration.
- `temporal_store.py` is about 1,280 lines and combines format encoding,
  publication, VRT generation, validation, reader caches, and reductions.
- `native_temporal.py` is about 770 lines and combines allocation policy,
  Python.NET type construction, streaming protocol, tile validation, scratch
  storage, and output publication.
- Native classes are substantially larger and mix GPU kernels, buffer pools,
  file I/O, pipelines, and public orchestration.

The answer is not a broad refactor. Split modules only along existing behavior
boundaries while preserving the public facade. For example, scenario horizon
I/O/cache, scenario plotting, and native scenario operations can be private
helpers composed by `Scenario`. Temporal manifest codec and publication logic
can be separated from reader/cache behavior. In C#, isolate session/buffer
lifetime from compute kernels and product writers.

### P2: API stability policy is looser than the package version suggests

The package declares version `0.1.0`, the changelog claims Semantic Versioning,
and the guide defines a broad public API that includes exported names,
documented module members, written file formats, exceptions, and error codes.
At the same time, the roadmap explicitly defers selecting a stable v0.1 surface
and allows continued refinement without a near-term freeze.

The curated root exports 87 names, including many exception types and a
PascalCase low-level `GenerateHorizons` function. Every documented error code
and manifest format increases compatibility cost. This can be managed, but it
should be intentional.

**Recommendation:** label APIs as stable, provisional, or experimental now,
even before 1.0. Keep the root surface focused on normal analyst workflows;
native protocol and diagnostic types can remain under `ls.native`. Add
`__version__` or an equivalent supported version query, document format-version
migration policy, and require changelog entries for changes to error codes or
durable formats.

### P2: Warning debt and static quality controls need attention

The Python suite passes with 117 warnings. The repeated CRS-to-PROJ conversion
warning indicates the code and fixtures still depend on lossy PROJ.4
representations even though WKT is authoritative. The scenario warnings come
from converting non-scalar NumPy results to Python scalars and are scheduled to
become errors in a future NumPy release. The committed C# build emits 16
nullable-reference warnings in native lighting code.

There is no linter, formatter policy, type-checker configuration, or warning
budget in project metadata. Type annotations are widespread, so a scoped mypy
or pyright rollout should provide value. If typed use by downstream projects is
intended, add and ship `py.typed` after the public annotations are checked.

**Recommendation:** fix the NumPy deprecations first, stop unnecessary PROJ.4
round-trips where possible, and classify remaining expected warnings in tests.
Adopt static tools incrementally by module rather than requiring a repository-
wide cleanup before CI can start.

### P2: The storage redesign needs an explicit compatibility decision

The current timestamped-GeoTIFF series is thoughtfully implemented and tested,
but the roadmap says it is no longer the target for large time series. The
two-file BigTIFF/HDF5 representation is still an example prototype, yet its
dependencies are already mandatory in the base package and the example is in
the deterministic suite.

The proposed dual-copy format can be appropriate for CephFS access patterns,
but it doubles logical data and creates consistency, repair, provenance, and
atomic-publication obligations. Those costs should be demonstrated against the
single-BigTIFF baseline and representative NRP storage before the format becomes
public API.

**Recommendation:** keep the prototype explicitly experimental and move its
dependencies to an extra. Define canonical authority when the TIFF and HDF5
copies disagree, staged commit/recovery behavior, checksums or generation IDs,
versioning, and migration from the current series before implementing the
production reader/writer.

### P2: Repository assets need provenance and size governance

The committed tree is approximately 102 MiB, dominated by SPICE kernels and
other native static assets. It includes binary kernels, an Excel workbook, a
Windows DLL, a large notebook with outputs, and a Git LFS overview raster. Only
the project Apache license is present; there is no third-party notices or asset
provenance document in the repository.

Some or all of these assets may be redistributable, but the project should not
leave that as an assumption. Large embedded kernels also duplicate the newer
Python manifest/download model and make clones and source distributions heavier.

**Recommendation:** inventory every binary/data asset with source URL, version,
license or redistribution terms, checksum, purpose, and packaging decision.
Remove generated notebook outputs, editor lock files such as
`examples/.#notebook_example.py`, and assets not required by the supported
runtime. Prefer verified downloads for large replaceable kernels unless offline
operation is an explicit requirement.

### P3: Local checkout hygiene is poor, although ignored correctly

The review checkout contains roughly 183 GiB under ignored native `TestResults`
and about 14 GiB in loose Git objects. These files are not tracked, so this is
not a source-tree defect, but it can materially affect developer disk space,
test discovery, backups, and tooling performance.

**Recommendation:** document native test artifact cleanup and consider directing
large validation outputs outside the repository by default. A maintenance
script may report artifact locations and sizes, but it should never delete data
implicitly.

## Architecture Review

### Public Python surface

The `import lunarscout as ls` facade is appropriate for notebook users. Naming
is mostly direct, and returning `(array, georef)` is consistent across raster
operations. The split between root analyst functions and `ls.native` runtime
details is also sound.

The main concern is surface growth. Exceptions, low-level native wrappers,
catalog utilities, plotting functions, storage writers, and domain objects all
share the root namespace. Apply a simple admission rule: root exports should be
common in ordinary analyst scripts; diagnostics, protocols, and experimental
formats should live in domain submodules.

### Raster, terrain, alignment, and regions

This is the most mature subsystem. Input shape and dtype validation is
consistent, CRS comparison is semantic, rotated bounds are handled, nodata is
explicit, and operations return georeferencing with the result. Tests cover
supported dtypes, metadata failures, resampling names, rotated grids, region
connectivity, cleanup, and output conventions.

Remaining work is chiefly documentation and compatibility: publish exact
nodata behavior, units, edge handling, connectivity defaults, supported CRS
assumptions, and the distinction between Rasterio's GDAL runtime and moonlib's
MaxRev GDAL runtime. Add a small set of golden rasters produced independently
of the implementation for release-level numerical regression.

### Temporal model and persistence

The UTC-aware `TimeRange` and immutable coordinate handling are clear. The
storage API correctly refuses to expose `.values` for a file-backed series,
and streaming reducers preserve bounded memory. Manifest path containment and
atomic directory replacement are well considered.

The current format should be treated as supported for moderate series until a
replacement has migration tooling. Its documentation should state structural
integrity limits and concurrency semantics. A writer is single-process by
design; readers should not assume cross-process cache coherence after external
replacement.

### SPICE

Lazy `spiceypy` import, explicit kernel state, checksummed default downloads,
UTC conversion, local NED conventions, and body validation form a useful API.
The code correctly keeps pandas and Matplotlib imports lazy.

Before calling this mature, declare the `spiceypy` dependency profile, record
kernel identities in outputs, make thread/process safety expectations explicit,
and test against known external ephemeris values in addition to fakes. The
default auto-download behavior should be prominent because it introduces
network and cache side effects during geometry calls.

### Scenario

Scenario-root containment and standard-path helpers are useful, and horizon
file caching improves interactive use. The class now serves as a high-level
workflow facade, not just a path object. That can be ergonomic if internal
responsibilities are split and lifecycle is explicit.

`close_horizon_file()` and context behavior deserve emphasis because an open
cached handle can surprise long notebook sessions or Windows file replacement.
If more cached/native resources are added, consider making `Scenario` a context
manager while retaining explicit close methods.

### Native bridge and C# implementation

The buffer-stream design, bounded queues, cancellation callbacks, output
validation, and bridge smoke checks show careful engineering. CPU-compatible
tests make much of the layer verifiable without a production GPU. The roadmap's
real-scenario evidence is also valuable.

The largest risk is lifecycle complexity across Python, CLR, GDAL, CSPICE,
HDF5, worker threads, and GPU resources. Each additional native library loaded
in-process increases the chance of ABI conflicts and unsafe teardown. The
roadmap's preference for Python-owned HDF5 in-process is sensible. Keep native
sessions disposable, avoid process-global mutable state where possible, and
treat process isolation as the supported mode for combinations that cannot be
made reliably unloadable.

## Testing Review

### Coverage strengths

- Public Python behavior is exercised across every major module.
- Deterministic examples are subprocess-tested end to end.
- Native wrappers use fakes, so normal Python CI need not own a GPU or moonlib.
- Overwrite, cancellation, staging, and corruption paths receive first-class
  tests.
- The dependency direction away from Lunar Analyst is checked by AST inspection.
- Native tests include reference algorithms and production-regression cases.

### Important gaps

- No clean-wheel test proves that declared dependencies and package data are
  sufficient.
- No CI proves support for Python 3.11, even though metadata claims it.
- Native tests currently target one Linux/.NET configuration only.
- There is no coverage measurement to identify untested branches; test count
  alone is not coverage.
- Plot/DataFrame tests skip when undeclared dependencies are absent, which can
  silently remove coverage in a clean base environment.
- Real native validation is manual or local rather than scheduled and archived
  by automation.
- Scientific golden data and tolerances are not organized as a versioned,
  externally traceable validation corpus.
- Failure injection does not yet cover disk-full, permissions, process death at
  each commit phase, or concurrent writers comprehensively.
- Performance evidence is documented, but there are no regression thresholds
  or comparable machine profiles.

### Recommended test tiers

| Tier               | Runs                 | Purpose                                                 |
| ------------------ | -------------------- | ------------------------------------------------------- |
| Fast               | Every change         | Unit tests, fakes, dependency boundary, import laziness |
| Workflow           | Every change         | Deterministic examples and file lifecycle tests         |
| Native CPU         | Every native change  | Build plus CPU/emulator/native contract suite           |
| Packaging          | Every change/release | Wheel/sdist build, fresh installs, package data         |
| Real scenario      | Scheduled/manual     | GPU/native parity, cancellation, resource evidence      |
| Release validation | Release candidate    | Golden scientific products, provenance, install matrix  |

## Documentation and Developer Experience

The user guide is the right primary document, but it currently mixes tutorial,
API inventory, architecture policy, maturity statements, and future stubs. As
the project stabilizes, split it into a short installation/quick-start path,
concept guides, operation reference, native setup/troubleshooting, product
format specifications, and contributor verification.

Immediate documentation fixes should be small and factual:

1. Correct all extraction-era paths and links in `README.md` and
   `examples/README.md`.
1. Reconcile extras and dependency installation with `pyproject.toml`.
1. Update the scenario module description and resolve database ownership.
1. Separate Python Rasterio/GDAL requirements from moonlib/MaxRev GDAL.
1. Mark the HDF5 two-file example and low-level native wrappers experimental.
1. Add supported platform statements rather than leaving platform behavior
   implicit.
1. Replace TODO-heavy maturity sections with issue links or a concise known
   limitations page.

For contributors, add a single verification entry point that runs the checks
available on the host and clearly reports skipped native/integration tiers.
Document how to clean or relocate native test artifacts and how to build from a
fresh clone without relying on another Lunar Analyst checkout.

## Recommended Delivery Plan

### Phase 1: Restore a truthful, reproducible baseline

Target: a clean source checkout can build, install, and run its supported tests.

- Finish or isolate the threshold work so the current native tree builds.
- Reconcile `pyproject.toml`, `requirements.in`, guide, README, examples, and
  bootstrap messages around one extras model.
- Fix stale paths and broken documentation links.
- Add Python and native CI, including wheel/sdist smoke installation.
- Fix NumPy deprecations and establish warning budgets.
- State the first supported platform matrix explicitly.

**Exit criteria:** base wheel installs without Python.NET/HDF5 prototype
dependencies; base tests and examples pass from the wheel; the native Linux
profile builds and passes its CPU-compatible tests in CI.

### Phase 2: Freeze operational contracts

Target: current products are safe to depend on while the broader API evolves.

- Classify public APIs and durable formats as stable, provisional, or
  experimental.
- Resolve scenario-state ownership in an architecture decision.
- Define common provenance, version, parameter, and input-identity metadata.
- Add map-product checksums and clarify temporal integrity semantics.
- Move direct native horizon execution behind the supported bridge or document
  it as a deliberate exception.
- Add release notes and compatibility tests for structured error codes and
  manifest versions.

**Exit criteria:** downstream code can identify supported APIs and formats;
durable native outputs are self-describing enough to reproduce or audit; all
governing documentation agrees on scenario and native boundaries.

### Phase 3: Reduce native and internal complexity

Target: maintenance cost matches the supported feature set.

- Inventory and trim moonlib source/assets to required functionality.
- Internalize unrelated C# public types and split the largest native classes.
- Split Python catch-all modules along existing domain boundaries without
  changing the public facade.
- Address nullable warnings and formalize native resource/session lifetime.
- Inventory third-party assets and add notices/provenance.

**Exit criteria:** supported native entry points are small and enumerated;
unused legacy code/assets are removed or quarantined; build warnings are
intentional and bounded.

### Phase 4: Decide the next storage format using representative evidence

Target: choose a low-file-count temporal format without prematurely burdening
the stable package.

- Benchmark BigTIFF-only and BigTIFF/HDF5 designs on local NVMe and NRP/CephFS.
- Define authority, consistency, staging, recovery, and migration semantics.
- Validate compression/plugin availability on target hosts.
- Implement the selected format behind an experimental namespace first.

**Exit criteria:** the format decision is supported by reproducible access,
storage, and recovery measurements; current-series users have a compatibility
or migration path.

## Suggested Release Gates

A first public development release should require all of the following:

- Clean checkout, wheel, and sdist builds.
- Fresh-environment install tests for every documented profile.
- Python and native CPU suites green with an explicit warning policy.
- No dependence on former Lunar Analyst repository paths.
- Correct README and installation instructions.
- Declared supported OS, Python, .NET, GDAL/Rasterio, and native-runtime ranges.
- Third-party asset inventory and redistribution review.
- Stable/provisional labels on the public API and product formats.
- Machine-readable provenance for native durable outputs.
- At least one reproducible real-scenario validation report tied to the release
  commit and runtime versions.

Native binary distribution should have additional gates:

- Reproducible payload assembly from a clean checkout.
- CSPICE, GDAL, PROJ, HDF5, and .NET dependency/license inventory.
- Verified runtime discovery without source-tree paths.
- Process-lifecycle and repeated-operation tests, including the known unload
  failure mode.
- A decision on whether native execution is supported in-process, isolated in a
  worker, or both for each operation.

## Bottom Line

Lunarscout's core design is on the right path. Explicit grids, ordinary NumPy
arrays, UTC temporal coordinates, safe publication, structured errors, and a
lazy native boundary are good foundations. The project already has enough
implementation and test depth to justify consolidation rather than another
architecture reset.

The next milestone should not be "more API." It should be a truthful and
reproducible package: consistent dependency profiles, clean CI builds, explicit
native and scenario ownership, durable provenance, and a small supported
surface. With those controls in place, the existing raster and temporal core
could credibly serve as the stable base while native products and the next
time-series format continue to mature experimentally.
