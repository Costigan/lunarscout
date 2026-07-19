# Public Product API Proposal

> Historical design record: the exact candidate signatures in this document
> are superseded by [PUBLIC_API_REVIEW.md](PUBLIC_API_REVIEW.md). In particular,
> the review uses `TimeRange`, `compress=True`, float `nodata=np.nan`, and the
> approved patch-level output-conversion contract.

Status: accepted decisions; public product facades implemented for PLAN1 M1

Last updated: 2026-07-18

This document records the proposed Python-only public API for horizon and
horizon-derived products. Decisions explicitly marked **accepted** may guide
implementation. Signatures remain provisional until the corresponding PLAN1
promotion gate is complete.

## M0 audit findings

The audit began at commit `5753112` on branch
`spike/python-horizon-port-evaluation`, with only the intentional untracked
`docs/PLAN1.md` user file.

The current public root still exports `native`, PascalCase `GenerateHorizons`,
`NativeHorizonProgress`, and the transitional `Native*` exceptions. `Scenario`
still routes horizon generation, PSR, temporal signals, and four terrain
methods through managed wrappers. Those paths are migration work and are not
used as the basis of the proposed API.

The private Python implementation exposes one CUDA horizon pipeline and focused
downstream entry points for lightmap, PSR, Sun/Earth elevation, safe haven, and
all four mission-duration calculations. Their scientific inputs are already
compatible enough to support one shared public vector/time boundary. Their
prototype progress types and exception classes differ and must be normalized
by public facades rather than re-exported.

Packaging does not yet describe the chosen architecture. `pyproject.toml`
still requires `pythonnet`, `h5py`, and `hdf5plugin`, does not require Numba,
and has no CUDA or SPICE extras. It declares Python `>=3.11` without an upper
bound or tested-platform classifiers. The first supported test matrix is
therefore Python 3.11 and 3.12 on Linux x86-64, and the first immutable package
candidate will be `0.1.0rc1`.

Most lighting examples still exercise the former native wrappers. The one
Python PSR example imports `_numba_horizon` directly. Public examples must be
converted only after the facade exists.

An import-boundary probe confirmed that `import lunarscout` does not load
Numba, Numba CUDA, SpiceyPy, Python.NET, CLR, moonlib, or HDF5. It currently
does import Rasterio through the curated root modules, but opens no dataset and
writes no file. The managed namespaces are still publicly present despite not
bootstrapping at import time.

## Accepted cross-cutting decisions

### Return values

File-producing operations return `pathlib.Path`. Backend reporting does not
change the return type.

### Backend selection

Every downstream horizon-consuming product has a keyword-only argument:

```python
backend: Literal["auto", "cpu", "cuda"] = "auto"
```

The meanings are:

- `auto` attempts to initialize the CUDA implementation and falls back to CPU
  only when CUDA capability or initialization is unavailable;
- `cpu` uses CPU execution without importing, probing, or initializing CUDA;
  and
- `cuda` requires CUDA and propagates capability, initialization, JIT, kernel,
  transfer, and synchronization failures without falling back.

Backend selection occurs once before product calculation begins. A failure
after a backend has begun calculating does not trigger an in-place backend
switch. The caller may resume the staged product later.

Horizon generation is CUDA-only and therefore does not offer a misleading
`auto` or `cpu` choice.

### Backend diagnostics and provenance

`lunarscout.cuda` will provide read-only lazy capability functions such as:

```python
ls.cuda.is_available() -> bool
ls.cuda.status() -> CudaStatus
```

`is_available()` is a capability probe, not proof of which backend completed a
specific product. Machine-readable confirmation is available from structured
progress events and persisted product metadata.

Every product manifest records the backend selected for the current run. A
resumed `auto` job may continue on a different backend. Its staged manifest and
final public metadata accumulate the ordered set of backends that durably
completed patches, for example `("cpu",)`, `("cuda",)`, or `("cpu", "cuda")`.
Backend provenance is not part of the immutable scientific job fingerprint, so
an otherwise compatible staged job can resume on another backend. Explicit
`cpu` and `cuda` requests are strict for the current invocation but may resume
patches created by an earlier invocation; the final metadata remains truthful
about both.

The completion journal must never claim backend provenance for a patch before
that patch's TIFF data is durable. Backend metadata follows the same durable
checkpoint boundary as patch completion. A failed invocation leaves accurate
resumable metadata and never changes an existing completed output.

### Verbose output and progress

Every long-running public product operation has:

```python
verbose: bool = False
progress_callback: Callable[[float], None] | None = None
progress_event_callback: Callable[[ProgressEvent], None] | None = None
```

With `verbose=False`, library code prints nothing. With `verbose=True`, it
prints immediately flushed descriptions of backend selection and major stages
to standard output. The simple callback receives a monotonic durable completion
fraction in `[0.0, 1.0]`. The structured callback receives immutable events
with operation, stage, completed and total units, fraction, selected backend,
message, and optional patch/path information.

Callback exceptions propagate from the operation. They are treated like other
operation failures: completed durable work remains resumable and incomplete
output is not published. Callbacks may run on a pipeline worker thread; GUI
callers are responsible for marshaling events to their UI thread.

### Cancellation and restart

Cancellation uses a cooperative callback:

```python
cancellation_requested: Callable[[], bool] | None = None
```

Cancellation raises `OperationCancelledError`, leaves compatible staged state,
and never publishes an incomplete product. It is checked between bounded work
units; an executing CUDA kernel is allowed to finish.

Downstream products resume compatible staged state by default. There is no
redundant public `resume=True` argument. `start_fresh=True` explicitly removes
the staged TIFF, manifest, journal, and mask before beginning. `overwrite=True`
allows replacement of a completed destination only after the new staged
product is complete. A failed overwrite preserves the old completed output.

### Vectors and times

Product functions accept either explicit Moon-ME vectors or request SPICE
generation from a time axis. Explicit vectors are C-contiguous finite
`float64[time, 3]` positions in meters and require matching UTC timestamps.
Supplying all required explicit vectors prevents SPICE import and kernel
loading.

Time-driven calls accept either `times=` or `start=`, `stop=`, and `step=`.
Explicit vectors take precedence over generated vectors, but conflicting or
incomplete inputs are rejected rather than silently ignored. Exact per-sample
UTC-to-ET conversion is the default.

## Naming table

The proposed root functions and focused-module functions use the same names:

| Root and focused-module name | `Scenario` method |
| --- | --- |
| `generate_horizons` | `generate_horizons` |
| `generate_lightmap` | `lightmap` |
| `generate_psr` | `psr` |
| `generate_sun_elevation` | `sun_elevation` |
| `generate_earth_elevation` | `earth_elevation` |
| `generate_safe_havens` | `safe_havens` |
| `mission_duration_from_sunlight` | `mission_duration_from_sunlight` |
| `mission_duration_from_sun_elevation` | `mission_duration_from_sun_elevation` |
| `mission_duration_from_sunlight_and_earth_elevation` | `mission_duration_from_sunlight_and_earth_elevation` |
| `mission_duration_from_sun_and_earth_elevation` | `mission_duration_from_sun_and_earth_elevation` |

The focused modules are proposed as `lunarscout.horizon` for horizon formats
and generation, `lunarscout.products` for file-producing downstream products,
and `lunarscout.cuda` for capability status and diagnostics. Backend names do
not appear in scientific function names.

## Proposed signatures

The following signatures show the intended public contract. Shared type aliases
and callback arguments are abbreviated only where repeating them would obscure
the scientific inputs.

```python
def generate_horizons(
    output_directory: str | Path,
    dem_paths: Sequence[str | Path],
    *,
    observer_height_m: float = 0.0,
    compress: bool = True,
    overwrite: bool = False,
    verbose: bool = False,
    progress_callback: ProgressCallback | None = None,
    progress_event_callback: ProgressEventCallback | None = None,
    cancellation_requested: CancellationCheck | None = None,
) -> Path: ...
```

Valid existing horizon tiles are resumable completion units. With
`overwrite=False` they are structurally validated and skipped; with
`overwrite=True` they are regenerated through sibling staging files and
atomically replaced one tile at a time.

The accepted facade is implemented as `lunarscout.horizon.generate_horizons`,
exported at the package root, and used by `Scenario.generate_horizons`. It has
no backend argument because production horizon generation is CUDA-only. A
source-tree and installed-wheel gated test executed the real multi-DEM CUDA
pipeline, published a structurally complete compressed tile, decoded its fixed
`(128, 128, 1440)` contract, and observed finite values for the valid pixel.

```python
def generate_lightmap(
    dem_path: str | Path,
    horizons_path: str | Path,
    output_path: str | Path,
    *,
    times: Iterable[TimeInput] | TimeRange | None = None,
    start: TimeInput | None = None,
    stop: TimeInput | None = None,
    step: timedelta | None = None,
    sun_vectors_m: ArrayLike | None = None,
    backend: Backend = "auto",
    observer_height_m: float = 0.0,
    invalid_value: int = 0,
    overwrite: bool = False,
    start_fresh: bool = False,
    verbose: bool = False,
    progress_callback: ProgressCallback | None = None,
    progress_event_callback: ProgressEventCallback | None = None,
    cancellation_requested: CancellationCheck | None = None,
) -> Path: ...
```

```python
def generate_psr(
    dem_path: str | Path,
    horizons_path: str | Path,
    output_path: str | Path,
    *,
    times: Iterable[TimeInput] | TimeRange | None = None,
    start: TimeInput | None = None,
    stop: TimeInput | None = None,
    step: timedelta | None = None,
    sun_vectors_m: ArrayLike | None = None,
    backend: Backend = "auto",
    observer_height_m: float = 0.0,
    invalid_value: int = 0,
    overwrite: bool = False,
    start_fresh: bool = False,
    verbose: bool = False,
    progress_callback: ProgressCallback | None = None,
    progress_event_callback: ProgressEventCallback | None = None,
    cancellation_requested: CancellationCheck | None = None,
) -> Path: ...
```

Sun- and Earth-elevation functions have the lightmap signature with the
corresponding `sun_vectors_m` or `earth_vectors_m` input and `float` invalid
payload. Values and thresholds are body-center elevation in degrees relative
to the interpolated local terrain horizon.

```python
def generate_safe_havens(
    dem_path: str | Path,
    horizons_path: str | Path,
    output_path: str | Path,
    *,
    times: Iterable[TimeInput] | TimeRange | None = None,
    start: TimeInput | None = None,
    stop: TimeInput | None = None,
    step: timedelta | None = None,
    sun_vectors_m: ArrayLike | None = None,
    earth_vectors_m: ArrayLike | None = None,
    earth_elevation_threshold_deg: float = 2.0,
    sunlight_fraction_threshold: float = 0.2,
    backend: Backend = "auto",
    observer_height_m: float = 0.0,
    invalid_value: float = 0.0,
    overwrite: bool = False,
    start_fresh: bool = False,
    verbose: bool = False,
    progress_callback: ProgressCallback | None = None,
    progress_event_callback: ProgressEventCallback | None = None,
    cancellation_requested: CancellationCheck | None = None,
) -> Path: ...
```

Safe-haven samples must be uniformly spaced. Each output band is one maximal
half-open Earth outage, and values are the longest contiguous low-Sun duration
in hours.

The four mission-duration functions share:

```python
dem_path: str | Path
horizons_path: str | Path
output_path: str | Path
times: Iterable[TimeInput] | TimeRange
evaluation_start: TimeInput
evaluation_stop: TimeInput
candidate_start_intervals: Sequence[CandidateStartInterval | tuple[TimeInput, TimeInput]]
sun_vectors_m: ArrayLike | None = None
earth_vectors_m: ArrayLike | None = None  # only Earth-constrained functions
output_unit: Literal["hours", "days"] = "hours"
backend: Backend = "auto"
observer_height_m: float = 0.0
invalid_value: float = 0.0
overwrite: bool = False
start_fresh: bool = False
verbose: bool = False
progress_callback: ProgressCallback | None = None
progress_event_callback: ProgressEventCallback | None = None
cancellation_requested: CancellationCheck | None = None
```

Their distinct threshold arguments are:

- `mission_duration_from_sunlight`: `sunlight_fraction_threshold`;
- `mission_duration_from_sun_elevation`: `sun_elevation_threshold_deg`;
- `mission_duration_from_sunlight_and_earth_elevation`:
  `sunlight_fraction_threshold` and `earth_elevation_threshold_deg`; and
- `mission_duration_from_sun_and_earth_elevation`:
  `sun_elevation_threshold_deg` and `earth_elevation_threshold_deg`.

Threshold comparisons are inclusive. Evaluation and candidate-start intervals
are half-open. Output duration is accumulated from actual UTC sample intervals,
including a clipped final interval through `evaluation_stop`.

## Scenario conventions

`Scenario` methods omit `dem_path` and default `horizons_path` to the canonical
scenario paths. Product output arguments are scenario-relative and must remain
below the scenario root. Explicit absolute paths remain available through the
root functions.

Scenario conveniences do not register products, mutate application databases,
or add slope, battery, thermal, traverse, or other policy masks.

## Proposed exception taxonomy

Public wrappers translate private implementation failures into these domains:

- `InputError` for invalid ordinary arguments;
- `GridError` for incompatible raster grids;
- `VectorError` and `ProductTimeError` for vector/time contracts;
- `HorizonError`, `HorizonFormatError`, and `HorizonGenerationError`;
- `ComputeBackendError` and `CudaError` for backend capability or execution;
- `ProductError` and `ProductCalculationError` for scientific calculation;
- `ProductStorageError` for staging, journals, TIFF writes, and publication;
  and
- `OperationCancelledError` for cooperative cancellation.

All inherit from `LunarscoutError`, carry stable `code` strings and
machine-readable `details`, and contain no managed-runtime remediation. The
transitional `Native*` names remain only while old public APIs are removed or
deprecated; new product APIs never raise them.

## Still provisional

Implementation review may refine parameter ordering, compression exposure, and
the exact immutable `CudaStatus` and `ProgressEvent` fields. Such refinements
must preserve the accepted backend, verbosity, return, provenance, restart,
and failure semantics above.
