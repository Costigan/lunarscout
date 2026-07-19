# Public Lighting API Review for `0.1.0rc1`

Status: approved for 0.1.0rc1

This document is the review surface for the file-producing horizon and
lighting API proposed for the first TestPyPI candidate. It records the exact
public names, parameters, defaults, return values, scientific meanings, and
shared operational behavior currently implemented.

Checking an item in the final decision checklist means that its public
contract is approved for `0.1.0rc1`. APIs remain provisional during `0.x`, but
the candidate should not knowingly ship accidental or inconsistent signatures.

## Scope

The review covers these root functions and their `Scenario` conveniences:

| Root function                                        | `Scenario` method              |
| ---------------------------------------------------- | ------------------------------ |
| `ls.generate_horizons()`                             | `scenario.generate_horizons()` |
| `ls.generate_lightmap()`                             | `scenario.lightmap()`          |
| `ls.generate_psr()`                                  | `scenario.psr()`               |
| `ls.generate_sun_elevation()`                        | `scenario.sun_elevation()`     |
| `ls.generate_earth_elevation()`                      | `scenario.earth_elevation()`   |
| `ls.generate_safe_havens()`                          | `scenario.safe_havens()`       |
| `ls.mission_duration_from_sunlight()`                | same name on `Scenario`        |
| `ls.mission_duration_from_sun_elevation()`           | same name on `Scenario`        |
| `ls.mission_duration_from_sunlight_and_earth_elevation()` | same name on `Scenario` |
| `ls.mission_duration_from_sun_and_earth_elevation()` | same name on `Scenario`        |

The established raster, temporal, coordinate, plotting, SPICE, and horizon
reading helpers are outside this focused release review.

## Shared Public Contracts

### Paths and return values

- Root functions accept `str | pathlib.Path` inputs.
- The product functions in this review write GeoTIFF and therefore require
  `.tif` or `.tiff` output paths. This is not a package-wide rule for future
  product families.
- Paths are expanded and resolved before use.
- Every operation returns a `Path` identifying the completed output file or,
  for horizon generation, the completed horizon directory.
- A successful return means final publication completed. Backend selection is
  not part of the return value.
- `Scenario` product methods accept scenario-relative output paths that may not
  escape the scenario root. `scenario.path(relative_path)` returns the absolute
  resolved path for any scenario-relative path; `scenario.output_path()` does
  the same while requiring a non-empty path below the root.

### Backends and diagnostics

- Horizon generation is NVIDIA-CUDA-only and has no `backend` parameter.
- Every downstream product accepts `backend="auto"`, `"cpu"`, or `"cuda"`.
- The default is `backend="auto"`.
- `"cpu"` does not probe or import CUDA.
- `"cuda"` never falls back to CPU.
- `"auto"` selects CUDA when a session can be initialized and otherwise uses
  CPU. A PTX, JIT, kernel, or other execution failure after CUDA work starts is
  reported and is not retried on CPU.
- `verbose=False` is the default. With `True`, the operation writes concise
  backend and progress messages to standard output.
- The selected backend is recorded in `ProgressEvent.backend`, staged restart
  metadata, and the final GeoTIFF `LUNARSCOUT_COMPUTE_BACKENDS` metadata. A
  resumed product preserves the set of backends that contributed patches.

### Times and vectors

Lightmap, PSR, elevation, and safe-haven products require `times=` as an
`ls.TimeRange`. This gives the time domain one public representation and avoids
duplicating `start`, `stop`, and `step` parameters on every product.

Supplying explicit `sun_vectors_m=` or `earth_vectors_m=` requires the same
`times=`. Explicit vectors are converted to a
finite, C-contiguous `float64` array shaped `(time, 3)`, in meters in the
Moon-ME frame. Their first dimension must match the time count. Explicit
vectors do not import SpiceyPy or load SPICE kernels.

Without explicit vectors, Lunarscout generates geometric Moon-ME vectors with
SpiceyPy. Each UTC timestamp is converted independently to ephemeris time.
Callers can generate the exact product-ready arrays themselves with
`ls.body_vectors_moon_me("sun", times)` or
`ls.body_vectors_moon_me("earth", times)`. This helper accepts either an
iterable of timestamps or an `ls.TimeRange` and returns C-contiguous
`float64[time, 3]` meters in Moon-ME.

Mission-duration functions use `evaluation_start=`, `evaluation_stop=`, and a
required `step=datetime.timedelta(...)`. They derive their sampling timestamps
from that evaluation interval and do not accept a separate `times=` argument.

### Output lifecycle

- `overwrite=False` raises
  `ProductStorageError(code="product_output_exists")` for an existing output.
  This preflight happens before DEM loading, SPICE, CUDA selection, or product
  computation, and the existing file is not modified.
- `overwrite=True` permits replacement, but the completed output remains in
  place until its replacement is atomically published.
- Downstream products resume compatible staged work by default.
- `start_fresh=True` discards the exact staged TIFF, manifest, journal, and
  mask sidecar before beginning again. It does not grant permission to replace
  a completed output; that remains the role of `overwrite`.
- Horizon generation resumes by structurally validating and skipping complete
  tiles. It has no `start_fresh` argument; `overwrite=True` regenerates every
  tile.
- Byte products use the dataset mask as the validity signal. An invalid byte
  pixel still needs a physical value in the TIFF, so Lunarscout writes zero by
  default (or the caller's explicit `invalid_value`) consistently. That value
  is only a storage payload: it does not mean nodata, and valid byte pixels may
  have the same value (notably valid non-PSR pixels with value zero).
- Float products expose `nodata=`, default it to `NaN`, write that value into
  invalid pixels, and also write the dataset mask. The mask remains the
  authoritative validity representation for consistent handling across byte
  and float products.
- Every tiled downstream GeoTIFF operation has `compress: bool = True` in its
  signature. Thus compression is enabled when the argument is omitted.
  `compress=False` disables compression but does not disable tiling.

### Progress and cancellation

```python
progress_callback: Callable[[float], None] | None
progress_event_callback: Callable[[ls.ProgressEvent], None] | None
cancellation_requested: Callable[[], bool] | None
```

`progress_callback` receives a monotonic durable fraction. A structured
`ProgressEvent` contains `operation`, `stage`, `completed`, `total`,
`fraction`, selected `backend`, `message`, optional tile coordinates, and the
output `path`. Callback exceptions propagate unchanged.

Cancellation is cooperative and is observed at bounded work boundaries. It
raises `ls.OperationCancelledError`, leaves resumable staged downstream work,
and never publishes an incomplete output.

### Optional output conversion

Every downstream GeoTIFF operation accepts these three keyword arguments:

```python
output_transform: Callable[[np.ndarray], np.ndarray] | None = None
output_dtype: DTypeLike | None = None
output_transform_id: str | None = None
```

The transform is applied patch-by-patch to calculated valid pixels immediately
before they are written. It must preserve the array shape. When a transform is
supplied, `output_dtype` is required and the returned array must have exactly
that dtype. `DTypeLike` is NumPy's usual dtype input, so these are equivalent
ways to request an unsigned 16-bit output:

```python
output_dtype=np.uint16
output_dtype=np.dtype("uint16")
output_dtype="uint16"
```

For example, a mission-duration result could be rounded to whole hours and
stored as unsigned 16-bit integers without changing its documented unit:

```python
def whole_hours(values: np.ndarray) -> np.ndarray:
    return np.rint(values).astype(np.uint16)

ls.mission_duration_from_sunlight(
    ...,
    output_transform=whole_hours,
    output_dtype=np.uint16,
    output_transform_id="round-to-whole-hours-v1",
    nodata=np.iinfo(np.uint16).max,
)
```

`output_transform_id` is optional. Its value, including `None`, is part of the
staged-job identity. If it is omitted on an original run and omitted again on
a restart, the jobs match. If it is supplied, the same value must be supplied
on restart. Lunarscout does not attempt to hash a Python callable; when the ID
is omitted, the caller is responsible for restarting only with compatible
transform behavior. Supplying an ID is therefore a useful guard for automated
or long-lived workflows, not a requirement for ordinary interactive use.

If conversion selects an integer dtype for a native floating product, the
caller must also choose an exactly representable integer `nodata` value; the
default `NaN` cannot be represented by an integer dtype.
The transform must not silently change the product's documented physical unit;
unit-changing conversions require a future metadata contract rather than this
dtype-conversion hook.

### Structured exceptions

All public domain errors derive from `ls.LunarscoutError` and expose stable
`code` and dictionary `details` attributes. The principal product exceptions
are:

- `ls.InputError`, `ls.GridError`, `ls.VectorError`, and
  `ls.ProductTimeError` for invalid inputs;
- `ls.HorizonFormatError` and `ls.HorizonGenerationError` for horizon data or
  generation failures;
- `ls.CudaError` for CUDA availability or execution failures;
- `ls.ProductCalculationError` and `ls.ProductStorageError` for downstream
  calculation or publication failures; and
- `ls.OperationCancelledError` for cooperative cancellation.

## Exact Root Signatures

### Horizon generation

```python
ls.generate_horizons(
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
) -> Path
```

`dem_paths` is ordered: the first DEM defines the output grid and later DEMs
extend surrounding terrain coverage. `observer_height_m` must be finite and in
the half-open range `[0, 100)`. `compress=True` writes `.cbin`; `False` writes
`.bin`. Output tiles are fixed 128 by 128 pixels with 1,440 `float32` azimuth
samples per pixel.

### Lightmap

```python
ls.generate_lightmap(
    dem_path: str | Path,
    horizons_path: str | Path,
    output_path: str | Path,
    *,
    times: TimeRange,
    sun_vectors_m: ArrayLike | None = None,
    backend: Literal["auto", "cpu", "cuda"] = "auto",
    observer_height_m: float = 0.0,
    invalid_value: int = 0,
    output_transform: Callable[[np.ndarray], np.ndarray] | None = None,
    output_dtype: DTypeLike | None = None,
    output_transform_id: str | None = None,
    compress: bool = True,
    overwrite: bool = False,
    start_fresh: bool = False,
    verbose: bool = False,
    progress_callback: ProgressCallback | None = None,
    progress_event_callback: ProgressEventCallback | None = None,
    cancellation_requested: CancellationCheck | None = None,
) -> Path
```

The result is a 128-by-128 tiled `uint8` BigTIFF with one band per UTC sample.
Values are `trunc(255 * visible_solar_fraction)`. The solar disk uses 16 slices
and a 0.27-degree half-angle. Tiles are compressed when `compress=True`, the
default.

### Permanent shadow

```python
ls.generate_psr(
    dem_path: str | Path,
    horizons_path: str | Path,
    output_path: str | Path,
    *,
    times: TimeRange,
    sun_vectors_m: ArrayLike | None = None,
    backend: Literal["auto", "cpu", "cuda"] = "auto",
    observer_height_m: float = 0.0,
    invalid_value: int = 0,
    output_transform: Callable[[np.ndarray], np.ndarray] | None = None,
    output_dtype: DTypeLike | None = None,
    output_transform_id: str | None = None,
    compress: bool = True,
    overwrite: bool = False,
    start_fresh: bool = False,
    verbose: bool = False,
    progress_callback: ProgressCallback | None = None,
    progress_event_callback: ProgressEventCallback | None = None,
    cancellation_requested: CancellationCheck | None = None,
) -> Path
```

The result is a single-band, 128-by-128 tiled `uint8` GeoTIFF. Value 255 means the upper solar
limb never clears the interpolated terrain horizon at the supplied samples;
value 0 means that it clears at least once. Both values are valid data. Tiles
are compressed when `compress=True`, the default.

### Sun elevation

```python
ls.generate_sun_elevation(
    dem_path: str | Path,
    horizons_path: str | Path,
    output_path: str | Path,
    *,
    times: TimeRange,
    sun_vectors_m: ArrayLike | None = None,
    backend: Literal["auto", "cpu", "cuda"] = "auto",
    observer_height_m: float = 0.0,
    nodata: float = np.nan,
    output_transform: Callable[[np.ndarray], np.ndarray] | None = None,
    output_dtype: DTypeLike | None = None,
    output_transform_id: str | None = None,
    compress: bool = True,
    overwrite: bool = False,
    start_fresh: bool = False,
    verbose: bool = False,
    progress_callback: ProgressCallback | None = None,
    progress_event_callback: ProgressEventCallback | None = None,
    cancellation_requested: CancellationCheck | None = None,
) -> Path
```

Each UTC sample becomes one band in a 128-by-128 tiled `float32` BigTIFF in degrees. Values are the
Sun center's elevation relative to the interpolated terrain horizon at its
azimuth, not elevation above a smooth local horizontal plane. Tiles are
compressed when `compress=True`, the default.

### Earth elevation

```python
ls.generate_earth_elevation(
    dem_path: str | Path,
    horizons_path: str | Path,
    output_path: str | Path,
    *,
    times: TimeRange,
    earth_vectors_m: ArrayLike | None = None,
    backend: Literal["auto", "cpu", "cuda"] = "auto",
    observer_height_m: float = 0.0,
    nodata: float = np.nan,
    output_transform: Callable[[np.ndarray], np.ndarray] | None = None,
    output_dtype: DTypeLike | None = None,
    output_transform_id: str | None = None,
    compress: bool = True,
    overwrite: bool = False,
    start_fresh: bool = False,
    verbose: bool = False,
    progress_callback: ProgressCallback | None = None,
    progress_event_callback: ProgressEventCallback | None = None,
    cancellation_requested: CancellationCheck | None = None,
) -> Path
```

Each UTC sample becomes one band in a 128-by-128 tiled `float32` BigTIFF in degrees. Values are the
Earth center's elevation relative to the interpolated terrain horizon at its
azimuth. Tiles are compressed when `compress=True`, the default.

### Safe havens

```python
ls.generate_safe_havens(
    dem_path: str | Path,
    horizons_path: str | Path,
    output_path: str | Path,
    *,
    times: TimeRange,
    sun_vectors_m: ArrayLike | None = None,
    earth_vectors_m: ArrayLike | None = None,
    earth_elevation_threshold_deg: float = 2.0,
    sunlight_fraction_threshold: float = 0.2,
    backend: Literal["auto", "cpu", "cuda"] = "auto",
    observer_height_m: float = 0.0,
    nodata: float = np.nan,
    output_transform: Callable[[np.ndarray], np.ndarray] | None = None,
    output_dtype: DTypeLike | None = None,
    output_transform_id: str | None = None,
    compress: bool = True,
    overwrite: bool = False,
    start_fresh: bool = False,
    verbose: bool = False,
    progress_callback: ProgressCallback | None = None,
    progress_event_callback: ProgressEventCallback | None = None,
    cancellation_requested: CancellationCheck | None = None,
) -> Path
```

An Earth outage is a maximal half-open interval strictly below
`earth_elevation_threshold_deg`. Each tiled `float32` output band represents
one outage and stores the longest complete contiguous duration, in hours, for
which the sunlight fraction is strictly below
`sunlight_fraction_threshold` and whose interval overlaps the Earth outage.
The low-Sun interval may begin before the Earth outage or end after it. Samples
must be uniformly spaced. Tiles are compressed when `compress=True`, the
default.

### Mission duration from sunlight

```python
ls.mission_duration_from_sunlight(
    dem_path: str | Path,
    horizons_path: str | Path,
    output_path: str | Path,
    *,
    evaluation_start: TimeInput,
    evaluation_stop: TimeInput,
    step: timedelta,
    candidate_start_intervals: Iterable[tuple[TimeInput, TimeInput]],
    sunlight_fraction_threshold: float,
    sun_vectors_m: ArrayLike | None = None,
    output_unit: Literal["hours", "days"] = "hours",
    backend: Literal["auto", "cpu", "cuda"] = "auto",
    observer_height_m: float = 0.0,
    nodata: float = np.nan,
    output_transform: Callable[[np.ndarray], np.ndarray] | None = None,
    output_dtype: DTypeLike | None = None,
    output_transform_id: str | None = None,
    compress: bool = True,
    overwrite: bool = False,
    start_fresh: bool = False,
    verbose: bool = False,
    progress_callback: ProgressCallback | None = None,
    progress_event_callback: ProgressEventCallback | None = None,
    cancellation_requested: CancellationCheck | None = None,
) -> Path
```

The condition is sunlight fraction greater than or equal to the required
unitless threshold.

### Mission duration from Sun elevation

```python
ls.mission_duration_from_sun_elevation(
    dem_path: str | Path,
    horizons_path: str | Path,
    output_path: str | Path,
    *,
    evaluation_start: TimeInput,
    evaluation_stop: TimeInput,
    step: timedelta,
    candidate_start_intervals: Iterable[tuple[TimeInput, TimeInput]],
    sun_elevation_threshold_deg: float,
    sun_vectors_m: ArrayLike | None = None,
    output_unit: Literal["hours", "days"] = "hours",
    backend: Literal["auto", "cpu", "cuda"] = "auto",
    observer_height_m: float = 0.0,
    nodata: float = np.nan,
    output_transform: Callable[[np.ndarray], np.ndarray] | None = None,
    output_dtype: DTypeLike | None = None,
    output_transform_id: str | None = None,
    compress: bool = True,
    overwrite: bool = False,
    start_fresh: bool = False,
    verbose: bool = False,
    progress_callback: ProgressCallback | None = None,
    progress_event_callback: ProgressEventCallback | None = None,
    cancellation_requested: CancellationCheck | None = None,
) -> Path
```

The condition is Sun-center terrain-relative elevation greater than or equal
to the threshold in degrees.

### Mission duration from sunlight and Earth elevation

```python
ls.mission_duration_from_sunlight_and_earth_elevation(
    dem_path: str | Path,
    horizons_path: str | Path,
    output_path: str | Path,
    *,
    evaluation_start: TimeInput,
    evaluation_stop: TimeInput,
    step: timedelta,
    candidate_start_intervals: Iterable[tuple[TimeInput, TimeInput]],
    sunlight_fraction_threshold: float,
    earth_elevation_threshold_deg: float,
    sun_vectors_m: ArrayLike | None = None,
    earth_vectors_m: ArrayLike | None = None,
    output_unit: Literal["hours", "days"] = "hours",
    backend: Literal["auto", "cpu", "cuda"] = "auto",
    observer_height_m: float = 0.0,
    nodata: float = np.nan,
    output_transform: Callable[[np.ndarray], np.ndarray] | None = None,
    output_dtype: DTypeLike | None = None,
    output_transform_id: str | None = None,
    compress: bool = True,
    overwrite: bool = False,
    start_fresh: bool = False,
    verbose: bool = False,
    progress_callback: ProgressCallback | None = None,
    progress_event_callback: ProgressEventCallback | None = None,
    cancellation_requested: CancellationCheck | None = None,
) -> Path
```

Both the unitless sunlight fraction and Earth-center terrain-relative
elevation conditions use inclusive lower thresholds.

### Mission duration from Sun and Earth elevation

```python
ls.mission_duration_from_sun_and_earth_elevation(
    dem_path: str | Path,
    horizons_path: str | Path,
    output_path: str | Path,
    *,
    evaluation_start: TimeInput,
    evaluation_stop: TimeInput,
    step: timedelta,
    candidate_start_intervals: Iterable[tuple[TimeInput, TimeInput]],
    sun_elevation_threshold_deg: float,
    earth_elevation_threshold_deg: float,
    sun_vectors_m: ArrayLike | None = None,
    earth_vectors_m: ArrayLike | None = None,
    output_unit: Literal["hours", "days"] = "hours",
    backend: Literal["auto", "cpu", "cuda"] = "auto",
    observer_height_m: float = 0.0,
    nodata: float = np.nan,
    output_transform: Callable[[np.ndarray], np.ndarray] | None = None,
    output_dtype: DTypeLike | None = None,
    output_transform_id: str | None = None,
    compress: bool = True,
    overwrite: bool = False,
    start_fresh: bool = False,
    verbose: bool = False,
    progress_callback: ProgressCallback | None = None,
    progress_event_callback: ProgressEventCallback | None = None,
    cancellation_requested: CancellationCheck | None = None,
) -> Path
```

Both the Sun- and Earth-center terrain-relative elevation conditions use
inclusive lower thresholds in degrees.

### Shared mission-duration interval semantics

For all four mission-duration operations:

- `evaluation_start` and `evaluation_stop` define the overall half-open
  interval.
- Each item in `candidate_start_intervals` is a half-open interval that
  controls where a qualifying run may start.
- A run may continue beyond its candidate-start interval, but never beyond the
  overall evaluation stop.
- The condition sampled at `times[i]` applies over
  `[times[i], times[i + 1])`, clipped to the evaluation stop.
- Each candidate-start interval becomes one `float32` output band.
- `output_unit` accepts only `"hours"` or `"days"` and defaults to hours.

## Current `Scenario` Signatures

`Scenario` supplies the canonical ``dem.tif`` and ``horizons/`` paths and resolves
the requested output below the scenario root.  The downstream methods now mirror
every root keyword parameter explicitly:

```python
scenario.lightmap(
    output: str | Path,
    *,
    times: TimeRange,
    sun_vectors_m: ArrayLike | None = None,
    backend: Literal["auto", "cpu", "cuda"] = "auto",
    observer_height_m: float = 0.0,
    invalid_value: int = 0,
    output_transform: Callable[[np.ndarray], np.ndarray] | None = None,
    output_dtype: DTypeLike | None = None,
    output_transform_id: str | None = None,
    compress: bool = True,
    overwrite: bool = False,
    start_fresh: bool = False,
    verbose: bool = False,
    progress_callback: ProgressCallback | None = None,
    progress_event_callback: ProgressEventCallback | None = None,
    cancellation_requested: CancellationCheck | None = None,
) -> Path
```

The other downstream Scenario methods (``sun_elevation``, ``earth_elevation``,
``safe_havens``, ``psr``, and the four mission-duration methods) have equivalent
explicit signatures matching their root functions, with the canonical DEM and
horizon paths supplied by the scenario.

The PSR-only ``horizons=`` override was removed for consistency.  Callers
needing custom DEM or horizon paths can use the root function directly.

Horizon generation now has the following Scenario-specific form (the private
``_generator`` injection parameter was removed):

```python
scenario.generate_horizons(
    *,
    dem_paths: Sequence[str | Path] | None = None,
    surrounding_dems: Sequence[str | Path] | None = None,
    observer_height_m: float = 0.0,
    compress: bool = True,
    overwrite: bool = False,
    verbose: bool = False,
    progress_callback: ProgressCallback | None = None,
    progress_event_callback: ProgressEventCallback | None = None,
    cancellation_requested: CancellationCheck | None = None,
) -> Path
```

When ``dem_paths`` is omitted, the canonical scenario DEM is placed first and
``surrounding_dems`` are appended. When ``dem_paths`` is supplied, it completely
defines the ordered DEM list and cannot be combined with ``surrounding_dems``.

## Review Findings and Recommendations

These are the places where the current callable surface should change before
it is frozen:

1. **Make Scenario signatures explicit.** (Completed.) The downstream Scenario
   methods now have explicit typed keyword signatures mirroring every root
   keyword parameter.
1. **Remove ``_generator`` from ``Scenario.generate_horizons()``.** (Completed.)
   The private test-injection hook was removed.  Tests now patch
   ``lunarscout.horizon.generate_horizons`` via ``monkeypatch``.
1. **Resolve the PSR-only ``horizons=`` override.** (Completed.) Removed for
   consistency.  Callers needing custom DEM or horizon paths can use the root
   function directly.
1. **Expand Scenario docstrings.** (Completed.) Each Scenario method now
   documents which canonical paths the scenario supplies and directs users to
   the root function's authoritative scientific and operational contract.
The approved float-nodata and patch-level output-conversion contracts above
are now implemented. Focused tests cover NaN TIFF nodata plus masks, conversion
to a requested dtype, exact returned-shape and dtype validation, and compatible
restart when both runs omit `output_transform_id`.

### GeoTIFF metadata fields

The following dataset-level and per-band metadata fields are public
compatibility promises for ``0.1.0rc1``.  Their presence and semantics are
tested and must remain stable:

**Dataset-level tags**

| Tag                           | Content                                                   |
| ----------------------------- | --------------------------------------------------------- |
| ``LUNARSCOUT_TIMESTAMPS_UTC`` | JSON array of ordered ISO-8601 UTC timestamps, one per band |
| ``LUNARSCOUT_COMPUTE_BACKENDS`` | JSON array of backend names (``"cpu"``, ``"cuda"``) accumulated across all durable patches |

**Per-band tags** (on time-series products)

| Tag             | Content                                          |
| --------------- | ------------------------------------------------ |
| ``TIMESTAMP_UTC`` | ISO-8601 UTC timestamp for this band             |

**Per-band tags** (on mission-duration products)

| Tag                 | Content                                      |
| ------------------- | -------------------------------------------- |
| ``DURATION_UNIT``   | ``"hours"`` or ``"days"``                    |
| ``CANDIDATE_START_UTC`` | ISO-8601 UTC start of the candidate interval |
| ``CANDIDATE_STOP_UTC``  | ISO-8601 UTC stop of the candidate interval  |

**Per-band tags** (on safe-haven products)

| Tag             | Content                                          |
| --------------- | ------------------------------------------------ |
| ``TIMESTAMP_UTC`` | ISO-8601 UTC of the first minimum-Earth-elevation sample within the outage |

Products written by Lunarscout are 128-by-128 tiled, band-interleaved,
compressed BigTIFF files with integer predictor 2 for integer dtypes and
floating-point predictor 3 for float dtypes.  Byte products use an
authoritative dataset validity mask and do not declare a nodata value.
Float products declare ``nodata=NaN`` and also write an authoritative
dataset mask.

## Decision Checklist

### Names and organization

- [x] Approve the ten root function names in the scope table.
- [x] Approve their corresponding `Scenario` method names.
- [x] Approve four plainly named mission-duration functions instead of a
  public mode parameter.
- [x] Approve keeping CUDA diagnostics under `ls.cuda` rather than adding
  CPU/GPU suffixes to scientific function names.

### Shared behavior

- [x] Approve returning `Path` without backend information in the return value.
- [x] Approve `backend="auto"` and `verbose=False` as downstream defaults.
- [x] Approve recording selected backend information in progress and file
  metadata, including backend history after restart.
- [x] Approve `TimeRange` for ordinary products; evaluation start/stop plus
  `timedelta` step for mission duration; and the explicit Moon-ME `(time, 3)`
  meter-vector contract.
- [x] Approve early `overwrite=False` rejection, `start_fresh=False`, and
  automatic compatible resume.
- [x] Approve float `nodata=NaN` by default with masks remaining authoritative.
- [x] Approve the output conversion callback, dtype, and restart-identity
  contract.
- [x] Approve the two progress callback forms and callable cancellation check.
- [x] Approve structured exceptions with stable `code` and `details` fields.

### Product-specific behavior

- [x] Approve CUDA-only horizon generation, ordered DEMs, observer height
  default `0.0`, and compressed `.cbin` output by default.
- [x] Approve lightmap `uint8` truncation, the 16-slice solar model, and one
  tiled BigTIFF band per time.
- [x] Approve PSR values 0 and 255, upper-solar-limb semantics, and a separate
  validity mask.
- [x] Approve separate Sun- and Earth-center terrain-relative elevation
  functions with `float32` degree bands.
- [x] Approve safe-haven defaults of 2.0 degrees Earth elevation and 0.2
  sunlight fraction, strict-below comparisons, full low-Sun intervals that
  overlap outage bands, and hour output.
- [x] Approve `compress=True` for every tiled GeoTIFF product, with
  `compress=False` producing tiled but uncompressed output.
- [x] Approve the four mission-duration threshold meanings, inclusive
  comparisons, half-open intervals, and `output_unit="hours"` default.

### Recommended signature cleanup

- [x] Approve explicit typed Scenario signatures rather than `**kwargs: Any`.
- [x] Approve removing the public `_generator` test hook.
- [x] Approve removing the PSR-only `horizons=` override, or specify that all
  downstream Scenario methods should receive a consistent override.
- [x] Approve annotating mission-duration `output_unit` as
  `Literal["hours", "days"]`.
- [x] Approve enhanced docstrings for all root product functions and Scenario
  methods, documenting times/evaluation intervals, vector precedence, backend
  behavior, compress, nodata/mask, output transforms, overwrite/restart,
  return value, progress, cancellation, and scientific thresholds.

After these decisions and any requested edits are implemented and verified,
the corresponding PLAN1 public-signature and docstring gates can be checked.
