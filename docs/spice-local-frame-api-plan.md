# SPICE Local-Frame API Plan

Status: implemented.

This plan tracks the first Lunarscout API for SPICE-backed Sun and Earth time
histories at a lunar surface point. The target workflow is notebook-first:
users provide a lunar longitude/latitude, a UTC time range, and a body name,
then receive NumPy arrays, pandas DataFrames, or matplotlib plots.

## Goals

- Provide a small public API for local Sun/Earth vectors and azimuth/elevation
  time histories.
- Use SpiceyPy for ephemeris calculations while keeping kernel loading
  understandable in notebooks.
- Keep `import lunarscout as ls` free of SPICE kernel side effects.
- Use fake or monkeypatched SpiceyPy calls in tests where practical so normal
  unit tests do not require downloaded NAIF kernels.

## Non-Goals

- Do not ship NAIF kernel binaries in the repository.
- Do not add Lunar Analyst application dependencies.
- Do not initialize native moonlib, Python.NET, or CLR as part of this API.

## Local Frame Convention

Use a lunar local NED frame:

- `x`: north, tangent to increasing planetocentric latitude.
- `y`: east, tangent to increasing longitude.
- `z`: down, toward the Moon center and opposite the local surface normal.

Azimuth/elevation convention:

- azimuth `0 deg` is north.
- azimuth `90 deg` is east.
- elevation increases upward.
- elevation `+90 deg` is straight up.

For an NED position vector `(x, y, z)`:

```python
azimuth_deg = degrees(atan2(y, x)) % 360.0
elevation_deg = degrees(atan2(-z, hypot(x, y)))
```

Returned vectors are position/range vectors in SPICE units, currently
kilometers, not unit vectors.

## Kernel Configuration

Default kernel metadata lives in:

```text
data/spice/default_kernels.toml
```

The TOML manifest is configuration, not a SPICE meta-kernel. It contains one
entry per kernel with:

- `id`
- `filename`
- `url`
- `kind`
- `load_order`
- `description`

Load-order convention:

- lower `load_order` values load first;
- broad support kernels load before ephemeris and frame kernels;
- broad date range / lower specificity kernels load before narrower,
  more-specific, or higher-accuracy kernels that should take precedence; and
- order values stay spaced to allow insertion without renumbering.

`LUNARSCOUT_SPICE_META_KERNEL` may point at a user-provided meta-kernel. When
set, default kernel loading should furnish that meta-kernel instead of a
generated local meta-kernel from the manifest.

When `LUNARSCOUT_SPICE_META_KERNEL` is not set, Lunarscout caches the kernel
files named by the manifest under `LUNARSCOUT_SPICE_KERNEL_DIR`, or under
`$XDG_DATA_HOME/lunarscout/spice/kernels` with fallback
`~/.local/share/lunarscout/spice/kernels` when that environment variable is
unset. Missing kernels are downloaded automatically from their manifest URLs.
Lunarscout then generates a temporary SPICE meta-kernel from those local files
and furnishes that generated meta-kernel.

## Public API

### Kernel State

Expose kernel controls through `ls.spice`.

- [x] Add `src/lunarscout/spice.py`.
- [x] Export `spice` from `src/lunarscout/__init__.py`.
- [x] Implement `ls.spice.furnish(path_or_paths, *, disable_autoload=True)`.
- [x] Accept a single `str | Path` or an iterable of `str | Path`.
- [x] Make `furnish(..., disable_autoload=True)` mark default autoload as
      disabled/unnecessary for this process.
- [x] Implement `ls.spice.ensure_default_kernels()`.
- [x] Implement `ls.spice.reload_default_kernels()`.
- [x] Implement `ls.spice.unload_default_kernels()`.
- [x] Implement `ls.spice.clear_kernels()`.
- [x] Implement `ls.spice.default_kernels_loaded()`.
- [x] Implement `ls.spice.autoload_enabled()`.
- [x] Implement `ls.spice.set_autoload_enabled(enabled)`.
- [x] Implement `ls.spice.download_default_kernels(overwrite=False)`.
- [x] Implement `ls.spice.default_kernel_directory()`.
- [x] Implement manifest-driven default meta-kernel generation for local kernel
      files.
- [x] Automatically download missing manifest kernels before generating the
      default meta-kernel.
- [x] Keep default kernel loading lazy; no kernels load at package import time.
- [x] Use `spiceypy.kclear()` in `clear_kernels()` and reset Lunarscout kernel
      bookkeeping.

### Time Iteration

- [x] Add an inclusive UTC datetime iterator:

  ```python
  def iter_times(
      start: datetime | str,
      stop: datetime | str,
      step: timedelta,
  ) -> Iterator[datetime]:
      ...
  ```

- [x] Include the stop datetime when it is exactly aligned to the step.
- [x] Reject non-positive steps.
- [x] Normalize string inputs through the existing UTC parsing conventions where
      possible.

### Surface Point

- [x] Add a public immutable struct:

  ```python
  @dataclass(frozen=True)
  class LonLat:
      longitude: float
      latitude: float
  ```

- [x] Treat longitude/latitude as planetocentric lunar degrees.
- [x] Validate longitude and latitude ranges before SPICE calls.

### Body Histories

Supported body names for the first slice:

```python
"sun"
"earth"
```

Body names map to well-known SPICE targets.

- [x] Implement:

  ```python
  def body_vectors_ned(
      point: LonLat,
      body: Literal["sun", "earth"],
      times: Iterable[datetime],
      *,
      ensure_kernels: bool = True,
  ) -> np.ndarray:
      ...
  ```

- [x] Return shape `(time, 3)` with `float64` columns `x`, `y`, `z`.
- [x] Implement:

  ```python
  def body_vectors_ned_dataframe(...) -> pandas.DataFrame:
      ...
  ```

- [x] DataFrame columns: `time`, `x`, `y`, `z`.
- [x] Implement:

  ```python
  def body_azimuth_elevation(...) -> np.ndarray:
      ...
  ```

- [x] Return shape `(time, 2)` with `float64` columns `azimuth`,
      `elevation`.
- [x] Implement:

  ```python
  def body_azimuth_elevation_dataframe(...) -> pandas.DataFrame:
      ...
  ```

- [x] DataFrame columns: `time`, `azimuth`, `elevation`.
- [x] Add `ensure_kernels` to every SPICE-backed function, defaulting to `True`.
- [x] If `ensure_kernels=True`, call `ls.spice.ensure_default_kernels()`.
- [x] If autoload has been disabled by `ls.spice.furnish(...)`, do not load
      defaults.

### Plotting

- [x] Implement:

  ```python
  def plot_body_elevation(
      point: LonLat,
      body: Literal["sun", "earth"],
      times: Iterable[datetime],
      *,
      grid: bool = True,
      ensure_kernels: bool = True,
  ):
      ...
  ```

- [x] Implement:

  ```python
  def plot_body_elevations(
      point: LonLat,
      bodies: Sequence[Literal["sun", "earth"]],
      times: Iterable[datetime],
      *,
      grid: bool = True,
      ensure_kernels: bool = True,
  ):
      ...
  ```

- [x] Return `(fig, ax)`.
- [x] Label axes and include a legend when plotting multiple bodies.

## Implementation Notes

- Prefer one implementation module for geometry, for example
  `src/lunarscout/spice_geometry.py`, plus `src/lunarscout/spice.py` for kernel
  state.
- Keep pandas and matplotlib imports lazy inside DataFrame and plotting
  functions so the core vector API does not require importing them.
- Use structured Lunarscout exceptions if SPICE setup or body validation fails.
- Keep SPICE imports lazy so environments without SpiceyPy can still import
  Lunarscout and use non-SPICE functionality.
- Avoid loading default kernels from `data/spice/default_kernels.toml` until a
  SPICE-backed function or explicit kernel helper asks for them.

## Verification

- [x] Unit test TOML manifest parsing and load-order sorting.
- [x] Unit test default-kernel download and cache reuse with fake HTTP.
- [x] Unit test generated meta-kernel creation from local manifest files.
- [x] Unit test `furnish()` with a fake SpiceyPy module.
- [x] Unit test autoload enable/disable behavior.
- [x] Unit test `clear_kernels()` resets bookkeeping.
- [x] Unit test `reload_default_kernels()` behavior.
- [x] Unit test `unload_default_kernels()` behavior.
- [x] Unit test `iter_times()` inclusive stop behavior.
- [x] Unit test `iter_times()` rejection of non-positive steps.
- [x] Unit test body-name validation.
- [x] Unit test NED-to-azimuth/elevation math with synthetic vectors.
- [x] Unit test DataFrame column names and row counts with fake vector output.
- [x] Unit test plotting helpers return `(fig, ax)` and respect `grid`.
- [ ] Add at least one optional local smoke script or example for real kernels
      when a real-kernel validation workflow is useful.

## Later Slices

- The first checksum set was bootstrapped by downloading the manifest URLs and
  recording SHA-256 digests from those files. This does not prove the initial
  files were known-good, but it protects later users from accidental drift or
  corruption relative to this checked-in manifest.
- [ ] Add richer descriptions to `data/spice/default_kernels.toml`.
