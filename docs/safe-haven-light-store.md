# The safe-haven light store (stage 0)

Status: stage 0 is built and tested. Stages 1 and 3 do not read the store yet; sections 5 and 6 are the
plan they follow, not code.

## 1. Why

Stages 1 and 3 each light the pole from the horizons: stage 1 for its monthly products, stage 3 for the
flood. Lighting costs far more than the work that follows it, and both stages decode the same 94 MB horizon
patches again. Stage 0 lights each patch once, thresholds the result and stores it; later stages read bits.

## 2. What is stored

Three yes/no planes per time step and pixel, packed as `(128, 2)` 64-bit words per step and plane (bit `k`
of word `w` in a row is column `64 * w + k`, the layout of `safe_haven_flood_packed`):

| plane | set where | stage-1 meaning |
|---|---|---|
| `lit_partial` | Sun visible fraction >= 0.2 | its complement is power-negative shadow |
| `lit_full` | Sun visible fraction >= 1.0 - 1e-6 | the rover operates |
| `earth_visible` | Earth-centre elevation above the pixel's own horizon >= 2 degrees | its complement is the outage (hibernation) |

A pixel outside the DEM has every bit clear. The time grid is every 2 hours from 2024-01-01 to 2032-01-01
(35,064 steps), on the global grid `k * step` after the Unix epoch (`FloodGrid`).

## 3. Files

One file per patch that has a horizon tile, `<row:03d>/light_<row:03d>_<column:03d>.lsl`, plus
`light_store.json` (settings, time range, patch grid, a hash of the DEM heights) and
`light_store.report.json`. A file is a magic number, a JSON header, an int64 offset table and independently
compressed chunks of `steps_per_chunk` steps (default 96, 8 days); a reader memory-maps the file and
decompresses only the chunks covering the steps it asks for. Files are renamed into place when complete.

## 4. Measurements (eight real patches, full 8 years, CPU)

| | |
|---|---|
| raw size per patch | 215 MB (12.1 TB for 56,169 patches) |
| zstd-3, 8-day chunks | 3.58 MB mean (4.08 MB without one all-dark patch), maximum 7.95 MB: 0.20 TB mean, 0.45 TB worst case |
| zstd-19, 16-day chunks | 2.67 MB mean |
| lz4, snappy, zlib | 9.6, 17.7 and 5.2 MB mean at 8 days without delta |
| one-day chunks | about 12% larger than 8-day chunks; 16 days gains almost nothing |
| exclusive-or of consecutive steps | made zstd files about 10% larger; it helped lz4, snappy and zlib |
| read 30 days | 1 to 2 ms, about 0.11 MB stored |
| read all 8 years | 0.09 s (the horizon tile takes 0.14 s to decode) |
| lighting, CPU | about 5 s per patch for Sun and Earth on all cores |

The sample is eight patches, one of them a permanently dark crater floor. Patches far from the pole where
Earth is never visible have an empty Earth plane. `scripts/measure_light_store.py` repeats the measurement.

## 5. Stage 1 on the store

`ls.generate_safe_haven_products(..., light_store_path=...)` (and `scripts/run_safe_haven_stage1.py`)
reduces each patch from its stored planes: no horizon is read and no Sun position is needed. The per-pixel
state machine is the one stage 1 always ran. It compares the Sun fraction with two thresholds and Earth's
margin with one, so the planes are turned back into a fraction of 0, 0.5 or 1 and a margin of -90 or +90
degrees, which carry exactly what the planes know. The machine's window parameters are given in hours and
become whole samples at the 2-hour step (the degenerate-outage window and the 24 h merge gap are 12 samples
each). With `backend="auto"` or `"cuda"` the planes are expanded on the device by a small kernel
(`safe_haven_cuda._build_expander`) and reduced by the production reducer kernel; `"cpu"` runs the same
machine on all cores.

What the store route requires and checks, as input errors before any work starts: the thresholds given to
stage 1 are the store's; the time samples are the store's step, start on its grid and lie inside its range;
the store was made for this DEM. A patch with no file (it had no horizon tile) is an invalid patch, as
without a store.

One input is not in the planes: the degenerate-outage rule looks for minima of Earth's elevation above a
flat horizon. That series differs by a few thousandths of a degree across a patch, so it is computed once,
at the valid pixel nearest the patch centre, and used for every pixel. A minimum within a step of a tie can
fall a step differently for a pixel away from the centre; on the test scenes the products are identical to
the horizon route's.

Measured on three to five real patches of the full 8 years (35,064 steps): about 0.15 s per patch on the
GPU (device reduction, including decoding the file), against 6 to 10 s on the CPU while the machine was
busy; the haven-pixel counts agree between the two. The whole store is 211 GB (3.7 MB per patch on average).

## 6. Stage 3 on the store, 2-hour step, 4 pixels per step

The flood steps every 2 hours with the lighting of a step taken as constant over it, and the rover makes up
to 4 one-pixel moves per step (80 m per 7,200 s, 0.0111 m/s, about 11% above the nominal 1 cm/s), each move
into a pixel that is passable and lit in that step. The 48-hour reach is 24 steps x 4 = 96 pixels and the
per-side footprint ceiling 193 x 193 = 37,249 pixels. `flood_step_minutes` and `moves_per_step` are
`WindowParameters` (they replace the rover speed).

Stage 3 reads the store: `run_pass_two(light_store=...)` (and `scripts/run_safe_haven_stage3.py`,
`scripts/plan_safe_haven_windows.py`) take pass 1's pixel timelines from `StoreSignals` and the lit units
from the store's operating plane, so no horizon is read and the window is not limited by how many horizons
fit in memory. The unit reads run at about 6,000 a second per thread on the real store (a pixel timeline of
700 samples takes about 1.5 ms), against about 390 units a second for the CPU lightmap the pipeline used
before. The horizon route (`horizon_store=`) stays as the reference: the tests build a store from a scene
and require the same table from both routes, for the block loop, worker processes and the pipeline. The
details are in `docs/safe-haven-windows-spec.md`, "Light store route".

## 7. Lander sites: the longest continuous Sun that starts in each month

A lander that carries no rover stays where it lands, so what matters is how long the Sun can stay up without a
break. `scripts/run_lander_sun_stage1.py` reads only this store and writes, for every pixel and calendar month,
the duration of the longest period of continuous Sun that starts in that month (the Sun plane of the threshold:
1.0 or 0.2, the two the store holds). It is independent of the rover chain.

* A period is consecutive 2-hour steps in Sun; one step out of Sun ends it (no bridging). It belongs to the month
  of its first step and counts in full when it runs into later months. A month in which no period starts is 0.
* A period under way at the first step has an unknown start and is credited to no month. A period under way at
  the last step is credited with the length seen, a lower bound, so the last months read low.
* The product is a 96-band uint16 GeoTIFF (counts of steps, hours = value x 2; 65534 saturates; 65535 is nodata
  where there is no valid terrain). A text file with zeros omitted would not be sparse: 88% to 98% of the
  pixel-months in the patches tried have a period starting in them.
* Measured on real patches: 0.4 to 0.8 s of one core per patch, and 0.0 to 1.1 MB per patch compressed (33 GB
  for the whole raster). A patch near the pole has a run of 156 days.

The reduction walks the packed bits one step at a time with per-pixel state (steps outermost, so the words and
the state are read in order), never unpacking a patch's 35,064 steps at once. The tests compare it with a
brute-force reference that unpacks each patch and finds each pixel's runs with NumPy, on hand-built bits (every
edge above) and on the scene store, and check resume, worker independence and the nodata cases.
