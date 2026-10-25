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

## 5. Planned: stage 1 on the store

Stage 1 would read the planes at 2-hour steps and run the same state machine, so its window parameters
scale with the step (the 4-sample degenerate-outage window and the 24 h merge gap become 12 samples). Its
results change slightly with the finer step, so it would be rerun.

## 6. Planned: stage 3 on the store, 2-hour step, 4 pixels per step

The flood would step every 2 hours with the lighting of a step taken as constant over it, and the rover
moves up to 4 pixels per step (80 m per 7,200 s, 0.0111 m/s, about 11% above the nominal 1 cm/s), each
move into a pixel that is passable and lit in that step. The 48-hour reach is then 24 steps x 4 = 96 pixels
and the per-side footprint ceiling 193 x 193 = 37,249 pixels. Stage 3 would read lit bits from the store
rather than lighting patches, so the window is no longer limited by how many horizons fit in memory.
