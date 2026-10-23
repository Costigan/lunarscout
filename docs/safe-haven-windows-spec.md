# Safe-haven windows: finding 5 km cells with a good haven and reachable lit terrain

Status: DRAFT SPECIFICATION FOR REVIEW. The work is run in three stages. Stage 1 is
`generate_safe_haven_products`. **Stage 2** scans the stage-1 hibernation product for the
cells that have a haven pixel and writes the list
(`src/lunarscout/_numba_horizon/safe_haven_cells.py`, run by
`scripts/run_safe_haven_stage2.py`). **Stage 3** is the rest of this document: pass 1
(section 5, the lighting ranges of section 8 and the report of section 9, in
`safe_haven_windows.py`, run on its own by `scripts/plan_safe_haven_windows.py`) and pass 2,
the flood (section 6, `safe_haven_flood.py` and `safe_haven_flood_packed.py`), the lit store
(section 8, `safe_haven_litstore.py`) and the block schedule with the table (sections 7 and
8, `safe_haven_blocks.py`), run by `scripts/run_safe_haven_stage3.py`, which reads the stage-2
list. Stage 3 is implemented as a CPU reference route. The GPU lighting and thresholding
kernels, and haven-density rasters, are not implemented. Numbers marked (estimate) come from
reasoning over measurements made on a small synthetic scene; they have not been measured on
the real pole. Section 10 lists the decisions still open.

This document specifies stage 3. Stage 1 is `generate_safe_haven_products` (see
`safe-haven-state-machine.md`). Stage 3 reads the stage-1 rasters, the stage-2 cell list and the
horizons, and produces a table of the 5.12 km cells and months worth studying.

## 1. Question answered

For each cell and calendar month: is there a pixel where the rover can survive
the hibernation (power-negative for at most `survival_limit_hours`), and from
which it can reach many lit pixels both before and after the hibernation?

The ranking uses reachable lit terrain, not haven area. Once a pixel meets the
survival limit, a larger contiguous haven is no better. Regions where the Sun is
high while Earth is low are therefore not rewarded: their Earth-visible windows
are dark, so the reachable-lit counts are zero and the cell-month is dropped.

Reachable-lit counts are a comparative metric between seeds, not an absolute
measure of what a rover could do. They stand in for real operating time. In some
years there is more Sun while Earth is low, so stage 1 alone marks too many
havens in places where the Sun is down whenever communications are available.

Single-pixel DEM height errors can create isolated "haven" pixels. The table
therefore reports several seeds per cell-month, and for each one the fraction of
nearby pixels that are also havens (section 5). A real haven shows a high
fraction; an isolated outlier shows a low one. Combining the columns into one
ranking is deferred.

## 2. Parameters

| Parameter | Default | Meaning |
|---|---|---|
| `survival_limit_hours` | 72 | Haven test (3 days, a multiple of 6 h). |
| `operational_hours` | 48 | Flood duration on each side of the hibernation. |
| `rover_speed_m_s` | 0.01 | Sets the flood step, section 6. |
| `connectivity` | 8 | 4 or 8 neighbour moves per step. |
| `max_seeds_per_cell_month` | 5 | Seeds selected, flooded and reported. |
| `timing_probes_per_seed` | 8 | Nearby pixels probed for a different timing, per accepted seed. |
| `max_consecutive_failures` | 50 | A cell-month is abandoned after this many failed timelines in a row. |
| `stage1_mismatch_limit` | 0.01 | Share of timelines that may disagree with stage 1 before the run stops (and at least 10). |
| `degenerate_outage_window_hours` | stage-1 default | Must equal the stage-1 run's value (24 h by default). |
| `haven_neighbourhood_m` | 100 | Radius for the nearby-haven fraction. |
| `min_nearby_haven_fraction` | 0.05 | Candidates below this are not considered. |
| `boundary_search_hours` | 24 | Longest shift of a flood boundary to find a lit time. |
| `seed_separation_m` | 250 | Spatial suppression radius. |
| `seed_timing_tolerance_hours` | 6 | Timing suppression tolerance. |
| `operating_sunlight_fraction` | 1.0 | Same meaning and fuzz as stage 1. |
| `cell_patches` | 2 | Cell edge in 128-pixel patches (256 px, 5.12 km at 20 m). |
| `block_patches` | 8 | Schedule block edge, section 8. |

`max_slope_deg` and `operating_sunlight_fraction` are read from the stage-1
product metadata. Stage 2 refuses to run if the caller's values differ, so the
slope mask and the lit rule always agree with stage 1.

## 3. Definitions

* **Cell:** a `cell_patches` x `cell_patches` block of horizon patches, aligned
  to the patch grid. Cells tile the DEM; cells with no valid pixel are skipped.
* **Window:** the cell plus a halo of `halo_patches` patches on every side. The
  halo is the smallest whole number of patches that is at least the flood reach
  (section 6). With the defaults it is 1 patch and the window is 4 x 4 patches
  (512 x 512 pixels).
* **Passable pixel:** a pixel that is valid in stage 1 (not NODATA, which means
  slope within the limit and a valid DEM node) and inside the DEM. Pixels outside
  the DEM are impassable.
* **Lit:** `fraction >= operating_sunlight_fraction - 1e-6` at the pixel, the
  stage-1 joint-operating rule. Earth visibility is not tested pixel by pixel during
  a flood; instead a seed is valid only if Earth is available at the seed for the
  whole flood window on each side (section 5, step 5), so every lit pixel is
  counted while communications are available.
* **Earth available:** Earth's elevation above the pixel's own horizon is at least
  `earth_elevation_threshold_deg` (2 degrees), the stage-1 rule. The threshold is
  a practical approximation: radio links are treated as reliable at 2 degrees or
  more above the terrain horizon, because signals reflected off the surface below
  that interfere enough to make the link unusable, and the frame error rate rises
  steeply as elevation falls.
* **Haven pixel for month m:** a pixel whose stage-1 `hibernation` value for
  month m is at most `survival_limit_hours`. That value is the longest merged
  shadow run that shares a sample with the outage, not the length of the outage
  itself (stage-1 definition, `safe-haven-state-machine.md`).
* **Hibernation of record:** for a haven pixel and month, the hibernation that
  starts in the month and has the lowest score; if two tie, the earlier one.
  Its outage start `T_s` is its first outage sample. Its outage end `T_e` is the
  first Earth-available sample after it.

## 4. Inputs

* The stage-1 `hibernation` and `joint_total` rasters.
* The horizon store, the DEM, and a sun-vector source on the flood grid
  (section 6) and on the stage-1 grid. The vector source is a callable taking
  UTC times, so SPICE is touched only for the times that are needed.
* Earth vectors on the stage-1 grid, for each seed's outage times.

## 5. Pass 1: filter and seed selection

Pass 1 is cheap and runs before any lighting. It also reports the counts that
size the rest of the run, section 9.

For each cell and month:

1. **Nearby-haven fraction.** For every haven pixel in the cell, compute the
   nearby-haven fraction (defined below the steps). Discard haven pixels with a
   fraction below `min_nearby_haven_fraction`, so single-pixel outliers never take
   a seed slot.
2. **Filter.** Keep the cell-month only if at least one haven pixel remains.
3. **Proxy.** Rank the remaining haven pixels by `joint_total` for the month,
   summed over a square of side `2 * reach + 1` pixels centred on the pixel (a box
   sum, linear time; `reach` is the flood step count, section 6). Ties break by
   row, then column.
4. **Candidates, in proxy order.** Walk the pixels in that order. A pixel at least
   `seed_separation_m` from every accepted seed is a candidate. A pixel closer than
   that to an accepted seed is probed for a different timing only while its nearest
   accepted seed has had fewer than `timing_probes_per_seed` (8) probes; after that
   it is skipped as a duplicate. This bounds the number of timelines per cell-month
   at about `N * (1 + timing_probes_per_seed)` plus the pixels that turn out to have
   no usable hibernation.
5. **Timeline of each candidate or probe.** Compute its outage and shadow timeline on
   the stage-1 grid, with the stage-1 rules, over the span needed to close the
   hibernation of record and every shadow run touching it. The span starts far
   enough before the month to hold any run that could still score at or under the
   limit, and is extended forward until the hibernation, every run touching it, and
   `operational_hours` of Earth visibility after it are inside it. Confirm that the
   recomputed score equals the stage-1 raster value (as stored, so a byte product
   is compared after its rounding); a mismatch is logged, counted (`invalid_stage1_mismatch`) and the pixel is skipped, because a sample
   within rounding of a threshold can fall either way between the GPU and the CPU. The run stops
   with `PlanConsistencyError` when mismatches number at least 10 and exceed
   `stage1_mismatch_limit` (1 percent) of the timelines evaluated, which means the settings
   differ from the stage-1 run's. This yields
   `T_s`, `T_e` and the score. The outage flags include the degenerate outages of
   `docs/safe-haven-state-machine.md` section 2.1 (the cycle minimum of Earth's
   elevation above a flat horizon, where no real outage lies within the window), so
   an Earth-side pixel's hibernation of record can be one sample long: `T_e` is the
   next sample. **Earth gate.** The seed is valid only if Earth is available at every sample of
   the `operational_hours` immediately before `T_s` and of the `operational_hours`
   starting at `T_e` (contiguous available time of at least `operational_hours` on
   each side, counted in whole samples). A margin dip inside either window, or a
   window cut short by the start or end of the series, fails the gate. There is no
   separate parameter: the gate window is the flood duration. A pixel with no
   hibernation of record, an outage that does not end inside the series, a failed
   Earth gate, or no horizon tile is dropped and counted (`invalid_no_record`,
   `invalid_outage_unclosed`, `invalid_earth_window`, `invalid_no_horizon`).
6. **Selection.** Accept a candidate unless an accepted seed lies closer than
   `seed_separation_m` AND has `T_s` within `seed_timing_tolerance_hours` AND has
   `T_e` within the same tolerance (differences equal to the tolerance count as
   within). Stop at N seeds. Seeds are cell-local, so the choice does not depend on
   how cells are grouped into blocks.
7. **Boundaries.** For each selected seed, find the two flood boundaries of
   section 6 from the seed pixel's own lit state on the flood grid. This needs
   only the seed's horizon column (it is already read for step 5), at most
   `boundary_search_hours` of samples on each side. A seed with no lit time inside
   the cap is invalid.
8. **Lighting ranges.** From the valid seeds' boundaries, set two time ranges for
   the cell-month (section 8): the before range and the after range.

Seeds lie in the cell's own pixels, not in the halo.

**Nearby-haven fraction (no flood).** For a pixel, take the disk of pixels whose
centres lie within `haven_neighbourhood_m` of its centre (81 pixels at 20 m and
100 m, the pixel included). `nearby_pixel_count` is the number of those pixels
inside the DEM. `nearby_haven_count` is the number that are haven pixels for the
month. `nearby_haven_fraction` is their ratio, and step 1 thresholds it. Pixels
that are NODATA in stage 1 (steep or invalid) count in the denominator and are
not havens. The neighbours' own hibernations are not recomputed: the stage-1
value for the month decides.

The counts are reported for each selected seed. Combining them with the score
into one ranking is deferred.

## 6. The flood

**Flood grid.** Times are `epoch + k * dt`. The step `dt` is the largest divisor
of 24 h, in whole minutes, that is at most `pixel_size_m / rover_speed_m_s`. At
20 m and 1 cm/s the speed-implied step is 33.3 min, so `dt` = 32 min (1,440 / 32 is
a whole number; 33 does not divide a day). The step count is
`n = ceil(operational_hours * 60 / dt)` (90 by default). One step moves
the rover one pixel, so the reach is `n` pixels.

**Boundaries.** A flood starts at a time when the seed is lit.

* Nominal before-boundary: `floor_to_grid(T_s)`. Nominal after-boundary:
  `ceil_to_grid(T_e)`.
* If the seed is lit at the nominal boundary, that is the boundary.
* Otherwise move the boundary one flood step at a time away from the hibernation
  (earlier for the before side, later for the after side) until the seed is lit,
  and stop at the first such time. If none is found within
  `boundary_search_hours`, the seed is invalid.

The shift is needed when the shadow run that sets the score began before `T_s`
or ends after `T_e`. "Lit" is the operating rule, so partial sunlight does not
count. Operations start at the first lit sample after the outage, and the last
reliable operations before it are the last lit sample before it. A sunlit gap
inside a merged shadow run counts as lit; the flood does not skip it.
`operating_sunlight_fraction` may be 1.0 or a value such as 0.99, and the same
1e-6 fuzz applies. It must equal the value stage 1 used.

**Before side.** Start at the before-boundary `t0` with `R = {seed}`. Then, for
`j = 1 .. n`:

    R(t0 - j*dt) = neighbours(R(t0 - (j-1)*dt)) AND passable AND lit(t0 - j*dt)

where `neighbours` includes the pixel itself (waiting) and the 4 or 8 adjacent
pixels. Time runs backward, so the frontier is the set of pixels from which the
seed is reachable by the end of the interval.

**After side.** Start at the after-boundary with `R = {seed}` and run the same
rule forward in time.

**Count.** `count_before` and `count_after` are `|R|` at step `n` on each side.
A side whose frontier empties at any step has count 0.

**Seed score.** `count_before + count_after`, valid only if both counts are
greater than 0. The seed is otherwise invalid.

The flood is clipped to the window. Seeds lie in the central cell and the halo is
at least the reach, so the reach never leaves the window. The boundary shift
moves the flood start, not its position, so it does not change this.

**Reference implementation.** `_numba_horizon/safe_haven_flood.py` holds the
plain-Python oracle for this section: `flood_side_reference`, `flood_seed_reference`
and `lit_reference_cube` (the per-pixel lit state from the single-pixel kernel). It is
for small windows and for testing the packed-bit and GPU flood; it is not the
production path. `tests/numba_horizon/test_safe_haven_flood_reference.py` checks it by
hand-worked cases, against an independent path-existence search on random inputs, and
on the synthetic scene.

**Packed-bit flood.** `_numba_horizon/safe_haven_flood_packed.py` is the production
CPU flood: rows of 64-bit words (bit `k` of word `w` is column `64 * w + k`), a
three-row, three-bit dilation per step, an AND with the passable and lit masks, and the
work at step `j` limited to the `j`-pixel box around the seed. It takes `lit[j - 1]`
as the lit mask `j` steps from the boundary, so the caller orders the before side
backward in time. `tests/numba_horizon/test_safe_haven_flood_packed.py` checks it
against the oracle on random inputs, including seeds and walls at word seams.

**Diagnostics.** For each seed, `earth_visible_before_h` and
`earth_visible_after_h`: the contiguous Earth-available time adjacent to the
hibernation, capped at `operational_hours`. Because of the Earth gate every reported seed has
both equal to `operational_hours`; the columns are kept as a check.

## 7. Output

One CSV row per reported seed. A cell-month with at least one valid seed has up to
`max_seeds_per_cell_month` rows, ranked by `score` (ties by smaller (row,
column)). Cell-months with no valid seed are omitted.

| Column | Meaning |
|---|---|
| `cell_row`, `cell_col` | Cell indices on the patch-aligned grid. |
| `x_min_m`, `y_min_m`, `x_max_m`, `y_max_m` | Cell bounds in the DEM projection. |
| `month` | `YYYY-MM`. |
| `rank` | 1 = highest score within the cell-month. |
| `seed_row`, `seed_col`, `seed_x_m`, `seed_y_m` | The seed. |
| `hibernation_hours` | Its stage-1 hibernation score. |
| `outage_start_utc`, `outage_end_utc` | `T_s`, `T_e`. |
| `boundary_before_utc`, `boundary_after_utc` | The flood start times after any shift. |
| `count_before`, `count_after`, `score` | Section 6. |
| `nearby_haven_count`, `nearby_pixel_count`, `nearby_haven_fraction` | Section 5, nearby-haven fraction. |
| `candidate_seeds`, `valid_seeds` | Selected seeds, and how many were valid. |
| `earth_visible_before_h`, `earth_visible_after_h` | Diagnostics. |

There is no weighted combination of columns. Raster views can be built from the
table later.

## 8. Schedule and memory

Per patch, the horizons are the dominant data: 128 x 128 x 1,440 azimuths as
float32 is 94.4 MB on the device. A 24 h chunk of lit bits for one patch is
90 KiB at the default 32 minute step (45 steps), about 1,000 times smaller.

**Lighting ranges.** The boundaries are known before any patch is lit (pass 1,
step 7), so the lighting covers exactly what the floods use, not a worst case.
For a cell-month with valid seeds `i`, with before-boundary `b_i` and
after-boundary `a_i`:

* before range: from `min(b_i) - operational_hours` to `max(b_i)`;
* after range: from `min(a_i)` to `max(a_i) + operational_hours`.

That is `operational_hours` beyond the latest operations start of the set on the after side,
and the same before the earliest last operations on the before side. The outage
between the two ranges is not computed. Ranges are rounded out to the 24 h chunk
grid below.

**Lit chunks and work buffers.** The unit of storage is `(patch, chunk)`: a 24 h
chunk is `1440 / dt` flood steps (45 at the default step). Chunks lie on a fixed global 24 h grid, so neighbouring
windows reuse them. For each patch, the GPU computes the sun fraction over the
ranges with the existing lightmap path, then a small second kernel thresholds
and bit-packs. Only the packed bits are stored. The float array and the
threshold output are work buffers: they are allocated once at the largest size
the ranges need, and reused for every patch. If a range exceeds the buffer, the
patch is processed in time slabs, as the lightmap session does with its time
batch. A flood assembles a `(n + 1, 512, 512)` boolean array from the stored
chunks, over the seed's own time range.

**CPU reference store.** `_numba_horizon/safe_haven_litstore.py` implements this
section on the CPU: `LitStore` (units keyed by patch row, patch column and chunk, packed
as `(steps_per_chunk, 128, 2)` words; 90 KiB at 32 minutes), `fill_lit_store` (reads each
patch's horizons once, lights only the missing units with the CPU lightmap, applies the
stage-1 lit threshold and packs), `LitStore.side_lit` (the masks of one flood side in
flood order) and `flood_seed_from_store` (both sides from the store, with the boundary
agreement rule below). A patch outside the DEM or without a horizon tile is never lit.
It is the reference for the GPU lighting and thresholding kernels, which must produce
the same bits.

**Boundary agreement.** The boundaries come from a CPU evaluation of the seed
pixel and the lit arrays from the GPU. A flood checks that its seed is lit at its
boundary in the assembled array. If a rounding difference at the fuzz limit makes
it unlit, the boundary moves by one step within the computed range; if that is
not possible the seed is invalid and is counted in the pass-1 report as a
boundary mismatch. Tests compare the two evaluations.

**Blocks.** The pole is processed in blocks of `block_patches` x `block_patches`
patches plus a one-patch halo, scanned in a snake so that consecutive blocks
share a strip of horizons.

1. Load the block's horizons into VRAM and keep them for all months.
2. For each month, take the seeds of the block's cells (pass 1). Compute the
   missing lit chunks into a per-month bit store in VRAM, then run the floods.
   Overlapping windows reuse the store. Discard the store at month end.
3. Append the block's rows to the output and checkpoint, so a run resumes.

Budgets for one RTX 4090 (24 GB) and 16 GB of host RAM (estimates):

| Item | Size |
|---|---|
| Block of 8 (10 x 10 patches with halo), float32 horizons | 9.4 GB VRAM |
| Block of 10 (12 x 12 patches with halo) | 13.6 GB VRAM |
| Per-month lit-bit store | about 0.1 GB VRAM |
| Host: reader prefetch, 3 decoded patches | about 0.3 GB |
| Host: stage-1 windows for a block, per product | about 0.16 GB |

The host holds nothing pole-wide. Horizon patches load about 1.55 times each at
block 8, falling to about 1.24 times with the snake carry-over (computed for the
237 x 237 patch grid by `plan_horizon_loads`; the resident block peaks at 100 patches,
9.4 GB, because the patches the next block does not need are dropped before it loads).

**CPU driver.** `_numba_horizon/safe_haven_blocks.py` runs this schedule on the CPU:
`block_order` (the snake), `HorizonResidency` (resident horizons with drop-before-load
and a read count), `plan_horizon_loads` (the load plan, also used to size a run) and
`run_pass_two`. For each block, in snake order, it runs pass 1 for the block's cells,
makes the horizons of their windows resident, then for each month fills the lit store
for the block's cell-months, floods their seeds and appends the rows to a checkpoint
(`<output>.checkpoint.jsonl`, one line per finished block, flushed to disk). A rerun
with the same configuration skips the finished blocks; a different configuration, or a
different `block_patches`, is refused. The CSV is written to a staging file and moved
into place at the end, and an existing output is not replaced unless asked. Rows are
sorted by cell, month and rank, so the table is identical for any `block_patches` and
after a resume. In this CPU route pass 1 still reads horizons itself through its own
small cache, so each patch is read once more than the resident count shows.

## 9. Pass 1 report and the decision gate

Pass 1 returns, without any GPU lighting: cell-months examined and kept, haven
pixels dropped by the nearby-haven threshold,
candidates, seeds, distinct `(patch, chunk)` tasks per block, and the total.
Lighting time is then the task count times a measured time per task.

## 10. Decisions

All settled in review:

* Survival limit 72 h. Connectivity 8, with reach used as a comparative metric.
* Seeds selected, flooded and reported: 5, from `max_seeds_per_cell_month`.
* NODATA pixels count in the nearby-haven denominator, so steep surroundings
  lower the fraction.
* Operations start at the first lit sample after the outage and end at the last
  lit sample before it, with no skipping of sunlit gaps inside merged shadow
  runs. `boundary_search_hours` (24 by default) caps the shift.
* Lighting covers 48 h after the latest operations start of the set of seeds and
  48 h before the earliest last operations, using over-allocated work buffers
  reused across patches.
* Boundary rounding as in section 6.
* The CPU runs the flood on packed bits and moves it to the GPU only if profiling
  shows the copy or the flood dominates.

## 11. Verification plan

* A brute-force set-based flood in plain Python, compared with the packed-bit
  flood on random masks: both connectivities, empty frontiers, window edges.
* Seed timelines reproduce the stage-1 `hibernation` value at every seed.
* The lit chunks match the stage-1 joint-operating criterion at times the two
  grids share, on CPU and CUDA.
* The minimum-fraction filter keeps an isolated haven pixel out of the seed pool
  and leaves a haven cluster in.
* Hand-built tiny scenes with known counts, one per rule in section 6.
* Boundary search: seed lit at the nominal boundary (no shift), shifted earlier
  and later, shifted exactly to the cap, and beyond the cap (seed invalid).
* Boundary agreement: CPU pixel evaluation against the GPU lit arrays at seeds,
  and the mismatch handling.
* The nearby-haven count against a brute-force disk count, including disks that
  cross the DEM edge and NODATA pixels.
* The table is identical for any `block_patches`, and after a resume.
* Seed selection is deterministic: ties break by (row, column).
* Public API tests in a fresh process, as for stage 1.

## 12. Not in scope

Travel distance and route planning, rover power and thermal models, and any
weighted score.

Planned next, after the table: raster views of the results, including a
haven-density raster from the stage-1 products alone (the nearby-haven fraction
over the whole pole), so clusters are visible by eye.
