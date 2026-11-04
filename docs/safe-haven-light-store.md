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

## 7. Lander sites: the longest continuous Sun from a landing in each month

A lander that carries no rover stays where it lands, so what matters is how long the Sun can stay up without a
break. `scripts/run_lander_continuous_sun.py` reads only this store and writes, for every pixel and calendar month,
the longest continuous Sun from a landing in that month: land at the best moment of the month and the Sun stays
up for N hours (the Sun plane of the threshold: 1.0 or 0.2, the two the store holds). It is independent of the rover chain.

* A period is consecutive 2-hour steps in Sun; one step out of Sun ends it (no bridging). Landing at a step in
  Sun gives the rest of the period from that step; a month's value is the longest such remainder over its steps.
  A period that starts in the month counts in full; one already under way at the month's first step counts from
  that step, so a long period decays by one month per band (it never reads 0 inside the period). A month with
  no step in Sun is 0.
* A period under way at the first step of the range counts from it. A period under way at the last step is
  credited with the length seen, a lower bound, so the last months read low.
* The product is a 96-band uint16 GeoTIFF (counts of steps, hours = value x 2; 65534 saturates; 65535 is nodata
  where there is no valid terrain). Few pixel-months are 0 (a month with no step in Sun).
* Measured on real patches: 0.4 to 0.8 s of one core per patch, and 0.0 to 1.1 MB per patch compressed (about 33 GB for the whole raster estimated from them). The full run
  wrote 45 GB in 3,602 s with 16 workers. A patch near the pole has a run of 156 days.

The reduction walks the packed bits one step at a time with per-pixel state (steps outermost, so the words and
the state are read in order), never unpacking a patch's 35,064 steps at once. The tests compare it with a
brute-force reference that unpacks each patch and finds each pixel's runs with NumPy, on hand-built bits (every
edge above) and on the scene store, and check resume, worker independence and the nodata cases.

## 8. Rover chain, stage 2: the plan seeds

`scripts/run_safe_haven_stage2.py` (module `safe_haven_plan.py`) replaces the earlier stage 2 (a list of cells) and
the planning half of the earlier stage 3. It picks the seeds that stage 3 floods from and stage 4 chains. It reads
the stage-1 hibernation product and the store (never a horizon) and writes one CSV row per seed, plus a JSON
report of the counts next to it.

* **Which pixels.** The earlier rules, unchanged: per 5.12 km cell and month with a haven pixel (hibernation at
  most 72 h), up to five seeds, best first by the joint operating total within the flood reach, at least 250 m
  apart unless their outage timing differs by more than 6 h, and with at least 5% of the pixels within 100 m also
  havens. Each seed's hibernation is recomputed from the store and must equal the stage-1 value (a disagreement
  is counted, and the run stops when it exceeds 1% of the timelines).
* **No Earth gate.** A seed need not have Earth available for 48 hours on either side
  (`WindowParameters.forward_only`). Seeds that the gate used to drop stay in.
* **Boundaries.** `after_step` is the first step lit at or after the outage ends, within 24 h; a seed with none is
  dropped. It is where stage 3's flood starts. `before_step` is the last lit step at or before the outage starts,
  within 24 h; it is where a rover arriving from an earlier seed must be, and is blank when the seed is not lit
  before its outage (it can then start a chain but not continue one). Steps count the flood step (120 minutes)
  from 1970-01-01T00:00:00Z.
* **Outage kind.** `earth_never_sets`: the one-sample outage the degenerate-outage rule defines at the Earth
  elevation minimum where Earth stays up through its whole cycle. `outage_sun_lit`: a real outage that no shadow
  touches (hibernation 0 h). `outage_in_shadow`: a real outage a shadow touches. A real outage that lasts a single
  sample (a ridge crossing) is `outage_sun_lit` or `outage_in_shadow`, not `earth_never_sets`: with the
  degenerate rule off it is still an outage.
* **Earth cycle.** The minima of Earth's elevation above a flat horizon at the valid pixel nearest the raster's
  centre, numbered from 0 (`earth_cycle`). The run is refused unless they are one per cycle, 20 to 35 days apart
  (the real series has 107 minima 27.0 to 27.3 days apart over 2024 to 2031). A seed takes the cycle of the
  minimum nearest the midpoint of its outage, or -1 when none is within 15 days.
* **Coordinates.** Each row names its pixel by raster `row` and `column` and by `x_m`, `y_m`: the pixel centre in the
  DEM's CRS (projected metres, three decimals; the DEM's transform is the pixel corner, so the centre is half a pixel
  in). `scripts/add_pixel_coordinates.py` adds the two columns to a table made before they existed (right after
  `column`) and overwrites them where they are present; it also works on the stage-4 table. The stage-3 reader
  accepts a seed table with or without them.
* **Order and ids.** Rows are ordered by month, row and column and numbered from 1 in that order, so the table does
  not depend on the number of workers.

Measured on two 36-cell blocks of the real raster (8 workers, an earlier trial before the slope limit moved to 15
degrees): 1,740 and 500 seeds, the first block in 10 s. In the polar block 16%
of the seeds have no before boundary (23 of 500 in the other); no seed has cycle -1. **Caveat on the cycle label:**
the midpoints of real outages are spread over the whole cycle, not clustered at the pole's Earth minimum (offsets
from the assigned minimum run from -13.6 to +14.4 days, with a median of 11.7 days before it in the second block), because the outage is set by the terrain horizon and Earth's azimuth as well as its elevation. The
label is a consistent ordering, but where a cycle starts and ends cuts through real outages at an arbitrary
phase, so a rule that links only seeds of consecutive cycle numbers can miss links that a rule based on outage
times would find. The table holds the outage times, so stage 3 or 4 can use either.

**Full real run (15-degree stage 1, 16 workers).** 483,827 seeds in 14,161 cells and 96 months, in 67.8 minutes
(4,066 s). 6,473 cells (46%) and 121,739 cell-months have seeds, of 1,359,456 cell-months examined, 276,620 of
which held a haven pixel. Selection gave 557,263 seeds; the 73,436 with no lit boundary within 24 h after the
outage are dropped (`invalid_no_lit_boundary`), which leaves 483,827 (the report also counts 148,115 timelines whose
outage never closed, and no seed lost to a missing horizon or record or to an Earth window; the light store matched the
stage-1 value for every timeline, 0 mismatches). By outage kind: `outage_sun_lit` 393,484 (81.3%),
`earth_never_sets` 69,564 (14.4%) and `outage_in_shadow` 20,779 (4.3%). 52,432 seeds (10.8%) have no before
boundary and 155 have no Earth cycle (-1); there are 107 Earth cycles. The 20-degree run had 524,223 seeds; the
15-degree NODATA limit in stage 1 removes the pixels on 15 to 20 degree slopes from the havens, so the plan has fewer.

## 9. Rover chain, stage 3: the forward floods

`scripts/run_safe_haven_stage3.py` (module `safe_haven_floods.py`) replaces the earlier stage 3 (the block and
pipeline route that scored a haven by the lit pixels around it 48 hours before and after its outage). It floods
forward in time from every seed of stage 2 and reads lit bits from the store as it goes.

* **The flood.** The rover starts on the seed pixel at the seed's `after_step`. In each 2-hour step it makes
  4 one-pixel moves (8-neighbour), each into a passable pixel that is lit (operating plane, Sun fraction 1.0) in
  that step; staying on a lit pixel counts. The frontier is every pixel it can be on. The flood ends when the
  frontier is empty or after 28 days. Lighting is read 24 hours of one patch at a time (one stored 8-day chunk is
  decoded and its days kept in a per-worker cache), so a flood that dies in a day reads a day. A flood reaches at
  most 4 x 336 = 1,344 pixels, so its window is 25 x 25 patches.
* **Passable.** Valid height, Horn slope at most 15 degrees, and a patch with a store file. It does not depend on
  the month. It is made once (about 40 s on the real raster) and kept as a 115 MB packed file; it is remade when
  the slope limit or the store changes. A seed on a pixel that is not passable is flagged (`seed_passable` 0); the
  rover still starts there and moves off it.
* **`lit_endurance_hours`.** From the after boundary through the last step with a non-empty frontier, at most
  28 days = 672 h; a value at the cap is a lower bound. **`reach48_pixels`**: the frontier's size after 24 steps.
* **PSR first touch.** Per size class, the hours after the boundary at which a passable PSR pixel of that class is
  next to the frontier (a PSR pixel is never lit, so it is never in the frontier), and that region's id. Small is
  up to 400 m2 (one pixel at 20 m, so a one-pixel region is small), medium up to 1,000,000 m2, large above (parameters).
* **Edges.** A seed A has an edge to a seed B of the next Earth cycle when B has a before boundary within the
  flood's 28 days and B's pixel is in A's frontier at that step (a rover that is on a lit pixel until then can wait
  there). `edge_rule="any_later"` links to every seed that is due later instead (see the caveat on cycle labels in
  section 8). Edges, with the hours from the after boundary, are in a second table.
* **Resume.** The run checkpoints after every cell-month (cell and month of the seeds) and resumes from
  the same inputs and settings; the tables do not depend on the number of workers.

Tests compare the kernel with a plain dilation of unpacked bits (SciPy) on the scene store: the endurance, the
48-hour count, every touch and every edge, for 4 and 8 neighbours, with walls in the passable mask, with targets due
at the last step the frontier lives, and with the lit units read both aligned with the store's chunks and not.

Measured on 218 of the 1,740 seeds of the first real stage-2 trial block (an earlier trial at the 20-degree limit,
8 workers, store cold): about 8 s of
flooding, after 40 s to make the mask and load the PSR ids; 39% of the seeds reach the 28-day cap, the median
endurance is 401 h, the median 48-hour reach is 5,115 pixels, 835 edges.

The full real run at 15 degrees (483,827 seeds, 16 workers, light store and passable mask warm) took 10.5 minutes
(630 s): 2,436 seeds (0.5%) reach the 28-day cap, the median endurance is 118 h, the median 48-hour reach is 17,817
pixels, and 110,510 edges leave 32,786 seeds (`edge_rule="next_cycle"`). The passable mask holds 747,218,080 pixels,
81.2% of the raster, and 6,791 seeds (1.4%) are not on passable terrain. PSR first touches: 387,058 seeds touch a
small region, 361,491 a medium one and 40,238 a large one. The 20-degree run (524,223 seeds) took 13.3 minutes and had
155,390 edges leaving 40,346 seeds.

## 10. Rover chain, stage 4: the chains

`scripts/run_safe_haven_stage4.py` (module `safe_haven_chains.py`) reads the stage-2 seed table and the stage-3
flood and edge tables and writes the final per-seed table. It lights and floods nothing: about 10 s on the real
tables, CPU only.

* **Chains.** An edge A -> B means a rover that wakes at A can be on B's pixel when B's outage begins, hibernates
  through it and wakes at B's `after_step`. Following edges gives a chain of seeds. An edge always leads to a later
  `after_step`, so the labels are made in time order in one pass.
* **One label per seed.** A seed with no incoming edge is its own root. Otherwise it takes the label of the incoming
  edge whose root starts earliest (the root's `after_step`), ties broken by the lowest root id; among edges with
  that same root the longer chain is kept, then the lowest predecessor id. The result does not depend on the order of
  the tables. Tests compare the labels with an exhaustive search over all ancestors of every seed on random graphs
  and on the scene, and check that the order of the edge file does not matter.
* **Coordinates.** `x_m` and `y_m` (the seed pixel's centre in the DEM's CRS) are made from the DEM's transform
  (`DEM_PATH`, header only), whatever the seed table holds, so stage 4 can be run on a stage-2 table made before the
  coordinates were added.
* **Columns added** to the stage-2 and stage-3 columns: `root_id`, `root_start_utc`, `predecessor_id` (0 for a
  root), `chain_length` (seeds, itself included), `mission_hours` (from the root's operations start to the end of
  this seed's lit endurance; a lower bound where the last flood hit the cap), `previous_seeds` and `next_seeds`
  (edge counts). A report beside the table gives the chain-length counts, the mission percentiles and the longest
  chains as lists of seed ids.

* **PSR counts.** `chain_psr_small`, `chain_psr_medium` and `chain_psr_large`: the number of distinct permanently
  shadowed regions (PSRs) of each size class that the chain's routes (root to this seed) pass next to, a PSR pixel being
  among the eight neighbours of a route pixel (`safe_haven_route_psr.py`). A root has no route and counts 0. The routes
  are the ones of section 11 (the predecessor floods run again from the light store, traced back), so stage 4 now
  needs the light store, stage 3's passable mask, the PSR ids and the region table and takes about 37 s (the
  15-degree run) instead of 10 s. Size classes are stage 3's: small up to 400 m2 (one pixel at 20 m), medium up to 1,000,000 m2, large above.
  On the real run at 15 degrees the chains of 5,371 seeds pass a small PSR, 4,765 a medium one and 11 a large one
  (7,334, 6,246 and 11 at 20 degrees).

Real run at 15 degrees (`next_cycle` edges): 451,951 of the 483,827 seeds are roots (no incoming edge; a root with no
descendant is a one-seed chain, and its mission is its endurance, median 120 h) and 31,876 have a predecessor. The seeds
by chain length are 2 (29,196), 3 (857), 4 (663), 5 (509), 6 (394), 7 (194), 8 (45), 9 (16) and 10 (2); the longest chain
has 10 seeds. The longest mission is 6,008 h (250 days), a chain of 10 seeds in 10 consecutive Earth cycles. The mission
percentiles are 22 h (p10), 128 h (median), 298 h (p90) and 572 h (p99). The 20-degree run had 38,447 seeds with a
predecessor and the same longest mission.

## 11. Rover chain maps

`scripts/run_safe_haven_chain_maps.py` (module `safe_haven_chain_maps.py`) draws the stage-4 chains: two chain sets,
each as a colour image on the DEM's grid, a copy with dilated lines and a GeoPackage, six files named
`safe_haven_stage_4_<set>{,_dilated}.tif` and `safe_haven_stage_4_<set>.gpkg`. It reads the stage-4 table (with its
`x_m`/`y_m` columns), the light store with stage 3's cached passable mask (for the routes), the PSR id raster and the
DEM's header. On the real table it takes about 90 s (about 35 s of it the routes).

* **`predecessor_edges`:** one segment from each seed that has a predecessor to the seed (31,876 on the real table at 15 degrees);
  every edge of the label forest once, branches included. An edge takes the colour of the longest chain that passes
  through it (the deepest seed at or below its target), so a chain does not change colour along its length.
* **`longest_chain_per_root`:** each root's longest chain (most seeds, then the longer mission, then the lowest seed
  id), drawn root to end in the colour of its length (10,503 chains at 15 degrees).
* **Images, not data.** The table is the data; the rasters are pictures for people: 4-band 8-bit RGBA GeoTIFFs with a
  transparent background (no data bands, no style files). Chain lengths fall in intervals: 2 seeds, then four
  intervals cut at the quartiles of the lengths of the chains of 3 or more (the same cut points for both sets, from
  the longest chains; repeated cut points merge). Each interval is a distinct, saturated, bright hue and brighter
  than the one before (a test checks rising luminance, distinct hues and distance from aqua and white); the colour key
  below names them. **Every haven is a white pixel** (every seed, isolated ones too) and **route
  pixels that pass next to a PSR** (a PSR pixel among the eight neighbours) are **aqua** `#00ffff`. Paint order, bottom
  to top: routes (longer chains over shorter), aqua, havens. The dilated copy stamps a square of `2 * DILATE_PIXELS + 1`
  pixels (default 5) along each segment, havens one pixel larger (the havens' larger stamps cover much of the aqua in
  it; we may drop the dilated copy). Full-resolution files are about 8 to 10 MB each (deflate; sparse).
* **GeoPackages.** Hand-written GeoPackage 1.3 (no GDAL or shapely dependency), not dilated and always straight:
  layer `predecessor_edges` (a line per edge with seed, predecessor, root, chain length through the edge, the seed's own
  chain length, mission hours and root start), or `chains` (one line through the seed centres of each chain with root,
  end seed, length, mission, start, the seed ids and the numbers of small, medium and large PSRs it passes) plus
  `isolated_seeds` points. The CRS is the DEM's, stored as WKT; `ogrinfo` reads both files.
* **QGIS styles.** Each GeoPackage layer carries a default style in QGIS's `layer_styles` table (`chain_map_styles.py`),
  applied when QGIS adds the layer: rule-based on `chain_length` with the same intervals and colours as the images (the colour key below) and
  a line width that grows with the interval (0.3 to 1.2 mm), drawn shortest first so the long chains are on top; the
  isolated seeds are small white dots. The file lists `isolated_seeds` before `chains`, so QGIS, which stacks each layer
  it adds above the previous one, draws the dots under the lines. The earlier styles were checked by loading the files
  in QGIS 3.34 headless (default style applied, feature ordering) and rendering them.
* **Rover routes in the images.** Stage 3 keeps frontiers, not paths, so the segment of an edge is found by flooding
  the predecessor again exactly as stage 3 did, keeping the frontier and the lit-and-passable mask after every step
  (boxes of `4 * step` pixels around the seed, about 100 MB for a 28-day flood), and tracing back from the successor's
  pixel at its before boundary (`safe_haven_paths.py`). One step of the trace recomputes the four sub-move frontiers in
  a 9 x 9 window around the current pixel from the stored frontier of the step before, and walks back preferring to
  stay, then the first neighbour in the frontier. The route is one feasible, time-respecting walk (at most 4 one-pixel
  moves per step, every pixel entered passable and lit in its step, staying only on lit pixels), not a shortest one; it
  has at least as many pixels as the straight line. Real table at 15 degrees: 10,998 predecessor floods and 31,876 routes
  (3.5 million pixels) in about 35 s with 16 workers; a spot check replayed 13 random routes against the real store and all were
  feasible. `RASTER_ROUTES = False` draws straight lines instead (no light store needed). The flood settings and the
  passable mask must be stage 3's.
* **Straight lines in the GeoPackages.** Their segments join seed pixel centres: a route would be thousands of short
  segments, so the vectors stay straight (a schematic that may cross terrain the rover cannot).

**Colour key.** The same key applies to the images of both sets, the yearly images and the QGIS styles of the
GeoPackages. The five chain colours run from the shortest chains to the longest. The interval limits come from the data
(the quartiles of the chain lengths); the limits and counts below are those of the 15-degree run
(`safe_haven_stage_4_chain_maps.report.json`), and a new run may cut them differently, but the colours stay in this order.

| What is drawn | Colour name | Hex | RGB | Chains (15-degree run) |
|---|---|---|---|---|
| Chain of 2 seeds | dark green | `#297d0f` | 41, 125, 15 | 10,383 |
| Chain of 3 seeds | rose (pinkish red) | `#ff4f6e` | 255, 79, 110 | 30 |
| Chain of 4 to 5 seeds | orange | `#ff7a1a` | 255, 122, 26 | 39 |
| Chain of 6 seeds | pink | `#fb9df8` | 251, 157, 248 | 26 |
| Chain of 7 or more seeds (7 to 10 here) | lime green | `#7dff3a` | 125, 255, 58 | 25 |
| Route pixel next to a permanently shadowed region (PSR) | aqua (cyan) | `#00ffff` | 0, 255, 255 | |
| Haven: every seed, isolated ones too, by shadow period | white (none), yellow (up to 12 h), cyan (12 to 24 h), violet (24 h to the limit) | `#ffffff`, `#f2c94c`, `#38b6c8`, `#7a5cd6` | | |
| Nothing drawn | transparent (alpha 0) | | | |

Tests check the sets against an exhaustive search over random forests, the intervals (the real run's distribution, the
degenerate cases), the palette, the Bresenham lines, the draw order (the longer chain on top in every input order), the
stamps and clipping, the PSR mark (kept under a longer chain; whole stamp), the paint order of the colours, the pixel
colours of a hand-built graph, the GeoPackage tables, WKB and styles, and `ogrinfo` on both files. The routes are tested
by replaying every route of the scene against the light store (a schedule of at most 4 moves per step into lit,
passable pixels exists), by running the trace kernel on random lit worlds with walls (4 and 8 neighbours, 2 to 4 moves,
compiled and plain Python) and by comparing the images with a plain painter over the routes, with and without PSRs.

### Yearly images

`scripts/run_safe_haven_yearly_chain_maps.py` (`run_yearly_chain_maps` in `safe_haven_chain_maps.py`) is the next step
of the sequence (`yearly-chain-maps` in the driver). It writes `safe_haven_stage_4_longest_chain_per_root_<year>.tif`
for each UTC year, non-dilated only: the `longest_chain_per_root` image restricted to one year, so the pattern of
chains and havens year by year can be compared.

* **Which year.** A chain belongs to the UTC year in which its root starts (`root_start_utc`), so every chain is in
  exactly one image and the years together draw the whole image. A chain that carries on into the next year stays whole
  in the year it began (on the real table 330 seeds of chains that began in 2024 are outages of 2025). The havens drawn
  are the seeds of that year's chains (every seed has its chain's root start), isolated ones too.
* **Colours.** The intervals are those of all the chains, so a colour means the same chain length in every year. A year
  without chains is a transparent image. `YEARS` (`--years 2026 2027`) chooses years; none gives every year from the
  first to the last in which a chain starts (the driver passes the years of its window, 2024 to 2031). The colours are
  the ones of the colour key of this section, repeated here for the yearly images:

  | What is drawn | Colour name | Hex |
  |---|---|---|
  | Chain of 2 seeds | dark green | `#297d0f` |
  | Chain of 3 seeds | rose (pinkish red) | `#ff4f6e` |
  | Chain of 4 to 5 seeds | orange | `#ff7a1a` |
  | Chain of 6 seeds | pink | `#fb9df8` |
  | Chain of 7 or more seeds | lime green | `#7dff3a` |
  | Route pixel next to a PSR | aqua (cyan) | `#00ffff` |
  | Haven (a seed of that year's chains), by shadow period | white, yellow, cyan, violet | `#ffffff`, `#f2c94c`, `#38b6c8`, `#7a5cd6` |
  | Nothing drawn | transparent | |

  The chain-length limits are the 15-degree run's; the yearly report
  (`safe_haven_stage_4_yearly_chain_maps.report.json`) lists the intervals of the run that wrote the images.
* **Cost.** The routes are reconstructed again (about 35 s on the real table, the same flood settings and passable mask
  as the chain maps), then each year's image is drawn and written (a few seconds each). Files are written through
  staging files, replacing earlier ones as the new ones are complete.
* **Tests.** The years are disjoint and, put together, equal the whole image pixel for pixel in colour; one year holding
  every chain equals the whole image; an empty year is transparent; chosen years; refusal to replace files and bad years.

## 12. Persistence overview maps

`scripts/run_safe_haven_overview_maps.py` (`run_overview_maps` in `safe_haven_overview.py`; driver step `overview-maps`)
shows where the groups of safe havens are around the pole in each year. It reads only the stage-2 seed table, the DEM
(header and pixels, for the backdrop) and the PSR id raster; it does no lighting, flooding or routing. A real run for
all eight years takes about 25 s.

* **Persistence.** The persistence of a 5.12 km cell in a year is the number of months of that year (0 to 12) in which
  the cell has at least one valid seed (`month`, `cell_row` and `cell_col` of the table). A count of seeds would mostly
  show stage 2's cap of five seeds per cell and month, so the maps count months instead. Every seed is placed by its own
  month, not by the year its chain's root starts (the rule of the yearly chain maps), so the seeds of a year here
  (56,261 in 2026) differ from the havens of the yearly chain images (55,875). A cell that holds haven pixels but no
  valid seed (stage 2 drops seeds with no lit boundary) counts as 0. `CELL_PATCHES` must be stage 2's.
* **Files**, per year in `OUTPUT_DIRECTORY`: `safe_haven_stage_2_persistence_<year>.tif`, a one-band uint8 GeoTIFF on
  the cell grid (119 x 119 cells of 5,120 m, the DEM's CRS and origin; value = months, 0 = none) for QGIS, and
  `safe_haven_stage_2_persistence_<year>.png`, the picture for people (about 2,000 x 2,200 pixels). For all years:
  `safe_haven_stage_2_persistence_contact_sheet.png` (small multiples, 4 columns, the same colour scale in every panel)
  and `safe_haven_stage_2_persistence.report.json` (per year the seeds, the cells with seeds, the number of cells with
  each count of months and their mean).
* **The picture.** A dark hillshade of the DEM (block means of 16 x 16 DEM pixels, light from the north-west), the PSRs
  darker with a muted outline (blocks that are at least 25% PSR), a polar graticule (parallels every 2 degrees from 88 to
  80 S, meridians every 45 degrees; longitude is east-positive and 0 is toward the top of the image), and one square per
  cell with at least one month, inset one pixel so the terrain shows between squares. Cells with no seed are left as
  terrain.
* **Colour key.** One blue hue, brighter for more months, on the dark surface `#1a1a19` (steps 450 to 100 of the reference
  blue ramp, interpolated to 12; tested to be brighter at every step):

  | Months with a seed | Colour name | Hex |
  |---|---|---|
  | 1 | strong blue | `#2a78d6` |
  | 2 | bright blue | `#3482e0` |
  | 3 | azure | `#418ce6` |
  | 4 | cornflower blue | `#5296e7` |
  | 5 | medium sky blue | `#62a0ea` |
  | 6 | sky blue | `#72aaed` |
  | 7 | light sky blue | `#81b3ee` |
  | 8 | pale sky blue | `#91bdf1` |
  | 9 | pale blue | `#a0c6f4` |
  | 10 | powder blue | `#b0cff5` |
  | 11 | very pale blue | `#bfd8f8` |
  | 12 | palest blue | `#cde2fb` |
  | 0 (no seed) | terrain: dark grey hillshade (`#121212` to `#3d3d3d`) | |
  | PSR | near black `#0d0d0d` with a muted grey outline | |
  | Background | charcoal `#1a1a19`; text in off-white `#f0efec` and light grey `#c3c2b7` | |

* **Real run (15 degrees).** Cells with a seed in at least one month, of 14,161: 3,941 (2024), 4,430, 4,513, 3,827,
  4,817, 4,904, 3,795 and 4,482 (2031); the mean months of those cells is 3.2 to 4.0. In 2026, 8 cells have seeds in all 12
  months. The seeds are densest and most persistent in the band toward 0 degrees longitude at the top of the image and
  absent from the lower half toward 180 degrees; the pattern shifts from year to year.
* **Tests.** The persistence from hand-made tables (distinct months, years, an empty year between, bad tables and
  grids), the ramp, the block means, the hillshade and PSR backdrop, the square placement, the graticule against the
  DEM's known corner coordinates, and a run on a small scene (files, PNG decoding, replacing files, refused settings).

## 13. Cell-quality maps

`scripts/run_safe_haven_quality_maps.py` (`run_quality_maps` in `safe_haven_quality_map.py`; driver step `quality-maps`)
is the companion of the persistence maps: where persistence says how many months a cell has havens, quality says how
long a mission can be from them. It reads the final stage-4 table, the DEM and the PSR id raster (about 25 s for all
eight years).

* **Quality.** The quality of a 5.12 km cell in a year is the longest `mission_hours` among the seeds of the cell whose
  `month` is in that year: the longest mission a rover can have completed by a haven in the cell. `mission_hours` is
  the time from the first wake-up of the chain that reaches the seed to the end of the seed's lit endurance, so it counts
  the whole chain, and a mission whose last haven is cut by the 28-day flood cap is a lower bound. Seeds are placed by
  their own month, as in section 12, so the two maps agree on which seeds belong to a year.
* **Files**, in `OUTPUT_DIRECTORY`: per year `safe_haven_stage_4_cell_quality_<year>.tif` (one band, float32, hours on the
  cell grid, nan where the cell has no seed, for QGIS) and `safe_haven_stage_4_cell_quality_<year>.png`; for all years
  `safe_haven_stage_4_cell_quality_contact_sheet.png` and `safe_haven_stage_4_cell_quality.report.json` (per year the
  seeds, the cells with seeds, the number of cells in each class, the median and the longest). The picture has the
  backdrop, graticule and layout of section 12.
* **Colour key.** One orange hue, brighter for longer missions, in eight classes of days. The classes are cut at 2, 4, 7,
  10, 14, 21 and 60 days, because the longest mission per cell and year (34,709 cell-years) has a median of 8.5 days,
  a 90th percentile of 14.8 and a 99th of 45, with a longest of 250 days; the ramp is tints of the reference orange
  `#d95926` chosen by eye to show on the dark hillshade, not a validated palette:

  | Longest mission | Colour name | Hex |
  |---|---|---|
  | under 2 days | rust brown | `#a8461c` |
  | 2 to 4 days | burnt orange | `#c4511f` |
  | 4 to 7 days | vermilion | `#d95926` |
  | 7 to 10 days | terracotta | `#e57444` |
  | 10 to 14 days | coral | `#ee8f66` |
  | 14 to 21 days | light salmon | `#f4aa88` |
  | 21 to 60 days | pale peach | `#f8c5ac` |
  | 60 days or more | pale apricot | `#fbe0d0` |
  | no seed | terrain: dark grey hillshade | |

* **Real run (15 degrees).** The median longest mission of the cells with seeds is 7.7 to 9.2 days in every year; the
  cells reaching 21 days or more are 63 to 91 per year (74 in 2028, whose longest mission is 250 days). The long
  missions are in the sunlit band toward 0 degrees and in a few cells at the pole itself.
* **Tests.** The class edges, the colours, the longest mission per cell and year from hand-made tables, empty years,
  bad tables and grids, and a run on a small scene (GeoTIFF values, PNG colours, a stale sidecar, refused settings).

## 14. Cluster boxes: groups of connected havens

`scripts/run_safe_haven_cluster_maps.py` (`run_cluster_maps` in `safe_haven_cluster_maps.py`; driver step
`cluster-maps`) draws where havens that can reach one another are grouped, and how big each group is. It reads the final
stage-4 table (and the stage-3 edge table with `CONNECT = "edges"`), the DEM and the PSR id raster. A real run takes
about 100 s.

* **Connected havens.** A seed that is joined to no other seed by a chain is dropped; the rest are the connected havens.
  Two connected havens are in one cluster when a chain of any length joins them. `CONNECT` says what a chain is:
  * `"chains"` (the default): the chains of stage 4. A seed has at most one predecessor, so a cluster is one root's whole
    tree, the root and every seed that descends from it. At 15 degrees: 10,503 clusters holding 42,379 connected
    havens (of 483,827 seeds).
  * `"edges"`: every stage-3 edge, kept by stage 4 as a predecessor or not. A seed that several chains can reach joins
    their trees, so there are fewer and larger clusters. At 15 degrees: 9,162 clusters holding 62,258 connected
    havens (19,879 more than in trees); the largest cluster of a year has 145 to 330 havens, and the 330 of 2027 has a
    box of about 80 by 108 km. The extra havens are ones that an edge reaches but stage 4 gave another predecessor.
* **Years.** A cluster belongs to the UTC year of its root's operations start (`root_start_utc`), as in the yearly chain
  maps (section 11). With `"edges"`, an edge between seeds whose roots start in different years is not used (3,046 of
  the 110,510 edges), so a cluster never spans two years.
* **The box.** The bounding box of a cluster is the smallest rectangle of DEM pixels that holds the pixels of its
  havens, on the DEM's grid and axes (north up). Most clusters are small: the median box is 8 by 4 pixels (160 by
  80 m), the 90th percentile about 240 pixels (4.8 km) and the 99th about 650 pixels (13 km).
* **Files**, in `OUTPUT_DIRECTORY`: per year `safe_haven_stage_4_cluster_boxes_<year>.tif`, an RGBA GeoTIFF on the DEM's
  grid with only the outline of each box drawn, `OUTLINE_PIXELS` (default 6, 120 m) wide just outside the box so the
  havens inside stay visible, transparent elsewhere (a tile no box touches is not stored); and
  `safe_haven_stage_4_cluster_boxes_<year>.png`, the picture (backdrop and graticule of section 12 with the boxes on
  top; a box smaller than 4 overview pixels, 1.3 km, is drawn at that size around its centre so it can be seen, and
  the footnote says so). For all years: `safe_haven_stage_4_cluster_boxes_contact_sheet.png`,
  `safe_haven_stage_4_cluster_boxes.csv` (one row per cluster: year, havens, the lowest seed id, pixel and CRS bounds,
  the longest mission; the data behind the pictures) and `safe_haven_stage_4_cluster_boxes.report.json`. At 20 m a
  6-pixel line is thin on a whole-pole view in QGIS: zoom in or raise `OUTLINE_PIXELS`. Boxes are drawn smaller
  clusters first, so a larger one is on top.
* **Colour key.** The colour is the number of connected havens in the cluster (its own havens, not every haven that falls
  inside the box, because boxes overlap). One aqua-green hue, brighter for more, in eight classes cut at 3, 4, 5, 7, 10,
  20 and 50 havens (the median cluster has 3 and the largest 91):

  | Havens in the cluster | Colour name | Hex |
  |---|---|---|
  | 2 | deep sea green | `#127a56` |
  | 3 | jungle green | `#168a62` |
  | 4 | emerald | `#199e70` |
  | 5 to 6 | mint green | `#3cb386` |
  | 7 to 9 | medium aquamarine | `#62c59c` |
  | 10 to 19 | pale aquamarine | `#8ad6b6` |
  | 20 to 49 | pale mint | `#b1e5cf` |
  | 50 or more | very pale mint | `#d7f3e7` |
  | outside the boxes | terrain: dark grey hillshade | |

  The ramp is chosen by eye, not validated.
* **Real run, `"chains"` (15 degrees).** Clusters per year: 944 (2024), 1,104, 1,973, 587, 1,936, 1,796, 802 and 1,361
  (2031), holding 3,754 to 7,765 connected havens; the largest cluster in a year has 44 to 91 havens. The clusters are
  concentrated at the top right of the image (toward 45 degrees east), with scattered small ones around the pole; 2027,
  with the fewest clusters, has the largest share of big ones.
* **Tests.** The clusters of both settings on a hand-made table (trees, an edge stage 4 did not keep, an edge between
  years), bad inputs, the classes and colours, the outline drawing (just outside the box, clipped to a strip and the grid,
  later boxes on top), and a run on a small scene (GeoTIFF colours, transparency inside a box, PNG, the table with its
  CRS bounds, years, refused settings).

## 15a. The monthly haven images (every haven pixel)

`scripts/threshold_safe_havens.py` (procedure `make_haven_month_images`; driver step `haven-months`) turns the stage-1 hibernation product into one GeoTIFF per month in `safe_havens/safe_haven_stage_1_haven_months/` (`safe_haven_stage_1_haven_months_<YYYY-MM>.tif`). A pixel is a haven in a month when its stage-1 count is at most the survival limit divided by 2 hours (24 counts at 48 h, rounded down as `Stage1Rasters.read_haven_window` does); the counts at or below the limit are kept, so the value is the hibernation in two-hour counts, and the rest is NODATA 255. Each file is a tiled, DEFLATE-compressed BigTIFF with overviews that keep the minimum of every 2 x 2 block (a haven is never lost at a coarser zoom), and carries `SAFE_HAVEN_BYTE_THRESHOLD`, `SAFE_HAVEN_SURVIVAL_LIMIT_HOURS`, `SAFE_HAVEN_HOURS_PER_COUNT` and `SAFE_HAVEN_PIXELS` tags. The step depends only on the stage-1 product and the survival limit, so a change of limit reruns it and the seed steps but not the light store or stage 1. A run of 96 months at 48 h took 10.6 minutes on the CPU and made 644 MB; the haven pixel-months sum to 1,001,373,799, the count stage 2 reports as `haven_pixels`. The threshold is derived from `SURVIVAL_LIMIT_HOURS`; the older `safe_havens_thresholded/` folder came from the same script with a raw threshold of 25 (50 h).

## 15b. The seed haven areas (the havens near each chain start)

The rules of every monthly layer, with a worked example, are in `docs/safe-haven-monthly-products.md`; this section describes the seed haven areas product.

`lunarscout.generate_seed_haven_areas(chains_path, dem_path, output_directory, ...)` is a public function (`ls.generate_seed_haven_areas`, `ls.read_seed_areas`, `ls.SeedHavenAreasReport`; module `lunarscout.safe_haven_areas`). `scripts/make_seed_haven_areas.py` (procedure `make_seed_haven_areas`; driver step `seed-haven-areas`, which runs after `haven-months` and before `monthly-chain-maps`, option `AREA_RADIUS_METERS`) wraps it with the script conventions. A chain is drawn each month as the part still ahead: a seed is passed once its own month is over, the chain's **current start** is its first seed not yet passed, and it is drawn while 2 or more seeds are left. For each month the function selects:

* the current start of each chain drawn in the month (the pixels marked as chain starts in the monthly chain pixels): the root in the month its chain starts, the continuation in each later month; isolated seeds start no chain;
* every haven of the month, a seed or not, whose pixel centre is within the radius of one of them (`radius_meters`, default 200 m, which is 10 pixels on the 20 m DEM; the DEM's pixel size sets the pixel count; the straight-line distance is used and a pixel exactly at the limit is included). A haven is a pixel of the month's haven image (section 15a) with a count at most that image's threshold. A start keeps the hibernation of the chain table; and
* the **lone seed** of each chain with only its final seed left in the month, as its pixel alone: no haven around it is added.

Other seeds (isolated ones, passed ones, those still ahead on a chain that are not a current start) are not in an area. A month without a haven image has its chain starts and lone seeds alone. Two products are written, through staging files:

* `safe_haven_stage_4_seed_haven_areas/safe_haven_stage_4_seed_haven_areas_<YYYY-MM>.tif`, one per month from the first chain start to the last seed month, on the DEM's grid for a GIS: uint8 counts of `hours_per_count` (2) hours, 255 NODATA, a palette in the shadow-period classes of section 15 (counts 0, 1-6, 7-12 and 13 or more), DEFLATE tiles and overviews in which each pixel is the minimum of its block, and tags (`SAFE_HAVEN_MONTH`, `SAFE_HAVEN_AREA_RADIUS_METERS`, `SAFE_HAVEN_AREA_STARTS`, `SAFE_HAVEN_AREA_LONE_SEEDS`, `SAFE_HAVEN_PIXELS`, `SAFE_HAVEN_HOURS_PER_COUNT`, `SAFE_HAVEN_HAVEN_IMAGE`);
* `safe_haven_stage_4_seed_haven_areas.npz` beside the chain table: the same pixels as lists (`row`, `column`, `hours`, `offsets` by month), `starts`, `alone` and `with_image` per month and `radius_meters`, which the viewer reads (`ls.read_seed_areas`).

Timing on the 15-degree run: about 4 minutes (most of it writing the 96 full-size images, 150 MB in all); the viewer's lists then build in about 10 s. (The counts of starts and pixels measured for the earlier rule, which counted a chain's root only, 10,461 starts and 163,830 pixels, do not apply to this rule and have not been measured again.)

## 15. The pole viewer

`scripts/view_safe_haven.py` opens a window on the pole: the dark hillshade with the products of one month on it, to pan
and zoom instead of reading the contact sheets. It needs PySide6 (`pip install "lunarscout[viewer]"`) and is not a step
of the driver.

*Controls.* Drag with the left mouse button to pan; turn the wheel to zoom about the pointer (there are no scroll bars);
the slider along the bottom picks the month, from the first to the last month of the dataset (96 months, 2024-01 to
2031-12 in the 15-degree run); the drop-down picks the layer; **Reset view** fits the whole pole.

*Layers.*

| Drop-down | What is drawn | Changes |
|---|---|---|
| Havens and cluster boxes (monthly) | the bounding box of each group of connected havens whose chain starts in the month, in the aqua classes of section 14 (the box is drawn one pixel outside the group; a box smaller than a few screen pixels is drawn solid so that it can be seen), with the month's seed haven areas drawn over them, in the colour of each pixel's shadow period (below) | every month |
| Seed haven areas (monthly) | the same havens without the boxes: the current start of each chain drawn in the month (its first seed not yet passed: the root in the month the chain starts, the continuation in each later month), every haven of the month, a seed or not, within 200 m (10 pixels on the 20 m DEM) of one of them, and the lone last seed of a chain as its pixel alone, as made by the `seed-haven-areas` step (section 15b; without its pixel file the layer shows the chain starts and lone seeds alone). Other seeds (isolated ones, passed ones, later seeds of a chain) are not drawn. The radius is the step's `AREA_RADIUS_METERS` (driver option, `--area-radius-meters` of `scripts/make_seed_haven_areas.py`); the viewer's lists are made again when the step's file changes | every month |
| All havens (monthly) | every pixel that is a haven in the month, not only the seeds: hibernation at most the survival limit, from the monthly haven images of section 15a. Coloured by the pixel's shadow period (its stage-1 hibernation) in four classes of decreasing brightness and different hues: white for none (0 h), yellow `#f2c94c` up to 12 h, cyan `#38b6c8` for 12 to 24 h and violet `#7a5cd6` for 24 h up to the survival limit (48 h). A legend in the lower left names the classes (the last one reads "24 to N h" for a limit of N); the caption gives the pixel count and the limit. The colours were checked with the palette validator: adjacent classes stay at least 17 apart (OKLab x 100) under simulated colour blindness, 20 or more under normal vision, and all are at least 3:1 against the dark surface (white fails the validator's chroma and lightness bands by design). At the coarser zooms a screen pixel shows the best class of its block, so white dominates there. Offered only when the images are beside the chain table | every month |
| Chains (monthly) | the longest chain of each root from the month it starts in until its last seed, each month as the part still ahead: a chain that goes on into later months is shorter in each, in the colour of its remaining length (the yearly image's routes, length colours and aqua PSR passes; the havens of the seed haven areas of the month, the same set as the Seed haven areas layer, in the colours of their shadow periods; and the current start of each chain drawn in the month marked in bright magenta, `#ff00ff`, a colour no length class uses; the entry is in the drop-down only when the monthly pixel file exists); two drop-downs beside the layer choose the chains drawn, together: those whose routes pass next to a PSR of at least a size (any chain, small, medium or large, by the whole chain's `chain_psr_*` counts) and those whose longest hop, the greatest straight-line distance between successive seeds, is at least 50, 100, 200, 500, 1,000 or 2,000 m; a chain passes in every month it is drawn, and the monthly file keeps each chain's pixels apart (a chain table of root, PSR class and hop) so the viewer filters at once | every month |
| Chains (yearly) | the yearly chain image `safe_haven_stage_4_longest_chain_per_root_<year>.tif` of the slider's year | when the slider crosses into another year |

*Permanent shadow, behind every layer.* The pixels of `PSR_PATH` (`psr.tif`, 255 is PSR, on the DEM's grid) are composited onto the hillshade at full resolution in light blue (`#8ecbff`) at 30% opacity, and the background pyramid (RGB, about 3.7 GB, in the cache) is the mean of 2 x 2 blocks of that picture, so a coarser pixel shows the fraction of its block that is shadow (a maximum, tried first, painted nearly every coarse pixel blue because `psr.tif` holds very many small PSRs). The pyramid is made again when `psr.tif` or the DEM changes; the legend names the colour. Without the file the hillshade alone is drawn and a note is printed. The rules of every monthly layer are in `docs/safe-haven-monthly-products.md`.

*Haven colours, in every layer.* Every layer that draws havens colours each by its shadow period (its hibernation), except the start of a chain, which keeps its magenta mark in the monthly chain layer: white for none, yellow `#f2c94c` up to 12 h, cyan `#38b6c8` for 12 to 24 h and violet `#7a5cd6` for 24 h up to the survival limit (48 h; longer if a seed's shadow period is longer, and the legend then reads "24 to N h"). The classes are defined once, in `chain_map_styles.HAVEN_SHADOW_CLASSES`. The colours are baked into the products: the chain images (`..._predecessor_edges*.tif`, `..._longest_chain_per_root*.tif` and the yearly images) mark each haven with its class (the image's `COLOURS` tag lists them), the monthly chain pixel file has one table class per shadow period (labels "haven, shadow 0 h" and so on, ordered so that the brightest is drawn last), and the `isolated_seeds` layer of the GeoPackage has a `hibernation_hours` column and a rule-based style by it. Where havens overlap, the brightest class is on top. The seed layers of the viewer use each seed's `hibernation_hours` from the stage-4 table (kept in `havens.npz`), and the all-havens layer uses the monthly haven images; a legend in the lower left names the classes on every layer that draws havens (and the chain start on the monthly chain layer). Chain images made before this change have white havens: run the `chain-maps`, `yearly-chain-maps` and `monthly-chain-maps` steps again (`--force`; the seeds, floods and chains are not recomputed). At the coarser zoom levels the viewer's pyramid takes the largest value of each channel over a block, so a block holding havens of two classes shows a blend of their colours.

*Seeds are not all havens.* Stage 2 keeps at most five seeds per 5.12 km cell and month, so a cell with thousands of haven pixels holds five. Most seeds (459,349 of 480,592 in the 15-degree run) are isolated: they start no chain, so the layers that draw the seed haven areas (the first two, and Chains (monthly)) do not draw them. The "All havens (monthly)" layer draws every pixel and is the one to use for the extent of the havens.

*One month rule.* A cluster belongs to the month its chain starts in (the `root_start_utc` of its lowest seed; the yearly maps
use the year of the same time), so its box appears in that month only. The chains are drawn each month as the part still ahead
(a seed is passed once its own `month` in the chain table is over, never before the chain's start), and the havens of the
box layer, of the "Seed haven areas" layer and of the monthly chain layer are one set: the current start of each chain drawn
in the month, the havens near it and the lone last seeds (section 15b), so the three layers agree and the magenta marks of the
chain layer are on havens of the other two. (An earlier version put a haven in its own month in the havens layer and from
its chain's start in the chains layer, so about 10 % of the seed-months differed: in 2024-09, 56 seeds only in the chains
layer, of chains that had started and whose own month was later, and 22 only in the havens layer, whose outage ends in the
following month, which is when the chain starts.) The numbers of clusters per month swing
with the lighting cycle (in the 15-degree run, 211 in 2026-03, 2 in 2027-03 and 260 in 2027-12), so a month with few
boxes is a result, not a fault.

*Pregenerated data.* The picture is a bitmap: tiles of 512 pixels from image pyramids, drawn at the level whose pixels
are between half and one screen pixel (full resolution, 20 m, when zoomed in). Nothing is computed while panning.

* `viewer/viewer.json` (beside the products): the months, the DEM grid, and per month the boxes as `[row_min, row_max,
  column_min, column_max, havens]` in DEM pixels, the yearly chain images and how the boxes were clustered
  (`CONNECT`, as in section 14);
* `viewer/havens.npz`: the DEM row and column (and shadow period) of every pixel of the seed haven areas of each month,
  copied from `safe_haven_stage_4_seed_haven_areas.npz` (section 15b), ordered by month, with each month's start
  offset; `viewer.json` also records that file's name, size and time, the radius and the chain starts per month;
* `safe_haven_stage_4_monthly_chain_pixels.npz` (beside the chain table; made by `scripts/run_safe_haven_monthly_chain_maps.py`,
  driver step `monthly-chain-maps`): for each month the pixels the yearly image would draw for the chains that are
  under way (row, column and colour class, with one extra pixel in the start colour on the current start of each chain drawn in the month), found by drawing the same
  routes month by month (225 s on the 15-degree run, the routes included). A chain is drawn from the month its root
  starts in until the month of its last seed, each month as the part still ahead (its seeds from that month on), in the
  colour of that remaining length; a seed's month is its own `month` in the chain table, never before the chain's
  start (the table's months never go back along a chain). The havens drawn are those of the seed haven areas of the
  month, read from `safe_haven_stage_4_seed_haven_areas.npz` (section 15b; the step runs after `seed-haven-areas` and
  refuses to run without that file). In the month a chain starts the routes are the yearly image's (the months of one
  year are not the yearly image, because chains go on and the havens are the areas'). Of the 10,503 chains, 9,221 are shown for
  one month, 1,183 for two and 99 for three to eight; as most of those have one seed left in the later months, which
  is a lone haven, only 265 chain-months are drawn as chains after the start month (10,768 chain-months in all). The
  caption gives the chains drawn and how many of them start in the month. A pixel that two chains share is drawn in
  the colour of the longer;
* the cache (default `/tmp/lunarscout_viewer`, about 11 GB for 8 years): the hillshade pyramid (the overview maps'
  hillshade and grey range at 20 m) and the pyramid of each yearly chain image from half size down (the full size is read
  from the GeoTIFF; reduced by the largest value of each 2 x 2 block, so thin lines survive). A pyramid is built once and
  found by the next run; it is rebuilt only when its source file changes, and it is visible only when complete.

The boxes and dots are not stored as bitmaps: a month has a few thousand of them, so the tiles in view are drawn from
the lists in milliseconds, and 96 months of full-size bitmaps would be hundreds of gigabytes.

`scripts/prepare_safe_haven_viewer.py` makes all of this ahead of time without a display; the viewer does the same, with
a progress window, when it finds something missing. On the 15-degree run: the lists 3 s, the hillshade 30 s, the eight chain
pyramids 65 s; the window then opens in under a second, and a tile takes about 3 ms to draw (34 ms the first time a
full-size chain tile is read).

Limits: the colour key is not drawn in the window (see section 14 for the colours); at full zoom a box outline is one DEM
pixel (20 m) wide, drawn as thick as a pixel; the monthly chains are the longest chain of each root only, as in the yearly image.

## 16. Checking the havens against a ray-cast horizon

`scripts/check_safe_haven_lighting.py` (module `safe_haven_check.py`) re-derives, for a sample of pixels, what stage 0
and stage 2 took from the horizon tiles. It uses no CUDA.

**The reference horizon.** For a pixel it traces all 1,440 azimuths with exact spherical rays (the geometry of
`direct_reference_trace`, vectorised and tested against it sample for sample): the largest slope along the ray, in
degrees, clamped to the tiles' plus and minus 50. The rays run as the horizon generator's do: through the primary DEM from
1 m to where the ray leaves it, through each later DEM from where the ray left the one before (a later DEM adds only the
terrain beyond), each DEM to 1.2 times its smaller side, 1,000 km at most. Samples are a twentieth of a pixel apart out to 5 km and
a quarter pixel beyond (a coarser step missed crests: on rugged terrain a quarter pixel put the horizon 0.5 degrees low).

**What is compared**, for a sample of seeds of the final chain table (spread over outage kind, score, year and chain length
by a least-represented-first rule, repeatable by `RANDOM_SEED`) and of non-havens found beside them, in the window of
`WINDOW_DAYS` either side of the seed's outage: the lighting from the reference horizon, from the production tile and as
stored, all with the patch kernel's own arithmetic. Stored against the tile tests stage 0 alone; tile against reference tests
the horizon as the lighting sees it. A stored bit the reference does not give is *explained* when it lies between the
reference bits for a horizon lowered and raised by `HORIZON_TOLERANCE_DEG` (default 0.3, the worst ray on real terrain in
`docs/numba-horizon-terrain-interpolation.md`); for a pixel that fails, the report gives the smallest of 0.1, 0.25, 0.5, 1 and
2 degrees that explains every bit. Then the stage-2 evaluator runs on the reference lighting (also with the horizon shifted
each way) and on the stored lighting, and the score, outage start and end and kind are compared with the seed's row:
`match`, `within_tolerance` (inside what the shifted horizons give, give or take one step) or `differs`. A non-haven must
not become a haven.

**Result on the real run** (40 seeds and 8 non-havens, 6 minutes, `safe_havens/lighting_check/`; the sample is 22,377
steps of three planes):

* Stage 0 is exact: the stored bits equal the lighting recomputed from the production tile in every step of every plane
  (0 differences).
* The production tile and the reference differ near the Sun and Earth by a median of 0.21 degrees per pixel (largest per
  pixel), at most 2.3 degrees, with no bias (mean -0.03 degrees). The lighting of 41 of 48 pixels is reproduced within 0.3
  degrees; the others need up to 2 degrees.
* Scores: 15 of 40 seeds match exactly, 19 more are within tolerance, 6 differ, and all 8 non-havens stay non-havens. Two of
  the six (outage a few hours long with no shadow in the table, 432 and 530 hours of shadow in the reference) are havens
  made by Earth's margin grazing the 2 degree threshold: the margin wiggles by 0.05 to 0.1 degrees as Earth sets behind a
  bumpy horizon, which makes a four-hour outage in sunlight (score 0) that the reference does not have; the real outage that
  follows is in shadow. The definition takes the lowest score of the month's hibernations, so such a blip makes the pixel a
  haven.
* In the whole chain table 95.7 % of the seeds have score 0 (81.3 % sun-lit outages, 14.4 % one-sample outages where Earth
  never sets) and 33 % have an outage of at most 6 hours, so a sizeable part of the havens rests on short outages whose
  existence depends on the horizon to a tenth of a degree.

**The upper-right edge strips** (`--region upper_right_edges`, 16,585 seeds in the region) do worse: 31 of 48 pixels
reproduced, 25 of 40 seeds keep their score, median horizon difference 0.41 degrees, at most 8.6. Most of it is steep rims
150 to 250 m from the pixel, where the converged reference is 1 to 2 degrees (at single bins up to 4) above the tile.
Running the production CPU oracles on the worst rays settles the cause: `traverse_hierarchy` on the subpatch-interpolated
ray reproduces the tile (14.18 against 14.19 degrees at (125, 20429) bin 1363), and `traverse_level0_fixed_step` on the same
ray gives 15.1, 17.3, 17.3, 18.4, 18.4 and 18.7 degrees at 16, 10, 5, 2, 1 and 0.5 m steps. So the ray fit and the interpolation
are not at fault; the traversal's steps (10 m within 100 m, 16 m beyond, on a 20 m DEM) step over crests. The last column
(observer at the DEM edge) has a few differences this does not explain, and (8174, 30309) matches the oracle at one bin and
not at the next three, probably because the oracle was run on an 8-pixel tile and not the production patch.

It is a sample of 48: it supports these rates only roughly, and says nothing about the 2 degree and 0.2 fraction thresholds
themselves. A larger `--seeds` is the way to tighten them (about 8 seconds per pixel). `ls.generate_horizons` is not
involved, so the check also stands if the horizon generator is changed and the tiles made again.

## 17. Running the sequence: options and the driver

**Options.** Every constant at the top of a stage script that can vary is also a command-line option,
`--constant-name value` (lower case, hyphens), with the constant as its default; `--help` lists them. The rover's flood
settings (`FloodSettings`) and stage 2's seed rules (`WindowParameters`) are one flag per field. Booleans are
`--flag/--no-flag`; a list such as `--cells 0,1 2,3` takes values; `none` gives a missing value. Each script defines a
procedure with exactly those options as keyword arguments (`run_light_store`, `run_stage1`, `run_stage2`, `run_stage3`,
`run_stage4`, `make_chain_maps`, `make_overview_maps`, `make_quality_maps`, `make_cluster_maps`, `make_haven_month_images`, `run_psr_regions`) and a `default_options()` that reads the constants when it is
called, so another program can load the script and call the procedure; `main(argv=())` reads the command line, sets up
logging and calls it. The helpers are in `lunarscout._script_cli`.

**The driver, `scripts/run_safe_haven.py`.** It runs `psr-regions`, `light-store`, `stage1`, `stage2`, `stage3`,
`stage4`, `chain-maps`, `yearly-chain-maps`, `haven-months`, `seed-haven-areas`, `monthly-chain-maps`, `overview-maps`, `quality-maps` and `cluster-maps` in that order by loading the scripts and calling their procedures with one set of shared
parameters, so a threshold is the same everywhere it is used. The values it captures: the paths (inside
`SCENARIO_DIRECTORY` unless given), the time grid, the lighting thresholds, the rover (`MAX_SLOPE_DEG = 15`, moves per
step, connectivity, flood days, PSR size classes, edge rule), the seed rules and the worker counts. The slope limit is
the rover's Horn slope limit in stages 3 and 4 and the chain maps and also the NODATA limit of stage 1 (which was 20
before; the old stage-1 products used it).

*Up to date, like make.* A step is skipped when this driver ran it with the same parameters (recorded in
`<output>/safe_haven_run_state.json`: a fingerprint of the values that change a result, not the numbers of workers or
the cache size), all its outputs exist, and no input is newer than its oldest output (a directory's time is that of
the newest file in it; files a step keeps for itself, like the passable mask, must exist but are not compared). A step
whose input was rebuilt, because an earlier step ran or a file changed, or whose parameters changed runs again, and so
does everything that depends on what it writes. Outputs that exist but that the driver never recorded are not trusted
and not replaced: it stops and tells you to give `--adopt-existing STEP` (record them as made with the current
parameters; nothing is run) or `--force STEP` (run and replace). `--dry-run` reports what each step would do;
`--steps` selects a subset; a failed or interrupted step stops the sequence and is not recorded, so running the driver
again resumes. Code changes are not tracked: `--force` after changing a script.

Tests run the whole sequence on the synthetic scene (all ten steps, then nothing on a second run), touch an input
(only the PSR step and what follows it run) and change a parameter (only stage 3 and what follows run), and cover the
decisions on small fake steps, the shared parameters and the options of every script against its procedure.
