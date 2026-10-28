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

Measured on two 36-cell blocks of the real raster (8 workers): 1,740 and 500 seeds, the first block in 10 s. In the polar block 16%
of the seeds have no before boundary (23 of 500 in the other); no seed has cycle -1. **Caveat on the cycle label:**
the midpoints of real outages are spread over the whole cycle, not clustered at the pole's Earth minimum (offsets
from the assigned minimum run from -13.6 to +14.4 days, with a median of 11.7 days before it in the second block), because the outage is set by the terrain horizon and Earth's azimuth as well as its elevation. The
label is a consistent ordering, but where a cycle starts and ends cuts through real outages at an arbitrary
phase, so a rule that links only seeds of consecutive cycle numbers can miss links that a rule based on outage
times would find. The table holds the outage times, so stage 3 or 4 can use either.

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

Measured on 218 of the 1,740 seeds of the first real stage-2 trial block (8 workers, store cold): about 8 s of
flooding, after 40 s to make the mask and load the PSR ids; 39% of the seeds reach the 28-day cap, the median
endurance is 401 h, the median 48-hour reach is 5,115 pixels, 835 edges.

The full real run (524,223 seeds, 16 workers, light store and passable mask warm) took 13.3 minutes: 3,332 seeds
(0.6%) reach the 28-day cap, the median endurance is 124 h, the median 48-hour reach is 22,301 pixels, and
155,390 edges leave 40,346 seeds (`edge_rule="next_cycle"`).

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
  needs the light store, stage 3's passable mask, the PSR ids and the region table and takes about 45 s instead of
  10 s. Size classes are stage 3's: small up to 400 m2 (one pixel at 20 m), medium up to 1,000,000 m2, large above.
  On the real run the chains of 7,334 seeds pass a small PSR, 6,246 a medium one and 11 a large one.

First real run (`next_cycle` edges): 485,776 seeds are roots (a one-seed chain, mission = its endurance, median
126 h) and 38,447 have a predecessor. Chain lengths are 2 (35,311 seeds), 3 (1,005), 4 (774), 5 (589), 6 (440),
7 (255), 8 (50), 9 (19) and 10 (4). The longest mission is 6,008 h (250 days), a chain of 10 seeds in 10
consecutive Earth cycles. The mission percentiles are 24 h (p10), 136 h (median), 308 h (p90) and 672 h (p99).

## 11. Rover chain maps

`scripts/run_safe_haven_chain_maps.py` (module `safe_haven_chain_maps.py`) draws the stage-4 chains: two chain sets,
each as a colour image on the DEM's grid, a copy with dilated lines and a GeoPackage, six files named
`safe_haven_stage_4_<set>{,_dilated}.tif` and `safe_haven_stage_4_<set>.gpkg`. It reads the stage-4 table (with its
`x_m`/`y_m` columns), the light store with stage 3's cached passable mask (for the routes), the PSR id raster and the
DEM's header. On the real table it takes about 90 s (about 35 s of it the routes).

* **`predecessor_edges`:** one segment from each seed that has a predecessor to the seed (38,447 on the real table);
  every edge of the label forest once, branches included. An edge takes the colour of the longest chain that passes
  through it (the deepest seed at or below its target), so a chain does not change colour along its length.
* **`longest_chain_per_root`:** each root's longest chain (most seeds, then the longer mission, then the lowest seed
  id), drawn root to end in the colour of its length (11,884 chains).
* **Images, not data.** The table is the data; the rasters are pictures for people: 4-band 8-bit RGBA GeoTIFFs with a
  transparent background (no data bands, no style files). Chain lengths fall in intervals: 2 seeds, then four
  intervals cut at the quartiles of the lengths of the chains of 3 or more (the same cut points for both sets, from
  the longest chains; repeated cut points merge). The real run gives 2 seeds (11,754 chains), 3 (39), 4-5 (45), 6 (22)
  and 7-10 (24). Each interval is a distinct, saturated, bright hue and brighter than the one before: violet `#6a5cff`,
  rose `#ff4f6e`, orange `#ff7a1a`, yellow `#ffcc00`, green `#7dff3a` (a test checks rising luminance, distinct hues
  and distance from aqua and white). **Every haven is a white pixel** (every seed, isolated ones too) and **route
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
  applied when QGIS adds the layer: rule-based on `chain_length` with the same intervals and colours as the images and
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
  has at least as many pixels as the straight line. Real table: 12,432 predecessor floods and 38,447 routes (5.2 million
  pixels) in about 35 s with 16 workers; a spot check replayed 13 random routes against the real store and all were
  feasible. `RASTER_ROUTES = False` draws straight lines instead (no light store needed). The flood settings and the
  passable mask must be stage 3's.
* **Straight lines in the GeoPackages.** Their segments join seed pixel centres: a route would be thousands of short
  segments, so the vectors stay straight (a schematic that may cross terrain the rover cannot).

Tests check the sets against an exhaustive search over random forests, the intervals (the real run's distribution, the
degenerate cases), the palette, the Bresenham lines, the draw order (the longer chain on top in every input order), the
stamps and clipping, the PSR mark (kept under a longer chain; whole stamp), the paint order of the colours, the pixel
colours of a hand-built graph, the GeoPackage tables, WKB and styles, and `ogrinfo` on both files. The routes are tested
by replaying every route of the scene against the light store (a schedule of at most 4 moves per step into lit,
passable pixels exists), by running the trace kernel on random lit worlds with walls (4 and 8 neighbours, 2 to 4 moves,
compiled and plain Python) and by comparing the images with a plain painter over the routes, with and without PSRs.

## 12. Running the sequence: options and the driver

**Options.** Every constant at the top of a stage script that can vary is also a command-line option,
`--constant-name value` (lower case, hyphens), with the constant as its default; `--help` lists them. The rover's flood
settings (`FloodSettings`) and stage 2's seed rules (`WindowParameters`) are one flag per field. Booleans are
`--flag/--no-flag`; a list such as `--cells 0,1 2,3` takes values; `none` gives a missing value. Each script defines a
procedure with exactly those options as keyword arguments (`run_light_store`, `run_stage1`, `run_stage2`, `run_stage3`,
`run_stage4`, `make_chain_maps`, `run_psr_regions`) and a `default_options()` that reads the constants when it is
called, so another program can load the script and call the procedure; `main(argv=())` reads the command line, sets up
logging and calls it. The helpers are in `lunarscout._script_cli`.

**The driver, `scripts/run_safe_haven.py`.** It runs `psr-regions`, `light-store`, `stage1`, `stage2`, `stage3`,
`stage4` and `chain-maps` in that order by loading the scripts and calling their procedures with one set of shared
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

Tests run the whole sequence on the synthetic scene (all seven steps, then nothing on a second run), touch an input
(only the PSR step and what follows it run) and change a parameter (only stage 3 and what follows run), and cover the
decisions on small fake steps, the shared parameters and the options of every script against its procedure.
