# Safe-haven products: definitions and the per-pixel state machine

This document specifies what `lunarscout.generate_safe_haven_products` computes
and how the code computes it in one forward pass over time. It is written for
review: every rule below is implemented step for step in
`src/lunarscout/_numba_horizon/safe_haven.py` (function `make_machine`), and the
same source runs on the CPU (Numba), on the GPU (a CUDA device function) and as
plain Python.

## 1. What is computed

For each pixel and each UTC calendar month ("band"), three numbers in hours:

| Product | Meaning |
|---|---|
| `hibernation` | The best (shortest) power-negative time among the *hibernations that start in the month*. NaN if none starts. |
| `joint_total` | Hours in the month with Earth available and the Sun fully visible. |
| `joint_longest` | Longest uninterrupted stretch of such hours, credited to the month the stretch starts in. |

All three are NaN where the slope exceeds `max_slope_deg` or the DEM node is
invalid. The products are screening aids for choosing places to study in more
detail. They model neither a particular rover nor its route.

## 2. Definitions

Time is sampled at a fixed step. For each pixel, at each sample `t`, two signals
come from the pixel's own horizon: the visible fraction `f` of the solar disk
and the elevation `e` of Earth's centre above the local terrain horizon.

* **Outage sample:** `e < earth_elevation_threshold_deg` (default 2 degrees).
  No reliable line of sight. Equality counts as available.
* **Shadow sample:** `f < sunlight_fraction_threshold` (default 0.2). Equality
  is not shadow.
* **Hibernation:** a maximal run of consecutive outage samples. The rover is in
  its low-power mode for the whole of it, and is power-negative whenever it is
  also in shadow.
* **Shadow run (merged):** shadow samples joined into one run when the sunlit
  gap between two of them lasts *strictly less than* `merge_gap_hours` (default
  24). The gap's samples belong to the run. The run's *span* is from its first
  to its last shadow sample, and its *length* is the number of samples in the
  span. Gaps of zero samples (consecutive shadow samples) always join. A gap that
  reaches `merge_gap_hours` ends the run at the last shadow sample before it.
* **Intersect:** a run intersects a hibernation when some outage sample of the
  hibernation lies inside the run's span (including a bridged gap).
* **Score of a hibernation:** the maximum length of the runs that intersect it,
  or 0 if none does. Whole runs count, including the parts before and after the
  outage and in other months.
* **Month of a hibernation:** the band containing its first sample. Nothing else
  decides it. A hibernation is therefore counted exactly once.
* **Value of a band:** if no hibernation starts in it, NaN. Otherwise the
  *minimum* score among the hibernations that start in it ("the best
  opportunity that month"). A month can contain two starts; the machine does not
  assume it cannot contain three.
* **Joint operating sample:** Earth available (not an outage sample) and
  `f >= operating_sunlight_fraction - 1e-6` (default fraction 1.0). The 1e-6
  tolerance lets a fully lit disk count even if float32 rounding puts its
  fraction a step below 1.

Two boundary conventions. A hibernation already under way at the first sample
has an unknown start; it is not credited to any month, but it still intersects
runs normally. Runs touching the first or last sample are limited to the
samples available.

## 3. Why one pass is hard

A pixel's timeline is two interleaved sequences, hibernations and shadow runs.
The score of a hibernation is not known when it ends, because a run that
intersects it can continue for days afterwards. A run can intersect several
hibernations, and a hibernation can intersect several runs. The month value is a
minimum over hibernations, so a hibernation cannot be folded into the band
until its final score is known (updating a minimum with a number that later
grows is wrong). The machine therefore keeps, for the one open run, a compact
summary of the hibernations that are waiting for it to finish, and settles them
all at once when it closes. A permanently shadowed pixel is the extreme case:
one run touches every hibernation, and the summary stays the same size.

## 4. State

Per pixel, 17 integers (`S[row, pixel]` in the code). `UNSET` is 2147483647.

| Row | Meaning | Initial |
|---|---|---|
| `RUN_LEN` | Samples in the open run through its last shadow sample. 0 means no open run. | 0 |
| `GAP` | Sunlit samples since the open run's last shadow sample. 0 while in shadow. | 0 |
| `IN_HIB` | 1 while a hibernation is active. | 0 |
| `HIB_BAND` | Band of the active hibernation's first sample; -1 if uncredited. | -1 |
| `HIB_SCORE` | Longest *closed* run known to intersect the active hibernation. | 0 |
| `HEAD` | 1 if the open run has a **head**: the hibernation that was active when the run began. | 0 |
| `HEAD_BAND` | Band of the head; -1 if uncredited. | -1 |
| `HEAD_PRIOR` | The head's score from runs that closed *before* this run began. | 0 |
| `HEAD_ACTIVE` | 1 while the head is still the active hibernation. | 0 |
| `MID_LO`, `MID_HI` | Lowest and highest band of ended hibernations, other than the head, that intersect the open run. `UNSET` and -1 when none. | |
| `TAIL_ACTIVE` | 1 while the active hibernation is a member of the open run's chain but not its head. | 0 |
| `PEND_LO`, `PEND_HI` | Lowest and highest band of hibernations that began and ended wholly inside the *pending gap* (not yet known to be bridged). | `UNSET`, -1 |
| `PEND_CUR` | 1 while the active hibernation began inside the pending gap. | 0 |
| `JOINT_RUN` | Samples in the current joint-operating run. | 0 |
| `JOINT_BAND` | Band of that run's first sample. | -1 |

Per band and pixel: `best` (int32, initially `UNSET`; the minimum score so far),
`have` (1 once a hibernation has started in the band), `joint_total` and
`joint_best` (counters).

The **chain** of the open run is the set of hibernations known to intersect it:
at most one head, the ended members summarised by the band range
`MID_LO..MID_HI`, and at most one active member (`TAIL_ACTIVE`, or the head if
`HEAD_ACTIVE`). Three facts make this summary sufficient (section 10):

1. only the head can have intersected an earlier run, so only the head carries a
   prior score (`HEAD_PRIOR`);
2. every other ended member scores exactly the run's final length;
3. an active member can still gain from later runs, so it is not settled when the
   run closes.

## 5. Rules for one sample

`outage`, `shadow` and the band `m` of the sample come from the inputs.
`gap_limit` is the smallest gap, in samples, that is not bridged:
`ceil(merge_gap_hours / step)`, so with a 6-hour step and 24 hours a gap of 3
samples (18 h) is bridged and a gap of 4 (24 h) is not. Steps run in this order.

### Step 1: hibernation starts or ends

* **Outage sample and `IN_HIB = 0`: a hibernation starts.** Set `IN_HIB = 1`,
  `HIB_SCORE = 0`. `HIB_BAND = m`, unless this is the first sample or `m` is
  outside every band, in which case `HIB_BAND = -1`. If `HIB_BAND >= 0`, set
  `have[HIB_BAND] = 1`.
* **Not an outage sample and `IN_HIB = 1`: the hibernation ended** at the
  previous sample. Set `IN_HIB = 0`, then, in this order:
  * If `HEAD_ACTIVE`: clear it. The head stays the head, now ended.
  * Else if `TAIL_ACTIVE`: clear it and add `HIB_BAND` to `MID_LO..MID_HI`.
  * Else if `PEND_CUR`: clear it and add `HIB_BAND` to `PEND_LO..PEND_HI`.
  * Else it belongs to no open run: its score is final. Commit `HIB_SCORE` to
    band `HIB_BAND` with `best = min(best, score)`.

  ("Add a band to a range" ignores -1.)

### Step 2: shadow runs

* **Shadow sample, no open run (`RUN_LEN = 0`):** open a run, `RUN_LEN = 1`. If a
  hibernation is active it is the head: `HEAD = 1`, `HEAD_BAND = HIB_BAND`,
  `HEAD_PRIOR = HIB_SCORE`, `HEAD_ACTIVE = 1`.
* **Shadow sample, run open:**
  * If `GAP > 0` the gap is bridged: `RUN_LEN += GAP + 1`, `GAP = 0`. The
    hibernations wholly inside the gap are now confirmed members: merge
    `PEND_LO..PEND_HI` into `MID_LO..MID_HI` and clear the pending range. Clear
    `PEND_CUR`; a hibernation that began in the gap and is still active is
    picked up by the next rule.
  * Otherwise `RUN_LEN += 1`.
  * If a hibernation is active but is neither the head nor already the tail, it
    began at this shadow sample or in the bridged gap: set `TAIL_ACTIVE = 1`.
* **Sunlit sample, run open:** `GAP += 1`. If a hibernation is active that is not
  a member and not already pending, set `PEND_CUR = 1`: it began inside a gap
  whose fate is not yet known. If `GAP >= gap_limit`, close the run (section 6).
* **Sunlit sample, no open run:** nothing.

### Step 3: joint operating

If the sample is not an outage and `f >= operating_sunlight_fraction - 1e-6`:
if `JOINT_RUN = 0`, set `JOINT_BAND = m`. Add one to `JOINT_RUN`; add one to
`joint_total[m]` (if `m >= 0`); set `joint_best[JOINT_BAND] =
max(joint_best[JOINT_BAND], JOINT_RUN)` (if `JOINT_BAND >= 0`). Otherwise
`JOINT_RUN = 0`.

## 6. Closing a run

The run is over, with final length `L = RUN_LEN` (the pending gap is not part of
it). Settle every member of the chain:

* **Head:** its score is `max(HEAD_PRIOR, L)`. If the head is still active it
  keeps going as the active hibernation with `HIB_SCORE` set to that score.
  Otherwise commit it to `HEAD_BAND`.
* **Ended members:** commit `L` to every band in `MID_LO..MID_HI`.
* **Active non-head member** (`TAIL_ACTIVE`): `HIB_SCORE = L`. It continues, and
  a later run may raise its score.
* **Hibernations wholly inside the unbridged gap** (`PEND_LO..PEND_HI`): they
  touched no run. Commit 0.
* Clear all chain and pending state. `PEND_CUR` is cleared too: a hibernation
  that began in the gap and is still active is now an ordinary active
  hibernation with score 0, and becomes a head if a later run opens during it.

Committing means `best[band] = min(best[band], score)` for each band in the
range. A band in the range with no hibernation has `have = 0` and is NaN anyway.

## 7. Finishing

After the last sample: if a run is open, close it (a pending gap is never
bridged); then, if a hibernation is active and credited, commit its `HIB_SCORE`
to its band. Every credited hibernation has now been committed exactly once.

## 8. Outputs

For each band: if `have = 0`, `hibernation` is NaN; otherwise it is `best` times
the step (a `best` still `UNSET` is an internal error and raises). `joint_total`
and `joint_longest` are their counters times the step. The pipeline then sets all
three to NaN at steep pixels (Horn slope above `max_slope_deg`, or slope not
computable) and at invalid DEM nodes.

## 9. Worked traces

`.` = sunlit and Earth available, `S` = shadow, `o` = outage but sunlit,
`X` = shadow and outage. 6-hour step, one band. Columns after each sample:
`L` = `RUN_LEN`, `gap`, `hib` = `IN_HIB`, `head`, `hact` = `HEAD_ACTIVE`, `tail` =
`TAIL_ACTIVE`, `pcur` = `PEND_CUR`, `mid`, `best` (samples).

**Example 1.** `..XXoo.XXX..`, merge gap 24 h (limit 4 samples). The first
hibernation (samples 2-5) is the head; its last two samples are sunlit, so they
are in the pending gap. The gap (samples 4-6) lasts 3 samples, 18 h, so it is
bridged by the shadow at sample 7, which also starts a second hibernation.

```
t  ch  L gap hib head hact tail pcur mid    best
2  X   1  0   1   1    1    0    0    -      .    run opens; hibernation 1 is the head
3  X   2  0   1   1    1    0    0    -      .
4  o   2  1   1   1    1    0    0    -      .    sunlit: the gap begins
5  o   2  2   1   1    1    0    0    -      .
6  .   2  3   0   1    0    0    0    -      .    hibernation 1 ends (head stays)
7  X   6  0   1   1    0    1    0    -      .    shadow: bridged, L = 2+3+1; hib 2 starts
8  X   7  0   1   1    0    1    0    -      .
9  X   8  0   1   1    0    1    0    -      .
10 .   8  1   0   1    0    0    0    0..0   .    hib 2 ends: an ended member of the run
11 .   8  2   0   1    0    0    0    0..0   .
finish: close with L = 8: head scores 8, mid scores 8  ->  best = 8 samples = 48 h
```

**Example 2.** `..XX...XXXXXX..`, merge gap 12 h (limit 2). The gap after sample 3
reaches 2 samples at sample 5 and the run closes, so the two hibernations are
scored separately, 12 h and 36 h, and the month reports the better, 12 h.

```
t  ch  L gap hib head hact tail pcur mid    best
3  X   2  0   1   1    1    0    0    -      .
4  .   2  1   0   1    0    0    0    -      .    hibernation 1 has ended; the run is still open
5  .   0  0   0   0    0    0    0    -      2    gap = 2 >= limit: close, head scores 2
7  X   1  0   1   1    1    0    0    -      2    a new run; hibernation 2 is its head
12 X   6  0   1   1    1    0    0    -      2
13 .   6  1   0   1    0    0    0    -      2
14 .   0  0   0   0    0    0    0    -      2    gap = 2: close, head scores 6 (min stays 2)
final best = 2 samples = 12 h
```

(The second close commits `min(2, 6) = 2`.)

**Example 3 (summary).** `SSXSSoSSSoSSXSS` with a 24 h gap limit is one run of 15
samples that touches four hibernations, each a single outage sample (samples 2,
5, 9 and 12). All four score 15 samples = 90 h, however many bands they fall in.
This is the case the summary is built for. The hibernation at sample 2 starts
inside the run, so it is a non-head member and becomes an ended member (`mid`)
when it ends. The ones at samples 5 and 9 begin in sunlit gaps: they are pending
(`pcur`) until the next shadow sample bridges the gap and merges them into `mid`.
The one at sample 12 is another non-head member. When the run closes at the end of
the data, one commit over `mid` scores all four.

## 10. Why this is correct

Let a run `R` have final length `L`, and let `h_1 .. h_k` be the hibernations
that intersect it, in time order.

* **Contiguity.** `R`'s span is contiguous in time, so every hibernation that
  starts between the first and last intersected hibernation also intersects it.
  The chain is a contiguous block, and the bands of its ended members lie in the
  range `MID_LO..MID_HI`, in which every hibernation that starts is a member. A
  band in the range with no hibernation has `have = 0` and is NaN.
* **Only the head has a prior.** A run that closed before `R` began ended before
  `R`'s first sample. A hibernation that starts after `R`'s first sample cannot
  have intersected it. So only `h_1`, if it was active when `R` opened, can have
  earlier intersecting runs: `HEAD_PRIOR`.
* **Other ended members score exactly `L`.** Any run that intersects such a
  hibernation overlaps `R`'s span, and runs do not overlap, so it is `R`.
* **The active member is not settled.** It can still intersect runs that begin
  after `R` closes, so it continues as the active hibernation with the score it
  has so far, and is committed only when it ends with no open run to wait for,
  or at the end of the data.
* **Pending hibernations.** Samples in a gap are in the run's span only if a
  shadow sample follows within the merge limit. Until then the machine records
  hibernations that began there (`PEND_CUR`, `PEND_LO..PEND_HI`) without
  attributing them. A later shadow sample moves them into the chain; reaching the
  merge limit instead means they intersected nothing and score 0.
* **Exactly once.** Every credited hibernation is committed once: when it ends
  with no run waiting, when a run it belongs to closes (head, mid), when it is
  the active member at the end of the run, or at the end of the data.

## 11. Verification

* **Offline oracle.** `tests/numba_horizon/test_safe_haven_state_machine.py`
  contains an independent implementation written the way a list-based
  algorithm would be, one small function per rule of section 2: build the whole
  shadow and outage timelines, merge runs, find hibernations, intersect them,
  and take the minimum per month. The streaming
  machine and the oracle must agree exactly on random timelines: 10 combinations
  of step (6 h and 1 h) and merge gap (0, 1, 12, 24 and 100 h), 400 pixels each,
  260 samples, 7 bands with random boundaries and samples outside every band.
  The timelines include flicker and one-sample gaps. In the cases measured, 792 of
  800 sampled pixels had a month with two starts and 700 had a run touching three
  or more hibernations.
* **Hand-derived cases** for each rule, including the three examples above.
* **Mutation checks.** Eleven deliberate defects, each caught by the oracle test:
  minimum replaced by maximum; ended members not committed; pending
  hibernations not scored 0; the pending range not merged on a bridge; the gap
  limit off by one; the head's prior ignored; the active member's score not set
  when a run closes; an ended head not committed; a hibernation at the first
  sample credited; the joint run credited to the sample's month instead of its
  start month; `PEND_CUR` not cleared when a run closes. Two further mutants
  survived and were examined: an explicit "promote the pending hibernation at a
  bridge" line, which was dead code (a general rule already does it) and was
  removed; and clearing `PEND_CUR` at a bridge, which does not change any output
  but keeps the stated invariant true, so it was kept.
* **Plain Python against compiled.** One test runs the Python form and the
  Numba form on the same random timelines and requires identical output.
* **CPU against CUDA.** The GPU tests (opt-in, `LUNARSCOUT_REQUIRE_NUMBA_CUDA=1`)
  compare the CUDA machine with the CPU machine on the same signals.

## 12. Conventions chosen, in one place

* Month of a hibernation = month of its first outage sample.
* Several starts in one month: the minimum (best) score.
* A hibernation under way at the first sample is not credited to a month.
* A sunlit gap bridges only if strictly shorter than `merge_gap_hours`; with
  coarse sampling the threshold is accurate to about one sample.
* A sunlit sample inside a bridged gap counts toward the run's length, and an
  outage sample there counts as intersecting the run.
* Outage and shadow comparisons are strict; operating Sun uses a 1e-6 tolerance.
* Joint totals count each sample in its own month; the joint longest run goes to
  the month it starts in.
* The first/last samples limit runs to the available data.

## 13. Cost

Per pixel the state is 17 integers plus four small per-band arrays; per sample
the work is a fixed number of integer operations, plus one loop over a range of
bands each time a run closes. Memory does not depend on the number of samples.
The same functions run in a CUDA kernel; the state table lives in global memory
between time batches.

On the CPU, the reducer takes whole batches of time steps (32 by default) and
each thread advances its pixels through every step of the batch, so one parallel
call does many steps of work per thread. Calling it once per time step spent
most of its time launching the parallel region.

The pipeline overlaps its stages: horizon tiles are read and decoded ahead by two
threads, the calculation runs on the calling thread (the CUDA context belongs
to it), and one thread writes the three products behind it, in order, so each
patch is journaled durably as before.

Measured on one laptop (Core Ultra 9 275HX: 8 performance and 16 efficiency
cores; RTX 5090 Laptop GPU), a synthetic 384 x 384 scene of 9 patches with 2,922
six-hour samples, through `generate_safe_haven_products`:

| Backend | Before overlap and batching | After |
|---|---|---|
| CUDA | 2.2 s | 0.85 s |
| CPU (24 threads) | 6.9 to 8.5 s | 4.9 to 5.3 s |

On the CPU, speedup over one thread at 24 threads rose from 3.9x to 6.9x. For one
patch the CUDA calculation itself takes about 0.04 s, against about 0.7 s on
the CPU. The CPU stages that remain are the Sun-fraction and Earth-elevation
calculations, which are memory-bound; the busiest stage of the CUDA run is now
writing the three products. These are single runs on one machine and scene.
