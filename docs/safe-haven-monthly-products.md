# Safe-haven monthly products: the rules

What each monthly layer of the safe-haven viewer shows, and the rules the code that generates them follows. This is
the reference when a layer looks wrong: if the image disagrees with a rule here, the image or the rule is a bug.
`docs/safe-haven-light-store.md` describes the pipeline that produces the inputs.

Units are metres on the DEM's grid. On the 20 m DEM, 200 m is 10 pixels.

## Terms

* **Seed**: a row of the stage-4 chain table. It has a pixel, a shadow period (``hibernation hours``) and a **month**,
  its own `month` column, but never earlier than the month its chain starts in (`root_start_utc`).
* **Chain**: the longest chain of one root, a list of seeds A, B, C, ... with the root first. Only chains of 2 or more
  seeds are chains. Seeds that are on no chain (96% of all seeds) are **isolated**. Seeds on a branch off the longest
  chain are not on the chain.
* **Passed**: a seed is passed once its month is over. The months never go back along a chain.
* **Remaining part** of a chain in month M: its seeds that are not yet passed (their month is M or later), from the
  first of them to the last.
* **Current start** of a chain in month M: the first seed of the remaining part, where the route begins that month. The
  chain is drawn in M only while its remaining part has 2 or more seeds. In the month the chain starts it is the root.
  In a later month it is the continuation: the first seed not yet passed.
* **Lone seed** of a chain in month M: the last seed of a chain whose remaining part is that seed alone.
* **Haven**: a pixel whose stage-1 hibernation count is at most the survival limit (the haven-months images).
  Each pixel keeps its **shadow-period colour**: white for no shadow, yellow up to 12 h, cyan up to 24 h, violet beyond
  that up to the limit.

Example. A chain of three seeds A (month 1), B (month 2) and C (month 3):

| month | route drawn          | remaining seeds | current start | lone seed |
|-------|----------------------|-----------------|---------------|-----------|
| 1     | A to B to C          | 3               | A             |           |
| 2     | B to C               | 2               | B             |           |
| 3     | none                 | 1               |               | C         |

## The layers

| Layer | What it shows |
|-------|---------------|
| Seed haven areas (monthly) | the havens of the month around the current starts |
| Chains (monthly) | the routes of the remaining parts, with the same havens as Seed haven areas |
| Havens and cluster boxes (monthly) | the cluster boxes, with the same havens as Seed haven areas drawn over them |
| All havens (monthly) | every haven of the month |
| Chains (yearly) | the longest chain of each root in the year it starts, all seeds of the year |

### Seed haven areas (monthly)

The havens of month M are:

1. every **current start** of M, at its own shadow colour (the table's hours decide it, not the haven image);
2. every haven pixel of M, seed or not, within `AREA_RADIUS_METERS` (default 200 m) of a current start, at the shadow
   colour of the month's haven image;
3. every **lone seed** of M, as that pixel alone: no haven around it is added.

Nothing else is drawn: isolated seeds, passed seeds, and seeds still ahead on a chain that are neither a current start
nor within the radius of one. A pixel that is in more than one of these appears once. A month without a haven image
shows the current starts and lone seeds alone.

The radius is a setting (`radius_meters` of `ls.generate_seed_haven_areas`; `AREA_RADIUS_METERS` in
`scripts/run_safe_haven.py` and `scripts/make_seed_haven_areas.py`). Products: a GeoTIFF per month in
`safe_haven_stage_4_seed_haven_areas/` and `safe_haven_stage_4_seed_haven_areas.npz`.

### Chains (monthly)

* **Routes.** Each month draws every chain of M that has a remaining part of 2 or more seeds, from its current start to
  its last seed. A chain that goes on into later months is shorter in each, because the seeds already passed drop off
  its start. It is coloured by the length of its remaining part (the intervals of the yearly images), so its colour
  changes as the chain shrinks. Routes, PSR marks and colours are those of the yearly images.
* **PSR marks.** Aqua pixels beside a PSR, as in the yearly images.
* **Start mark.** A magenta pixel on the current start of every chain drawn in M: the root in the month the chain
  starts, the continuation in each later month.
* **Havens.** Exactly the set of Seed haven areas (monthly) for the month, in the same colours, under the routes and
  marks. A route can therefore end at a seed that is not drawn.
* No boxes.

The step reads the seed haven areas file, so it runs after `seed-haven-areas` and refuses to run (error code
`chain_maps_seed_areas_missing`) when that file does not exist. It does not fall back to another set of havens.

### Havens and cluster boxes (monthly)

* **Boxes.** One box per **cluster**: with `--connect chains` (the default), one root's whole tree, the root and every
  seed that descends from it, branches included. A cluster needs 2 or more seeds. The box is the bounding rectangle of
  the pixels of all those seeds, drawn as an outline one pixel wide just outside it, solid when it is only a few pixels
  across on the screen. It is fixed: it does not shrink as seeds are passed. It appears in the one month the root's chain
  starts in. The colour is one aqua hue, brighter for more seeds: 2, 3, 4, 5-6, 7-9, 10-19, 20-49, 50 or more.
* **Havens.** The same set as Seed haven areas (monthly), drawn over the boxes.

### All havens (monthly)

Every pixel that is a haven in the month, from the month's haven image, with no selection by chains or radius.

### Chains (yearly)

Unchanged by the rules above: the longest chain of each root, drawn whole in the year its root starts in, with all the
seeds of the year, isolated ones included.

## Regeneration

* `seed-haven-areas` depends on the chain table, the DEM and the haven-months images. A new radius rebuilds it.
* `monthly-chain-maps` depends on the chain table, the light store and passable raster (for the routes) and the seed
  haven areas. Changing the radius rebuilds both.
* The viewer rebuilds its lists when the areas file changes.
* The driver (`scripts/run_safe_haven.py`) runs `haven-months`, then `seed-haven-areas`, then `monthly-chain-maps`.

## Open

* Whether a cluster box should also be drawn in later months for the continuation of a chain, around the seeds not yet
  passed. Not decided. The boxes stay as defined above.
