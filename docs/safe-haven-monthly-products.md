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
| Chains (monthly) | the routes of the remaining parts, with the same havens as Seed haven areas; two filters, for touching a PSR and for the longest hop |
| Havens and cluster boxes (monthly) | the cluster boxes, with the same havens as Seed haven areas drawn over them |
| All havens (monthly) | every haven of the month |
| Chains (yearly) | the longest chain of each root in the year it starts, all seeds of the year |

Every layer is drawn over the permanent-shadow background (below).

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

**Two filters, chosen in the viewer.** Both are properties of the whole chain (root to last seed), so a chain that passes
passes in every month it is drawn. They apply together: a chain is shown when it passes both.

* **Touching a PSR**: *Any chain* (the default), *Small or larger*, *Medium or larger*, *Large*. A chain touches a PSR
  when the chain table says its routes pass next to one (a PSR pixel among the eight neighbours of a route pixel):
  `chain_psr_small`, `chain_psr_medium` and `chain_psr_large` count the distinct PSRs of each size class (limits
  `psr_small_max_m2` 400 m2 and `psr_medium_max_m2` 1,000,000 m2). The chain's **PSR class** is 3 if it passes a large PSR,
  else 2 if a medium one, else 1 if a small one, else 0; a chain passes the filter when its class is at least the choice
  (*Small or larger* is any PSR, *Medium or larger* a medium or large one).
* **Longest hop at least**: *Off* (the default), 50, 100, 200, 500, 1,000 or 2,000 m. A chain's **longest hop** is the
  greatest straight-line distance between successive seeds of its longest chain, measured between the seeds' pixel centres
  (`hypot(row difference, column difference)` times the DEM's pixel size, metres). For a chain of 2 seeds, the distance
  between them. A chain passes when its longest hop is at least the choice (equal passes).

**What is drawn for a chain that passes** is what the layer draws for it with no filter: its route, PSR marks, magenta mark
on its current start, its current start, the havens within the radius of that start and its lone last seed. Nothing is
drawn for a chain that does not pass. A pixel shared by several chains is drawn when at least one of the chains that
**own** it passes, with the same precedence as now (aqua over route colours, a haven over aqua, the start mark on top). A
current start or lone seed is owned by its chain; a haven pixel of the area is owned by every chain whose current start it is
within the radius of.

**Product.** One file, `safe_haven_stage_4_monthly_chain_pixels.npz`, keeps each chain's pixels separate: for each month the
pixels (row, column, colour class) with the chain that owns each, and a table of the chains (the root seed, PSR class and
longest hop in metres). The viewer picks the chains that pass and draws them. The three `..._psr_<size>.npz` files, the
`PSR_SIZES` option and the layer "Chains (monthly, touch PSR)" of the earlier version are no longer made or offered. When the chain table has no PSR
counts (stage 4 ran without the routes) the file says the classes are unknown and the viewer offers only *Any chain*. The
caption gives the chains shown and the chains drawn without a filter. With both filters off the picture is the one made
today.

The step reads the seed haven areas file, so it runs after `seed-haven-areas` and refuses to run (error code
`chain_maps_seed_areas_missing`) when that file does not exist. It does not fall back to another set of havens. The areas
file records, for each area pixel, the chains that own it (their root seed ids; `psr_level` stays, the largest PSR class of
them). The step builds each chain's pixels from the routes directly, not by drawing 30,336 by 30,336 images, so it should
be faster than today's.

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

## Permanent shadow background

The ground of every layer is the hillshade with the permanent shadow on it. The pixels of `psr.tif` (255 is PSR, 0 is not; the
DEM's grid, 30,336 by 30,336 pixels of 20 m) are composited onto the hillshade **at full resolution**: a PSR pixel takes the
hillshade's grey 70% and light blue `#8ecbff` 30% (30% opacity). The background pyramid is then made from that picture, each
coarser level the **mean** of 2 x 2 blocks of the level below, so a coarser pixel shows the fraction of its block that is
shadow: a block of a few PSR pixels stays nearly the hillshade, a block that is mostly PSR is clearly blue, and a small PSR
fades at low zoom and shows at full strength when you zoom in. The pyramid is RGB (about 3.7 GB for the full pole) and is
made once; it is made again when `psr.tif`, the DEM, the colour or the opacity change. The viewer reads `PSR_PATH` (default
the scenario's `psr.tif`). A legend entry names the colour ("permanent shadow"). Without the file the viewer draws the
hillshade alone and prints a note. Every layer is drawn over this ground.

## Exporting the pictures

`scripts/export_safe_haven_images.py` writes the viewer's pictures as full-resolution GeoTIFFs on the DEM's grid, tiled and
DEFLATE compressed: `hillshade_psr_rgb.tif` (the background above as 3-band RGB) and, per month, the Chains (monthly) picture of the
chains that pass the two filters (`PSR_CLASS`, `HOP_M`; default medium or larger and 500 m) as a 4-band RGBA image without the
background, named `chains_monthly_<psr>_hop_<metres>m_<month>.tif`. The background is compressed with `--background-compression`: `deflate` (lossless, default), `webp` (lossless, smaller, slower, needs a WebP-enabled reader) or `jpeg` (lossy, `--background-quality`, default 95); the chain images are always lossless. Alpha is 255 where a pixel is drawn and 0 elsewhere. A month with no chain
passing gets no file (`--write-empty` writes a transparent one). The pixels are those the viewer draws at full zoom. It needs the viewer
to have been prepared first (the cache holds the background).

## Regeneration

* `seed-haven-areas` depends on the chain table, the DEM and the haven-months images. A new radius rebuilds it.
* `monthly-chain-maps` depends on the chain table, the light store and passable raster (for the routes) and the seed
  haven areas. Changing the radius rebuilds both.
* The viewer rebuilds its lists when the areas file changes.
* The driver (`scripts/run_safe_haven.py`) runs `haven-months`, then `seed-haven-areas`, then `monthly-chain-maps`.
* `monthly-chain-maps` makes the one file with every chain's pixels and its chain table; the areas file must carry the owners of its pixels (`chain_maps_seed_areas_outdated` otherwise), so a file made before them is made again.

## Open

* Whether a cluster box should also be drawn in later months for the continuation of a chain, around the seeds not yet
  passed. Not decided. The boxes stay as defined above.
