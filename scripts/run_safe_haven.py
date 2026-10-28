#!/usr/bin/env python3
"""Edit the constants below or give them on the command line, then run this script with the project environment.

    .venv/bin/python scripts/run_safe_haven.py [--steps stage3 stage4 chain-maps] [--max-slope-deg 15] [--dry-run]

The safe-haven sequence, one step after another, with the parameters that the steps share set once, here:

    psr-regions   scripts/generate_psr_regions.py     the PSR id raster and region table
    light-store   scripts/run_light_store.py           stage 0: the thresholded-lighting store (a long GPU run)
    stage1        scripts/run_safe_haven_stage1.py     the three screening products
    stage2        scripts/run_safe_haven_stage2.py     the plan seeds
    stage3        scripts/run_safe_haven_stage3.py     the forward floods and the edges
    stage4        scripts/run_safe_haven_stage4.py     the chains and the final table
    chain-maps    scripts/run_safe_haven_chain_maps.py the colour images and GeoPackages

The driver loads those scripts and calls the procedures they define (run_stage3(...) and so on) with the values below,
so a threshold such as MAX_SLOPE_DEG (the Horn slope the rover may enter or stay on, and the limit above which stage 1
marks a pixel NODATA) is the same in every step that uses it. Each script still runs on its own, with its own constants
or command-line options.

Like make, it does not redo what is up to date. A step is up to date when this driver has run it with the same
parameters (recorded in STATE_PATH, with the parameters that change a result: not the numbers of workers), all its
outputs exist, and none of its inputs is newer than its oldest output. A step whose input was rebuilt (an upstream step
ran again, or a file changed) or whose parameters changed is run again, and so is everything after it that depends on
what it wrote. Outputs that exist but that this driver never recorded are not trusted and not replaced: the step stops
with a message; --adopt-existing STEP records them as made with the current parameters (nothing is run) and --force STEP
runs the step and replaces them. --dry-run only reports what each step would do. Code changes are not tracked: use
--force after changing a script.

A stale step replaces its outputs (the scripts' OVERWRITE is set by the driver). The driver stops at the first step that
fails or is interrupted; run it again to continue (the long steps resume). It needs the NVIDIA GPU only for the
light-store step when LIGHTING is auto or cuda. Importing this file starts nothing.

Progress and status messages are logged: to the console, colored with rich on a terminal when
`pip install "lunarscout[console]"` is installed and plain when piped, and to a plain-text log file
(LOG_PATH, COLOR and LOG_LEVEL below).
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.util
import json
import logging
import os
import sys
import time as clock
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np  # noqa: E402

from lunarscout._numba_horizon.safe_haven_floods import FloodSettings  # noqa: E402
from lunarscout._numba_horizon.safe_haven_windows import WindowParameters  # noqa: E402
from lunarscout._script_cli import Option, add_logging_options, add_options, defaults_from, options_from, run_with_logging  # noqa: E402

SCRIPTS = Path(__file__).resolve().parent
STEPS = ("psr-regions", "light-store", "stage1", "stage2", "stage3", "stage4", "chain-maps")

# ---- where things are. Edit these before running; every other path is made from them.
SCENARIO_DIRECTORY = Path("/e/lunar_analyst_scenarios/70south")
DEM_PATH = Path("/d/viper/maps/lola/ldem_80s_20m.clipped.tif")
# None for the paths below means "inside SCENARIO_DIRECTORY" (the name in the comment), so that moving the scenario
# directory moves them all.
HORIZONS_PATH = None                    # lighting/horizons
PSR_PATH = None                         # psr.tif: 255 = PSR, 0 = not PSR
PSR_IDS_PATH = None                     # psr_ids.tif
PSR_TABLE_PATH = None                   # psr_regions.csv
LIGHT_STORE_PATH = None                 # light_store
OUTPUT_DIRECTORY = None                 # safe_havens: stages 1 to 4 and the chain maps
STATE_PATH = None                       # <OUTPUT_DIRECTORY>/safe_haven_run_state.json: what this driver ran, and how

# ---- the time grid. STOP_UTC is exclusive; the last sample of stages 1 and 2 is the step before it.
START_UTC = "2024-01-01T00:00:00"
STOP_UTC = "2032-01-01T00:00:00"
STEP_MINUTES = 120

# ---- lighting thresholds, the same in the light store and stages 1 and 2
EARTH_ELEVATION_THRESHOLD_DEG = 2.0     # Earth below this elevation is an outage
SUNLIGHT_FRACTION_THRESHOLD = 0.2       # a Sun fraction below this is power-negative shadow
OPERATING_SUNLIGHT_FRACTION = 1.0       # at or above this the rover operates
MERGE_GAP_HOURS = 24.0                  # shadow runs closer than this are one run
DEGENERATE_OUTAGE_WINDOW_HOURS = 24.0   # the one-sample outage rule where Earth never sets
OBSERVER_HEIGHT_M = 0.0                 # must match the horizon tiles

# ---- the rover
MAX_SLOPE_DEG = 15.0                    # the Horn slope of a pixel it may enter or stay on; stage 1 marks steeper pixels NODATA
MOVES_PER_STEP = 4                      # one-pixel moves per step (0.0111 m/s at 20 m and 2-hour steps)
CONNECTIVITY = 8                        # 4 or 8 neighbours per move (also joins a PSR's pixels)
MAX_DAYS = 28.0                         # a flood, and the lit endurance, end here
PSR_SMALL_MAX_M2 = 400.0                # PSR size classes: small up to this (one pixel at 20 m), ...
PSR_MEDIUM_MAX_M2 = 1_000_000.0         # ... medium up to this, large above
REACH_HOURS = 48.0                      # the reach in pixels is counted this long after the boundary
EDGE_RULE = "next_cycle"                # which seeds a flood may arrive at: next_cycle or any_later

# ---- the seed rules (stage 2)
SURVIVAL_LIMIT_HOURS = 72.0
OPERATIONAL_HOURS = 48.0
MAX_SEEDS_PER_CELL_MONTH = 5
SEED_SEPARATION_M = 250.0
SEED_TIMING_TOLERANCE_HOURS = 6.0
HAVEN_NEIGHBOURHOOD_M = 100.0
MIN_NEARBY_HAVEN_FRACTION = 0.05
BOUNDARY_SEARCH_HOURS = 24.0
CELL_PATCHES = 2

# ---- how the light store is laid out (a different value needs a new store directory)
STEPS_PER_CHUNK = 96                    # independently compressed chunks of this many steps (96 = 8 days at 2 hours)
CODEC = "zstd-3"

# ---- the chain maps
DILATE_PIXELS = 2
RASTER_ROUTES = True

# ---- resources: they never change a result, so they are not part of the recorded parameters
LIGHTING = "auto"                       # light store: auto uses the GPU when there is one, cuda requires it, cpu never
BACKEND = "auto"                        # stage 1: auto, cpu or cuda
LIGHT_WORKERS = 8
PLAN_WORKERS = 16
FLOOD_WORKERS = 16
ROUTE_WORKERS = 16

# ---- what to do
RUN_STEPS = STEPS                       # a subset, in order
FORCE: tuple = ()                       # steps to run whatever their state
ADOPT_EXISTING: tuple = ()              # steps whose existing outputs are recorded as made with these parameters
DRY_RUN = False                         # only report

LOG_PATH = None                         # None writes <OUTPUT_DIRECTORY>/safe_haven_run.log
COLOR = "auto"
LOG_LEVEL = "INFO"
LOG = logging.getLogger("lunarscout.scripts.safe_haven")

OPTIONS = (
    Option("SCENARIO_DIRECTORY", "path", "the scenario directory: the paths below default to places inside it"),
    Option("DEM_PATH", "path", "the DEM GeoTIFF"),
    Option("HORIZONS_PATH", "optional_path", "the horizon tile store (none: <scenario>/lighting/horizons)"),
    Option("PSR_PATH", "optional_path", "the PSR map: 255 = PSR, 0 = not PSR (none: <scenario>/psr.tif)"),
    Option("PSR_IDS_PATH", "optional_path", "the PSR id raster (none: <scenario>/psr_ids.tif)"),
    Option("PSR_TABLE_PATH", "optional_path", "the PSR region table (none: <scenario>/psr_regions.csv)"),
    Option("LIGHT_STORE_PATH", "optional_path", "the light store directory (none: <scenario>/light_store)"),
    Option("OUTPUT_DIRECTORY", "optional_path", "where stages 1 to 4 and the chain maps write (none: <scenario>/safe_havens)"),
    Option("STATE_PATH", "optional_path", "the driver's record of what it ran (none: <output>/safe_haven_run_state.json)"),
    Option("START_UTC", "str", "first step (UTC)"),
    Option("STOP_UTC", "str", "end of the range, exclusive (UTC)"),
    Option("STEP_MINUTES", "int", "the step; it divides 24 hours"),
    Option("EARTH_ELEVATION_THRESHOLD_DEG", "float", "Earth below this elevation is an outage"),
    Option("SUNLIGHT_FRACTION_THRESHOLD", "float", "a Sun fraction below this is shadow"),
    Option("OPERATING_SUNLIGHT_FRACTION", "float", "Sun fraction at or above which the rover operates"),
    Option("MERGE_GAP_HOURS", "float", "shadow runs closer than this are one run"),
    Option("DEGENERATE_OUTAGE_WINDOW_HOURS", "float", "the one-sample outage rule where Earth never sets"),
    Option("OBSERVER_HEIGHT_M", "float", "observer height (m); must match the horizon tiles"),
    Option("MAX_SLOPE_DEG", "float", "the rover's Horn slope limit (degrees); stage 1's NODATA limit too"),
    Option("MOVES_PER_STEP", "int", "one-pixel moves per step"),
    Option("CONNECTIVITY", "int", "4 or 8 neighbours per move", choices=(4, 8)),
    Option("MAX_DAYS", "float", "a flood, and the lit endurance, end here (days)"),
    Option("PSR_SMALL_MAX_M2", "float", "PSR size classes: small up to this area (m2)"),
    Option("PSR_MEDIUM_MAX_M2", "float", "PSR size classes: medium up to this area (m2), large above"),
    Option("REACH_HOURS", "float", "the reach in pixels is counted this long after the boundary"),
    Option("EDGE_RULE", "str", "which seeds a flood may arrive at", choices=("next_cycle", "any_later")),
    Option("SURVIVAL_LIMIT_HOURS", "float", "the longest hibernation a haven may have"),
    Option("OPERATIONAL_HOURS", "float", "operating time wanted on each side of a hibernation"),
    Option("MAX_SEEDS_PER_CELL_MONTH", "int", "seeds per cell and month"),
    Option("SEED_SEPARATION_M", "float", "seeds are at least this far apart unless their timing differs (m)"),
    Option("SEED_TIMING_TOLERANCE_HOURS", "float", "outage timing difference that lets seeds be closer"),
    Option("HAVEN_NEIGHBOURHOOD_M", "float", "radius of the neighbourhood checked for other havens (m)"),
    Option("MIN_NEARBY_HAVEN_FRACTION", "float", "the share of that neighbourhood that must be havens"),
    Option("BOUNDARY_SEARCH_HOURS", "float", "how far to look for the lit boundary around an outage"),
    Option("CELL_PATCHES", "int", "cell size in patches of 128 pixels"),
    Option("STEPS_PER_CHUNK", "int", "light store: steps per independently compressed chunk"),
    Option("CODEC", "str", "light store: chunk codec"),
    Option("DILATE_PIXELS", "int", "the dilated chain images' lines are 2 * this + 1 pixels wide"),
    Option("RASTER_ROUTES", "bool", "the chain images follow the rover's routes; false draws straight lines"),
    Option("LIGHTING", "str", "light store lighting", choices=("auto", "cuda", "cpu")),
    Option("BACKEND", "str", "stage 1 backend", choices=("auto", "cpu", "cuda")),
    Option("LIGHT_WORKERS", "int", "light store: patches in flight"),
    Option("PLAN_WORKERS", "int", "stage 2: processes"),
    Option("FLOOD_WORKERS", "int", "stage 3: processes"),
    Option("ROUTE_WORKERS", "int", "stage 4 and the chain maps: processes"),
    Option("RUN_STEPS", "strs", "the steps to run, in order", choices=STEPS),
    Option("FORCE", "strs", "steps to run whatever their state", choices=STEPS),
    Option("ADOPT_EXISTING", "strs", "steps whose existing outputs are recorded as made with these parameters", choices=STEPS),
    Option("DRY_RUN", "bool", "only report what each step would do"),
)

#: Parameters that change how a step runs but not what it makes: left out of the recorded fingerprint.
OPERATIONAL = frozenset({
    "workers", "plan_workers", "flood_workers", "route_workers", "progress_every_cells", "progress_every_groups",
    "progress_every_seeds", "overwrite", "start_fresh", "verbose", "status_seconds", "gpu_device_id",
    "gpu_chunks_per_batch", "unit_cache_mib", "lighting", "backend", "seed_ids_trial",
})

_SCRIPTS: dict = {}


def load_script(name: str):
    """The module of ``scripts/<name>.py``, loaded once (the same object every time, so it can be patched)."""
    if name not in _SCRIPTS:
        spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        _SCRIPTS[name] = module
    return _SCRIPTS[name]


# ---------------------------------------------------------------------------
# The steps
# ---------------------------------------------------------------------------


@dataclass
class Step:
    """One step: the script and procedure to call, the values to call it with (the script's defaults overridden by
    this driver's parameters), and the files it reads and writes."""

    name: str
    script: str
    procedure: str
    options: dict
    inputs: list
    outputs: list            # files whose times say whether the step is up to date
    caches: list             # files the step keeps for itself: they must exist, their times are not compared

    def call(self, *, overwrite: bool) -> int:
        module = load_script(self.script)
        values = dict(self.options)
        if "overwrite" in values:
            values["overwrite"] = overwrite
        return getattr(module, self.procedure)(**values)


def last_sample(stop_utc: str, step_minutes: int) -> str:
    """The last sample of the grid that ends before ``stop_utc`` (exclusive), as ``YYYY-MM-DDTHH:MM:SSZ``."""
    last = np.datetime64(stop_utc.rstrip("Z"), "s") - np.timedelta64(step_minutes, "m")
    return str(last) + "Z"


def flood_settings(p: dict) -> FloodSettings:
    return FloodSettings(max_days=p["max_days"], moves_per_step=p["moves_per_step"], connectivity=p["connectivity"],
                         max_slope_deg=p["max_slope_deg"], psr_small_max_m2=p["psr_small_max_m2"],
                         psr_medium_max_m2=p["psr_medium_max_m2"], reach_hours=p["reach_hours"], edge_rule=p["edge_rule"])


def window_parameters(p: dict) -> WindowParameters:
    return WindowParameters(
        survival_limit_hours=p["survival_limit_hours"], operational_hours=p["operational_hours"],
        flood_step_minutes=p["step_minutes"], moves_per_step=p["moves_per_step"],
        max_seeds_per_cell_month=p["max_seeds_per_cell_month"], seed_separation_m=p["seed_separation_m"],
        seed_timing_tolerance_hours=p["seed_timing_tolerance_hours"], haven_neighbourhood_m=p["haven_neighbourhood_m"],
        min_nearby_haven_fraction=p["min_nearby_haven_fraction"], boundary_search_hours=p["boundary_search_hours"],
        operating_sunlight_fraction=p["operating_sunlight_fraction"], cell_patches=p["cell_patches"],
        degenerate_outage_window_hours=p["degenerate_outage_window_hours"], forward_only=True)


def resolve_paths(p: dict) -> dict:
    """``p`` with every path that is None put inside the scenario directory."""
    scenario = Path(p["scenario_directory"])
    named = {"horizons_path": "lighting/horizons", "psr_path": "psr.tif", "psr_ids_path": "psr_ids.tif",
             "psr_table_path": "psr_regions.csv", "light_store_path": "light_store", "output_directory": "safe_havens"}
    resolved = {**p}
    for key, name in named.items():
        resolved[key] = Path(p[key]) if p[key] is not None else scenario / name
    resolved["state_path"] = (Path(p["state_path"]) if p["state_path"] is not None
                              else resolved["output_directory"] / "safe_haven_run_state.json")
    return resolved


def build_steps(p: dict) -> dict[str, Step]:
    """The steps for the parameters ``p`` (the driver's options by their lower-case names, paths resolved)."""
    out = Path(p["output_directory"])
    store = Path(p["light_store_path"])
    stage1 = {"hibernation": out / "safe_haven_stage_1_hibernation_hours.tif",
              "total": out / "safe_haven_stage_1_joint_operating_total_hours.tif",
              "longest": out / "safe_haven_stage_1_joint_operating_longest_hours.tif"}
    seeds = out / "safe_haven_stage_2_plan_seeds.csv"
    floods, edges = out / "safe_haven_stage_3_floods.csv", out / "safe_haven_stage_3_edges.csv"
    passable = out / "safe_haven_stage_3_passable_mask.npy"
    chains = out / "safe_haven_stage_4_chains.csv"
    maps = [out / f"safe_haven_stage_4_{name}{suffix}" for name in ("predecessor_edges", "longest_chain_per_root")
            for suffix in (".tif", "_dilated.tif", ".gpkg")]
    first, stop = p["start_utc"].rstrip("Z"), p["stop_utc"].rstrip("Z")
    last = last_sample(p["stop_utc"], p["step_minutes"])
    step_hours = p["step_minutes"] / 60.0
    steps: dict[str, Step] = {}

    def add(name, script, procedure, overrides, inputs, outputs, caches=()):
        base = load_script(script).default_options()
        unknown = set(overrides) - set(base)
        if unknown:
            raise KeyError(f"{script} has no option(s) {sorted(unknown)}")
        steps[name] = Step(name, script, procedure, {**base, **overrides}, list(inputs), list(outputs), list(caches))

    add("psr-regions", "generate_psr_regions", "run_psr_regions",
        dict(psr_path=p["psr_path"], ids_path=p["psr_ids_path"], table_path=p["psr_table_path"],
             connectivity=p["connectivity"], overwrite=True),
        [p["psr_path"]], [p["psr_ids_path"], p["psr_table_path"]])
    add("light-store", "run_light_store", "run_light_store",
        dict(dem_path=p["dem_path"], horizons_path=p["horizons_path"], output_directory=store,
             observer_height_m=p["observer_height_m"], start_utc=first, stop_utc=stop, step_minutes=p["step_minutes"],
             shadow_fraction=p["sunlight_fraction_threshold"], operating_fraction=p["operating_sunlight_fraction"],
             earth_elevation_deg=p["earth_elevation_threshold_deg"], steps_per_chunk=p["steps_per_chunk"],
             codec=p["codec"], lighting=p["lighting"], workers=p["light_workers"]),
        [p["dem_path"], p["horizons_path"]], [store])
    add("stage1", "run_safe_haven_stage1", "run_stage1",
        dict(dem_path=p["dem_path"], horizons_path=p["horizons_path"], light_store_path=store, output_directory=out,
             name_prefix="safe_haven_stage_1", start_utc=first + "Z", stop_utc=last, step_hours=step_hours,
             earth_elevation_threshold_deg=p["earth_elevation_threshold_deg"],
             sunlight_fraction_threshold=p["sunlight_fraction_threshold"],
             operating_sunlight_fraction=p["operating_sunlight_fraction"], merge_gap_hours=p["merge_gap_hours"],
             degenerate_outage_window_hours=p["degenerate_outage_window_hours"], max_slope_deg=p["max_slope_deg"],
             observer_height_m=p["observer_height_m"], backend=p["backend"], overwrite=True, start_fresh=False),
        [p["dem_path"], store], list(stage1.values()))
    add("stage2", "run_safe_haven_stage2", "run_stage2",
        dict(dem_path=p["dem_path"], light_store_path=store, stage1_directory=out, stage1_prefix="safe_haven_stage_1",
             output_path=seeds, overwrite=True, start_utc=first + "Z", stop_utc=last, step_hours=step_hours,
             earth_elevation_threshold_deg=p["earth_elevation_threshold_deg"],
             sunlight_fraction_threshold=p["sunlight_fraction_threshold"], merge_gap_hours=p["merge_gap_hours"],
             plan_workers=p["plan_workers"], parameters=window_parameters(p), cells=None),
        [p["dem_path"], store, stage1["hibernation"], stage1["total"]], [seeds])
    settings = flood_settings(p)
    add("stage3", "run_safe_haven_stage3", "run_stage3",
        dict(dem_path=p["dem_path"], light_store_path=store, seeds_path=seeds, psr_ids_path=p["psr_ids_path"],
             psr_table_path=p["psr_table_path"], passable_path=passable, output_path=floods, edges_path=edges,
             overwrite=True, seed_ids=None, flood_workers=p["flood_workers"], settings=settings),
        [p["dem_path"], store, seeds, p["psr_ids_path"], p["psr_table_path"]], [floods, edges], [passable])
    add("stage4", "run_safe_haven_stage4", "run_stage4",
        dict(dem_path=p["dem_path"], seeds_path=seeds, floods_path=floods, edges_path=edges, output_path=chains,
             step_minutes=p["step_minutes"], light_store_path=store, passable_path=passable,
             psr_ids_path=p["psr_ids_path"], psr_table_path=p["psr_table_path"], route_workers=p["route_workers"],
             overwrite=True, flood_settings=settings),
        [p["dem_path"], seeds, floods, edges, store, passable, p["psr_ids_path"], p["psr_table_path"]], [chains])
    add("chain-maps", "run_safe_haven_chain_maps", "make_chain_maps",
        dict(dem_path=p["dem_path"], chains_path=chains, output_directory=out, dilate_pixels=p["dilate_pixels"],
             overwrite=True, light_store_path=store, psr_ids_path=p["psr_ids_path"], passable_path=passable,
             raster_routes=p["raster_routes"], route_workers=p["route_workers"], flood_settings=settings),
        [p["dem_path"], chains, store, passable, p["psr_ids_path"]], maps)
    return steps


# ---------------------------------------------------------------------------
# What is up to date
# ---------------------------------------------------------------------------


def _jsonable(value):
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {k: _jsonable(v) for k, v in dataclasses.asdict(value).items()}
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def _strip(value):
    if isinstance(value, dict):
        return {k: _strip(v) for k, v in value.items() if k not in OPERATIONAL}
    if isinstance(value, list):
        return [_strip(v) for v in value]
    return value


def parameters_of(step: Step) -> dict:
    """The values that decide what a step makes: its options without the operational ones."""
    return {k: _strip(_jsonable(v)) for k, v in sorted(step.options.items()) if k not in OPERATIONAL}


def fingerprint(step: Step) -> str:
    return hashlib.sha256(json.dumps(parameters_of(step), sort_keys=True).encode()).hexdigest()[:16]


def stamp(path) -> float | None:
    """The modification time of a file, or of the newest file in a directory; None when it is not there."""
    path = Path(path)
    if path.is_file():
        return path.stat().st_mtime
    if path.is_dir():
        newest = None
        for root, _dirs, files in os.walk(path):
            for name in files:
                time = os.stat(os.path.join(root, name)).st_mtime
                newest = time if newest is None or time > newest else newest
        return newest if newest is not None else path.stat().st_mtime
    return None


def read_state(path: Path) -> dict:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def write_state(path: Path, state: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(path.name + ".partial")
    staging.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(staging, path)


@dataclass
class Status:
    """``kind`` is ``current`` (up to date), ``stale`` (to run), ``unrecorded`` (outputs exist that this driver never
    ran) or ``blocked`` (an input is missing)."""

    kind: str
    reason: str


def status_of(step: Step, state: dict, pending: frozenset = frozenset()) -> Status:
    """The state of ``step``; ``pending`` is the outputs of earlier steps that have not run yet (a dry run)."""
    rebuilt = [str(p) for p in step.inputs if str(p) in pending]
    if rebuilt:
        return Status("stale", "an input will be rebuilt: " + ", ".join(rebuilt[:2]))
    missing_inputs = [str(p) for p in step.inputs if stamp(p) is None]
    outputs = [stamp(p) for p in step.outputs]
    present = [t for t in outputs if t is not None]
    exists = len(present) == len(step.outputs) and all(Path(p).exists() for p in step.caches)
    if missing_inputs:
        return Status("blocked", "input missing: " + ", ".join(missing_inputs))
    if not present:
        return Status("stale", "no outputs yet")
    recorded = state.get(step.name)
    if recorded is None:
        return Status("unrecorded", "outputs exist but this driver never ran the step")
    if not exists:
        return Status("stale", "an output is missing")
    if recorded.get("fingerprint") != fingerprint(step):
        changed = sorted(k for k in set(recorded.get("parameters", {})) | set(parameters_of(step))
                         if recorded.get("parameters", {}).get(k) != parameters_of(step).get(k))
        return Status("stale", "parameters changed: " + (", ".join(changed[:6]) or "?"))
    newest_input = max((t for t in (stamp(p) for p in step.inputs) if t is not None), default=0.0)
    oldest_output = min(present)
    if newest_input > oldest_output:
        newer = [str(p) for p in step.inputs if (stamp(p) or 0.0) > oldest_output]
        return Status("stale", "input newer than the outputs: " + ", ".join(newer[:3]))
    return Status("current", "up to date")


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------


def default_options() -> dict:
    """The options with this file's constants as their values (read now)."""
    return {**defaults_from(OPTIONS, globals())}


def run_safe_haven(**options) -> int:
    """Run the steps in ``options["run_steps"]`` that are not up to date; returns 0, or 1 at the first failure or
    unrecorded output. ``options`` are the driver's options by their lower-case names (``default_options()``)."""
    p = resolve_paths({**default_options(), **options})
    unknown = [s for key in ("run_steps", "force", "adopt_existing") for s in p[key] if s not in STEPS]
    if unknown:
        raise ValueError(f"unknown step(s) {sorted(set(unknown))}; the steps are {', '.join(STEPS)}")
    steps = build_steps(p)
    state = read_state(p["state_path"])
    selected = [s for s in STEPS if s in set(p["run_steps"])]
    started = clock.monotonic()
    ran = 0
    pending: set = set()                  # outputs that a dry run says would be rebuilt
    for number, name in enumerate(selected, start=1):
        step = steps[name]
        status = status_of(step, state, frozenset(pending))
        label = f"[{number}/{len(selected)}] {name}"
        if name in p["adopt_existing"] and status.kind in ("unrecorded", "stale") and all(
                stamp(path) is not None for path in step.outputs):
            if not p["dry_run"]:
                state[name] = {"fingerprint": fingerprint(step), "parameters": parameters_of(step), "adopted": True,
                               "finished": time_text()}
                write_state(p["state_path"], state)
            LOG.info("%s: existing outputs %s with the current parameters", label,
                     "would be recorded as made" if p["dry_run"] else "recorded as made")
            continue
        if name in p["force"]:
            action = f"run (forced; was: {status.reason})"
        elif status.kind == "current":
            LOG.info("%s: up to date", label)
            continue
        elif status.kind == "blocked":
            LOG.error("%s: %s", label, status.reason)
            if p["dry_run"]:
                continue
            return 1
        elif status.kind == "unrecorded" and p["dry_run"]:
            LOG.warning("%s: %s; a real run would stop here unless --adopt-existing %s or --force %s is given.",
                        label, status.reason, name, name)
            continue
        elif status.kind == "unrecorded":
            LOG.error("%s: %s. Not replacing it: use --adopt-existing %s to trust it as made with these "
                      "parameters, or --force %s to run the step and replace it.", label, status.reason, name, name)
            return 1
        else:
            action = f"run ({status.reason})"
        if p["dry_run"]:
            LOG.info("%s: would %s", label, action)
            pending.update(str(path) for path in step.outputs + step.caches)
            continue
        LOG.info("%s: %s", label, action)
        step_started = clock.monotonic()
        code = step.call(overwrite=True)
        if code != 0:
            LOG.error("%s stopped with status %s after %.1f min; run the driver again to continue.", name, code,
                      (clock.monotonic() - step_started) / 60)
            return 1
        state[name] = {"fingerprint": fingerprint(step), "parameters": parameters_of(step), "finished": time_text()}
        write_state(p["state_path"], state)
        ran += 1
        LOG.info("%s: done in %.1f min", label, (clock.monotonic() - step_started) / 60)
    if p["dry_run"]:
        LOG.info("Dry run: nothing was run.")
    else:
        LOG.info("Safe-haven sequence finished: %d of %d steps run, %.1f min", ran, len(selected),
                 (clock.monotonic() - started) / 60)
    return 0


def time_text() -> str:
    return clock.strftime("%Y-%m-%dT%H:%M:%S%z")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the safe-haven sequence, skipping what is up to date. Every option's default is the constant "
                    "of the same name at the top of this file.")
    add_options(parser, OPTIONS, default_options())
    parser.add_argument("--list-steps", action="store_true", help="print the steps and exit")
    add_logging_options(parser, globals())
    return parser


def main(argv=()) -> int:
    """Read the options in ``argv`` (none: the defaults), set up logging and run the sequence."""
    args = build_parser().parse_args(list(argv))
    if args.list_steps:
        print("\n".join(STEPS))
        return 0
    options = options_from(args, OPTIONS)
    output_directory = resolve_paths({**default_options(), **options})["output_directory"]
    output_directory.mkdir(parents=True, exist_ok=True)
    return run_with_logging(args, default_log_path=output_directory / "safe_haven_run.log", logger=LOG,
                            label="Safe-haven sequence", procedure=lambda: run_safe_haven(**options))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
