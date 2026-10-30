#!/usr/bin/env python3
"""Edit the constants below or give them on the command line, then run this script with the project environment.

    .venv/bin/python scripts/check_safe_haven_lighting.py [--seeds 40 --non-havens 8 --random-seed 1 ...]

Every constant in the OPTIONS list below is a command-line option, --constant-name (lower case, hyphens), and the
constant is its default; the seed rules of PARAMETERS are one flag per field; `--help` lists them.
check_safe_haven_lighting(...) takes the same values as keyword arguments.

Checks the safe havens against a horizon traced by ray casting. It takes SEEDS seeds from the final chain table,
spread over their outage kinds, scores, years and chain lengths, and up to NON_HAVENS pixels beside them that are not
havens. For each pixel it

* traces the horizon of all 1,440 azimuths with exact spherical rays (the geometry of direct_reference_trace, over the
  primary DEM and every DEM of OUTER_DEM_PATHS, as the horizon generator did), without using the horizon tiles;
* lights the pixel from that horizon, from the production horizon tile and from the stage-0 light store for
  WINDOW_DAYS either side of its outage, and compares the three (the lighting arithmetic is the patch kernel's own, so
  a difference comes from the horizon or from the store). A stored bit is explained when the reference reproduces it
  with the horizon shifted down or up by HORIZON_TOLERANCE_DEG; one that is not is a failure;
* runs the stage-2 evaluator on the reference lighting, with the horizon shifted either way as well, and on the stored
  lighting, and compares score, outage start and end and outage kind with the seed's table row.

It writes OUTPUT_DIRECTORY/lighting_check.json (everything, per pixel), lighting_check.csv (one line per pixel) and
lighting_check.md (the summary) and returns 0 when every pixel passed and 1 otherwise. A trace takes about
10 s per pixel. It is CPU only and needs no NVIDIA device; SPICE supplies the Sun and Earth. The DEMs are read whole
(the 20 m DEM is about 3.7 GB). Importing it starts nothing.

Progress and status messages are logged: to the console, colored with rich on a terminal when
`pip install "lunarscout[console]"` is installed and plain when piped, and to a plain-text log
file in OUTPUT_DIRECTORY (LOG_PATH, COLOR and LOG_LEVEL below).
"""

from __future__ import annotations

import argparse
import logging
import sys
import time as clock
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import lunarscout as ls  # noqa: E402
from lunarscout._numba_horizon.file_format import HorizonTileStore  # noqa: E402
from lunarscout._numba_horizon.safe_haven_check import (  # noqa: E402
    FLAT_BASES, CheckSettings, read_chain_table, run_check,
)
from lunarscout._numba_horizon.safe_haven_stage0 import LightStoreDirectory  # noqa: E402
from lunarscout._numba_horizon.safe_haven_windows import WindowParameters, spice_vector_source  # noqa: E402
from lunarscout._script_cli import (  # noqa: E402
    Option, add_logging_options, add_options, add_window_parameters, defaults_from, options_from, run_with_logging,
    window_parameters_from,
)
from lunarscout.products import _load_dem  # noqa: E402


# Edit these paths before running. They match run_safe_haven_stage2.py and scripts/generate_70s_horizons_on_laptop.py.
DEM_PATH = Path("/d/viper/maps/lola/ldem_80s_20m.clipped.tif")
OUTER_DEM_PATHS = ("/d/viper/maps/lola/ldem_70s_118m.tif",)   # the DEMs after the primary that the horizons were made with
HORIZONS_PATH = Path("/e/lunar_analyst_scenarios/70south/lighting/horizons")
LIGHT_STORE_PATH = Path("/e/lunar_analyst_scenarios/70south/light_store")
STAGE_DIRECTORY = Path("/e/lunar_analyst_scenarios/70south/safe_havens")
CHAINS_PATH = STAGE_DIRECTORY / "safe_haven_stage_4_chains.csv"
OUTPUT_DIRECTORY = STAGE_DIRECTORY / "lighting_check"
OVERWRITE = False                       # replace the report files of an earlier check

# What to check.
SEEDS = 40                              # seeds of the chain table, spread over its strata
NON_HAVENS = 8                          # pixels beside seeds that are not havens in the seed's month
RANDOM_SEED = 0                         # the same value gives the same sample
WINDOW_DAYS = 15.0                      # the lighting is compared this long either side of the seed's outage
HORIZON_TOLERANCE_DEG = 0.3             # a stored bit is explained by a horizon shifted by this much (worst ray on real terrain: 0.29)
ARITHMETIC_TOLERANCE = 1e-3             # stored against the same horizon: a value this near a threshold is a tie
FLAT_BASIS = "patch_centre"             # Earth above a flat horizon for the degenerate-outage rule: the patch centre (stage 1 to 3) or the pixel

# Must equal the stage-1 to stage-2 runs.
START_UTC = "2024-01-01T00:00:00Z"
STOP_UTC = "2031-12-31T22:00:00Z"
STEP_HOURS = 2.0
EARTH_ELEVATION_THRESHOLD_DEG = 2.0
SUNLIGHT_FRACTION_THRESHOLD = 0.2
MERGE_GAP_HOURS = 24.0
OBSERVER_HEIGHT_M = 0.0                 # must match the horizon tiles

# The seed rules, as in run_safe_haven_stage2.py (the check uses the survival limit, the operational hours, the
# degenerate-outage window and forward_only).
PARAMETERS = WindowParameters(
    survival_limit_hours=72.0,
    operational_hours=48.0,
    flood_step_minutes=120,
    degenerate_outage_window_hours=24.0,
    forward_only=True,
)

# Messages go to the console (colored with rich on a terminal when "lunarscout[console]" is
# installed; plain when piped) and to a plain-text log file, which replaces `| tee`. LOG_PATH
# None writes next to the outputs. COLOR is "auto", "always" (even into a pipe) or "never".
LOG_PATH = None
COLOR = "auto"
LOG_LEVEL = "INFO"
LOG = logging.getLogger("lunarscout.scripts.check_lighting")


OPTIONS = (
    Option("DEM_PATH", "path", "the primary DEM GeoTIFF (the one the horizon tiles are for)"),
    Option("OUTER_DEM_PATHS", "strs", "the DEMs after the primary that the horizons were made with (none: no more)"),
    Option("HORIZONS_PATH", "path", "the production horizon tiles"),
    Option("LIGHT_STORE_PATH", "path", "the stage-0 light store"),
    Option("CHAINS_PATH", "path", "the final stage-4 chain table (the seeds)"),
    Option("OUTPUT_DIRECTORY", "path", "where lighting_check.json, .csv and .md are written"),
    Option("OVERWRITE", "bool", "replace the report files of an earlier check"),
    Option("SEEDS", "int", "seeds to check, spread over outage kind, score, year and chain length"),
    Option("NON_HAVENS", "int", "non-haven pixels beside seeds to check as well"),
    Option("RANDOM_SEED", "int", "the sample is the same for the same value"),
    Option("WINDOW_DAYS", "float", "days of lighting compared either side of a seed's outage"),
    Option("HORIZON_TOLERANCE_DEG", "float", "a stored bit is explained by a reference horizon shifted this much"),
    Option("ARITHMETIC_TOLERANCE", "float", "stored against the same horizon: a value this near a threshold is a tie"),
    Option("FLAT_BASIS", "str", "where Earth above a flat horizon is taken: patch_centre (stages 1 to 3) or pixel",
           choices=FLAT_BASES),
    Option("START_UTC", "str", "first sample (UTC); must equal the stage-1 run"),
    Option("STOP_UTC", "str", "last sample (UTC); must equal the stage-1 run"),
    Option("STEP_HOURS", "float", "sample step in hours"),
    Option("EARTH_ELEVATION_THRESHOLD_DEG", "float", "Earth below this elevation is an outage"),
    Option("SUNLIGHT_FRACTION_THRESHOLD", "float", "a Sun fraction below this is shadow"),
    Option("MERGE_GAP_HOURS", "float", "shadow runs closer than this are one run"),
    Option("OBSERVER_HEIGHT_M", "float", "the observer height of the horizon tiles"),
)


def default_options() -> dict:
    """The options with this file's constants as their values (read now), the seed rules as ``parameters``."""
    return {**defaults_from(OPTIONS, globals()), "parameters": PARAMETERS}


def check_safe_haven_lighting(*, dem_path, outer_dem_paths, horizons_path, light_store_path, chains_path,
                              output_directory, overwrite, seeds, non_havens, random_seed, window_days,
                              horizon_tolerance_deg, arithmetic_tolerance, flat_basis, start_utc, stop_utc,
                              step_hours, earth_elevation_threshold_deg, sunlight_fraction_threshold,
                              merge_gap_hours, observer_height_m, parameters) -> int:
    """Check the sample and write the reports; returns 0 when every pixel passed, 1 otherwise."""
    started = clock.monotonic()
    LOG.info("Reading the DEMs")
    primary, _georef = _load_dem(dem_path)
    dems = [primary] + [_load_dem(path)[0] for path in outer_dem_paths]
    LOG.info("Opening the light store %s (checking it against the DEM)", light_store_path)
    light_store = LightStoreDirectory(light_store_path, primary)
    LOG.info("Reading the chain table %s", chains_path)
    rows = read_chain_table(chains_path)
    times = ls.times(start_utc, stop_utc, step_hours=step_hours)
    LOG.info("%d seeds in the table; checking %d and up to %d non-havens", len(rows), seeds, non_havens)
    report = run_check(
        output_directory, dems=dems, light_store=light_store, horizon_store=HorizonTileStore(horizons_path),
        chain_rows=rows, times=times.values, sun_vectors_at=spice_vector_source("sun"),
        earth_vectors_at=spice_vector_source("earth"), parameters=parameters, seeds=seeds, non_havens=non_havens,
        random_seed=random_seed,
        settings=CheckSettings(window_days=window_days, horizon_tolerance_deg=horizon_tolerance_deg,
                               arithmetic_tolerance=arithmetic_tolerance, flat_basis=flat_basis),
        earth_elevation_threshold_deg=earth_elevation_threshold_deg,
        sunlight_fraction_threshold=sunlight_fraction_threshold, merge_gap_hours=merge_gap_hours,
        observer_height_m=observer_height_m, overwrite=overwrite, progress=LOG.info)
    LOG.info("%s: lighting reproduced for %d of %d pixels, %s; reports in %s (%.1f min)",
             "PASS" if report.passed else "FAIL", report.lighting_ok, report.lighting_ok + report.lighting_failed,
             report.verdicts, output_directory, (clock.monotonic() - started) / 60)
    return 0 if report.passed else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Check safe havens against a ray-traced horizon: the stored lighting and the haven score. Every "
                    "option's default is the constant of the same name at the top of this file.")
    add_options(parser, OPTIONS, default_options())
    add_window_parameters(parser, PARAMETERS)
    add_logging_options(parser, globals())
    return parser


def main(argv=()) -> int:
    """Read the options in ``argv`` (none: the defaults), set up logging and run the check."""
    args = build_parser().parse_args(list(argv))
    options = {**options_from(args, OPTIONS), "parameters": window_parameters_from(args, PARAMETERS)}
    output_directory = Path(options["output_directory"])
    output_directory.mkdir(parents=True, exist_ok=True)
    return run_with_logging(args, default_log_path=output_directory / "lighting_check.log", logger=LOG,
                            label="Lighting check", procedure=lambda: check_safe_haven_lighting(**options))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
