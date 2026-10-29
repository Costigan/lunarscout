#!/usr/bin/env python3
"""Edit the constants below or give them on the command line, then run this script with the project environment.

    .venv/bin/python scripts/prepare_safe_haven_viewer.py [--rebuild --connect edges ...]

Every constant in the OPTIONS list below is a command-line option, --constant-name (lower case, hyphens), and the
constant is its default; `--help` lists them. prepare_safe_haven_viewer(...) takes the same values as keyword arguments.

Makes everything the pole viewer (scripts/view_safe_haven.py) draws, so that it opens at once. The viewer does the same
when it finds something missing; run this to do it ahead of time or without a display. It needs no PySide6.

* VIEWER_DIRECTORY/viewer.json: the months of the dataset, and for every month the bounding box of each cluster of
  connected havens whose chain starts in the month ([row_min, row_max, column_min, column_max, havens] in DEM pixels),
  and the yearly chain images found next to CHAINS_PATH;
* VIEWER_DIRECTORY/havens.npz: the DEM pixel of every seed, ordered by month;
* CACHE_DIRECTORY: image pyramids of the dark hillshade (about 1 GB, 30 s) and of each yearly chain image
  (safe_haven_stage_4_longest_chain_per_root_<year>.tif; about 1.2 GB each, about a minute for all years together).

A pyramid is built once and found by the next run; it is rebuilt when its source file changes. REBUILD makes the
lists again (it is also made again when the chain table, the DEM or CONNECT changed). The lists are written through
staging files and a pyramid becomes visible only when complete, so a failed run leaves nothing half built. CPU only; no
NVIDIA device is used. Importing it starts nothing.

Progress and status messages are logged: to the console, colored with rich on a terminal when
`pip install "lunarscout[console]"` is installed and plain when piped, and to a plain-text log
file in VIEWER_DIRECTORY (LOG_PATH, COLOR and LOG_LEVEL below).
"""

from __future__ import annotations

import argparse
import logging
import sys
import time as clock
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from lunarscout._numba_horizon.safe_haven_viewer_data import prepare_viewer  # noqa: E402
from lunarscout._script_cli import (  # noqa: E402
    Option, add_logging_options, add_options, defaults_from, options_from, run_with_logging,
)


# Edit these paths before running. They match view_safe_haven.py.
DEM_PATH = Path("/d/viper/maps/lola/ldem_80s_20m.clipped.tif")
STAGE_DIRECTORY = Path("/e/lunar_analyst_scenarios/70south/safe_havens")
CHAINS_PATH = STAGE_DIRECTORY / "safe_haven_stage_4_chains.csv"
EDGES_PATH = STAGE_DIRECTORY / "safe_haven_stage_3_edges.csv"     # needed only for CONNECT = "edges"
VIEWER_DIRECTORY = STAGE_DIRECTORY / "viewer"
CACHE_DIRECTORY = Path("/tmp/lunarscout_viewer")
CONNECT = "chains"              # what joins two havens in a cluster: "chains" (stage 4's chains) or "edges" (every stage-3 edge)
REBUILD = False                 # make viewer.json and havens.npz again even when they are current

# Messages go to the console (colored with rich on a terminal when "lunarscout[console]" is
# installed; plain when piped) and to a plain-text log file, which replaces `| tee`. LOG_PATH
# None writes next to the outputs. COLOR is "auto", "always" (even into a pipe) or "never".
LOG_PATH = None
COLOR = "auto"
LOG_LEVEL = "INFO"
LOG = logging.getLogger("lunarscout.scripts.prepare_viewer")


OPTIONS = (
    Option("DEM_PATH", "path", "the DEM GeoTIFF"),
    Option("CHAINS_PATH", "path", "the final stage-4 table (the yearly chain images are looked for beside it)"),
    Option("EDGES_PATH", "path", "the stage-3 edge table (used when CONNECT is edges)"),
    Option("VIEWER_DIRECTORY", "path", "where viewer.json and havens.npz are written"),
    Option("CACHE_DIRECTORY", "path", "where the image pyramids are cached"),
    Option("CONNECT", "str", "what joins two havens in a cluster: chains or edges", choices=("chains", "edges")),
    Option("REBUILD", "bool", "make viewer.json and havens.npz again even when they are current"),
)


def default_options() -> dict:
    """The options with this file's constants as their values (read now)."""
    return defaults_from(OPTIONS, globals())


def prepare_safe_haven_viewer(*, dem_path, chains_path, edges_path, viewer_directory, cache_directory, connect,
                              rebuild) -> int:
    """Make the viewer's lists and pyramids that are missing; returns 0."""
    started = clock.monotonic()
    prepare_viewer(viewer_directory, cache_directory, chains_path=chains_path, dem_path=dem_path, connect=connect,
                   edges_path=edges_path, rebuild=rebuild, progress=LOG.info)
    LOG.info("Viewer data in %s and %s ready in %.1f s", viewer_directory, cache_directory, clock.monotonic() - started)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Make the lists and image pyramids behind the safe-haven pole viewer. Every option's default is "
                    "the constant of the same name at the top of this file.")
    add_options(parser, OPTIONS, default_options())
    add_logging_options(parser, globals())
    return parser


def main(argv=()) -> int:
    """Read the options in ``argv`` (none: the defaults), set up logging and prepare the viewer's data."""
    args = build_parser().parse_args(list(argv))
    options = options_from(args, OPTIONS)
    viewer_directory = Path(options["viewer_directory"])
    viewer_directory.mkdir(parents=True, exist_ok=True)
    return run_with_logging(args, default_log_path=viewer_directory / "prepare_viewer.log", logger=LOG,
                            label="Viewer data", procedure=lambda: prepare_safe_haven_viewer(**options))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
