#!/usr/bin/env python3
"""Edit the constants below, then run this script with the project environment.

    .venv/bin/python scripts/run_lander_continuous_sun.py

Lander sites: the longest continuous Sun (src/lunarscout/_numba_horizon/lander_sun_duration.py). A lander that carries no
rover stays where it lands, so what matters at a pixel is how long the Sun can stay up without a break
after it lands. For every pixel and calendar month this writes the longest continuous Sun from a landing
in that month ("land at the best moment of the month and the Sun stays up for N hours"), as a GeoTIFF
with one band per month. It reads only the light store
(scripts/run_light_store.py); it does not depend on the rover safe-haven stages. The step that
follows (clustering the pixel values into candidate regions) reads this product.

* Sun is "Sun fraction at least THRESHOLD". The store holds two thresholds, 0.2 and 1.0, so THRESHOLD must be
  one of them (1.0: the Sun fully visible); another needs the store lit again.
* A period is consecutive 2-hour steps in Sun; one step out of Sun ends it. Landing at a step in Sun gives the
  rest of the period from that step, and a UTC month's value is the longest such remainder over its steps. A
  period that starts in the month counts in full; one already under way at the month's first step counts from
  that step, so a long period decays by one month per band. A month with no step in Sun is 0.
* A period under way at the first step of the range counts from that step. A period still under way at the
  last step is credited with the length seen (a lower bound), so the last months read low.
* Each band holds a uint16 count of steps (hours = value x 2; 65534 saturates). 65535 (nodata) marks pixels
  with no valid terrain: an invalid DEM node, or a patch with no store file.

The run is restartable: run it again after a kill, a crash or Ctrl-C and it does only the patches not yet
written; a different THRESHOLD or range needs a new OUTPUT_PATH or START_FRESH. A first Ctrl-C lets the
patches in flight finish. It uses the CPU only (no GPU): about 1 s of one core per patch, WORKERS patches at
a time, and about 45 GB of output (the first full run: 45 GB in about 1 hour with 16 workers). Importing this file starts nothing.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from lunarscout._numba_horizon.lander_sun_duration import SunDurationSettings, generate_sun_duration  # noqa: E402
from lunarscout._numba_horizon.safe_haven_stage0 import LightStoreDirectory  # noqa: E402
from lunarscout._script_logging import reset_script_logging, setup_script_logging  # noqa: E402
from lunarscout.products import _load_dem  # noqa: E402

# Edit these paths before running. They match run_light_store.py.
DEM_PATH = Path("/d/viper/maps/lola/ldem_80s_20m.clipped.tif")
LIGHT_STORE_PATH = Path("/e/lunar_analyst_scenarios/70south/light_store")
OUTPUT_PATH = Path("/e/lunar_analyst_scenarios/70south/lander_sun_longest_period.tif")      # about 45 GB

THRESHOLD = 1.0             # Sun fraction at or above this is Sun: 1.0 or 0.2 (what the store holds)
START_UTC = None            # None: the store's first step (2024-01-01); else a time on the 2-hour grid
STOP_UTC = None             # exclusive; None: the end of the store (2032-01-01)
WORKERS = 16                # patches reduced at once, one thread each
CHECKPOINT_PATCHES = 32     # patches written between journal checkpoints
STATUS_SECONDS = 60.0
OVERWRITE = True           # replace the existing product when the new one is complete (the old one stays until then)
START_FRESH = False         # discard a partly written product instead of resuming it

LOG_PATH = None             # None writes next to OUTPUT_PATH
COLOR = "auto"
LOG_LEVEL = "INFO"
LOG = logging.getLogger("lunarscout.scripts.lander_continuous_sun")


def _run() -> int:
    dem, georef = _load_dem(DEM_PATH)
    report = generate_sun_duration(
        OUTPUT_PATH, directory=LightStoreDirectory(LIGHT_STORE_PATH, dem), dem=dem, georef=georef,
        settings=SunDurationSettings(
            threshold=THRESHOLD, start_utc=START_UTC, stop_utc=STOP_UTC, workers=WORKERS,
            checkpoint_patches=CHECKPOINT_PATCHES, status_seconds=STATUS_SECONDS),
        overwrite=OVERWRITE, start_fresh=START_FRESH)
    LOG.info("Lander sun duration %s: %d patches written, %d already done, %d without a store file, %d bands, %.0f s",
             report.status, report.patches_written, report.patches_already_done, report.patches_without_store_file,
             report.bands, report.elapsed_seconds)
    if report.status != "completed":
        LOG.warning("The run is %s; run this script again to continue.", report.status)
        return 1
    LOG.info("Product written to: %s", OUTPUT_PATH)
    return 0


def main() -> int:
    log_path = LOG_PATH if LOG_PATH is not None else OUTPUT_PATH.with_name(OUTPUT_PATH.name + ".log")
    setup_script_logging(log_path, color=COLOR, level=LOG_LEVEL)
    try:
        return _run()
    except Exception as error:
        LOG.error("Lander continuous Sun stopped: %s: %s", type(error).__name__, error)
        raise
    finally:
        reset_script_logging()


if __name__ == "__main__":
    raise SystemExit(main())
