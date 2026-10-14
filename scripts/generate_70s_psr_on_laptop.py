import lunarscout as ls
from pathlib import Path

DEM = "/d/viper/maps/lola/ldem_80s_20m.clipped.tif"
HORIZONS = "/e/lunar_analyst_scenarios/70south/lighting/horizons"
OUTPUT = "/e/lunar_analyst_scenarios/70south/psr.tif"

output = Path(OUTPUT)
output.parent.mkdir(parents=True, exist_ok=True)

times = ls.times(
    "1970-01-01T00:00:00Z",
    "2044-01-01T00:00:00Z",
    step_hours=6,
)

psr = ls.generate_psr(
    DEM,
    HORIZONS,
    OUTPUT,
    times=times,
    observer_height_m=0.0,
    backend="auto",
    verbose=True,
)

print(psr)
