"""Generate a native permanent-shadow byte mask.

Inputs: --scenario with dem.tif, horizons, SPICE, and native runtime.
Outputs: analysis/native_psr.tif in the supplied scenario.
Semantics: 255 is permanent shadow; 0 receives direct Sun in the native period.
"""

from __future__ import annotations

import lunarscout as ls
import numpy as np
import rasterio

from _example_support import example_parser, require_native_scenario


def report(progress: ls.native.NativeProductProgress) -> None:
    print(f"{progress.percent:6.2f}% [{progress.stage}] {progress.message}")


def main() -> None:
    args = example_parser(__doc__, native=True).parse_args()
    scenario = require_native_scenario(args.scenario)
    output = scenario.psr(
        "analysis/native_psr.tif",
        overwrite=args.overwrite,
        progress_callback=report,
    )
    values, _georef = ls.read_geotiff(output)
    with rasterio.open(output) as dataset:
        validity = dataset.read_masks(1) != 0
    print(
        f"output={output}, psr_pixels={np.count_nonzero(validity & (values == 255))}, "
        f"sunlit_pixels={np.count_nonzero(validity & (values == 0))}, "
        f"unknown_pixels={np.count_nonzero(~validity)}"
    )


if __name__ == "__main__":
    main()
