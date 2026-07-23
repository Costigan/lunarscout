"""Explore canonical validity, where, coalesce, and valid zero values.

Inputs: none; all values and lunar grid metadata are deterministic.
Outputs: printed values and validity masks.
Resources: tiny in-memory CPU arrays; no files, SPICE kernels, or GPU.
"""

from __future__ import annotations

import lunarscout as ls
import numpy as np

from _example_support import example_parser, synthetic_georef


ma = ls.map_algebra


def show(label: str, raster: ls.Raster) -> None:
    print(f"\n{label} ({raster.units or 'dimensionless'})")
    print("values:")
    print(raster.values)
    print("valid:")
    print(raster.valid)


def main() -> None:
    example_parser(__doc__).parse_args()
    grid = synthetic_georef(width=4, height=2, nodata=None)

    # Several invalid payloads look scientifically plausible. Payload values
    # alone therefore cannot determine whether a cell is valid.
    illumination = ma.raster(
        np.array(
            [[0.0, 0.7, 0.8, 0.2], [0.6, 0.0, 0.9, 0.4]],
            dtype=np.float32,
        ),
        grid,
        valid=np.array(
            [[True, True, False, True], [True, False, True, True]],
            dtype=np.bool_,
        ),
        units="fraction",
        name="illumination",
    )
    earth_visibility = ma.raster(
        np.array(
            [[0.9, 0.0, 0.8, 0.4], [0.7, 0.9, 0.0, 0.6]],
            dtype=np.float32,
        ),
        grid,
        valid=np.array(
            [[True, True, True, False], [True, True, True, True]],
            dtype=np.bool_,
        ),
        units="fraction",
        name="Earth visibility",
    )
    show("Illumination (notice the valid zero at [0, 0])", illumination)
    show("Earth visibility", earth_visibility)

    # Ordinary arithmetic uses strict validity intersection.
    show("Strict intersection from addition", illumination + earth_visibility)

    earth_visible = earth_visibility >= 0.5
    selected = ma.where(earth_visible, illumination, ma.invalid)
    show("where: retain illumination only where Earth is visible", selected)

    fallback = ma.raster(
        np.full((2, 4), 0.25, dtype=np.float32),
        grid,
        units="fraction",
        name="fallback estimate",
    )
    show("coalesce: first valid illumination, then fallback", ma.coalesce(illumination, fallback))
    show("coalesce in reverse order", ma.coalesce(fallback, illumination))

    too_dim = illumination < 0.5
    show("set_invalid: reject valid cells below 0.5", ma.set_invalid(illumination, too_dim))
    filled = ma.fill_invalid(illumination, 0.0)
    show("fill_invalid: invalid cells become valid zero values", filled)

    print("\nMasked-array interchange view (mask=True means invalid):")
    print(illumination.masked())
    print(
        "\nCanonical validity is separate from payload. A zero may be valid, "
        "and a plausible nonzero payload may be invalid."
    )


if __name__ == "__main__":
    main()
