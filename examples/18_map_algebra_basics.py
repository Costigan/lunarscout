"""Build eager rasters and apply local map algebra to small lunar arrays.

Inputs: none; all values and lunar grid metadata are deterministic.
Outputs: printed values, validity, metadata, and eager operation results.
Resources: tiny in-memory CPU arrays; no files, SPICE kernels, or GPU.
"""

from __future__ import annotations

import lunarscout as ls
import numpy as np

from _example_support import example_parser, synthetic_georef


ma = ls.map_algebra


def show(label: str, raster: ls.Raster) -> None:
    """Print the pieces that make a Raster different from a bare array."""
    print(f"\n{label}")
    print(
        f"  shape={raster.shape}, dtype={raster.dtype}, units={raster.units!r}, "
        f"valid={int(raster.valid.sum())}/{raster.valid.size}, "
        f"memory={raster.nbytes} bytes"
    )
    print("  values:")
    print(raster.values)
    print("  valid:")
    print(raster.valid)


def main() -> None:
    example_parser(__doc__).parse_args()
    grid = synthetic_georef(width=4, height=3, nodata=None)
    values = np.array(
        [
            [1.0, 4.0, 9.0, 16.0],
            [2.0, 6.0, 10.0, 14.0],
            [0.0, 3.0, 8.0, 12.0],
        ],
        dtype=np.float32,
    )
    valid = np.array(
        [
            [True, True, True, True],
            [True, False, True, True],
            [True, True, True, False],
        ],
        dtype=np.bool_,
    )
    slope = ma.raster(
        values,
        grid,
        valid=valid,
        units="degrees",
        name="synthetic slope",
    )
    show("Input slope", slope)
    print(f"  affine transform={slope.georef.affine_transform}")

    # Raster helpers return new objects. They never modify slope.
    renamed = slope.with_name("renamed slope")
    unitless = slope.with_units(None)
    more_valid = slope.with_validity(np.ones(slope.shape, dtype=np.bool_))
    copied = slope.copy()
    readonly = slope.readonly()
    print(
        "\nNon-mutating helpers:",
        renamed.name,
        unitless.units,
        f"with_validity={int(more_valid.valid.sum())}",
        f"copy_is_distinct={copied is not slope}",
        f"readonly={not readonly.values.flags.writeable}",
    )
    print("Filled array (for interchange only):")
    print(slope.filled(-9999.0))
    print("NumPy masked view:")
    print(slope.masked())

    show("Add a scalar threshold offset", slope + 1.0)
    show("Cell-wise minimum with 8 degrees", ma.minimum(slope, 8.0))
    show("Clipped to [2, 10] degrees", ma.clip(slope, lower=2.0, upper=10.0))
    show("Square root (domain-sensitive operation)", ma.sqrt(slope))

    gentle = slope <= 8.0
    nonzero = slope > 0.0
    candidate = gentle & nonzero
    show("Gentle AND nonzero", candidate)
    show("NOT candidate", ~candidate)
    show("Explicit valid cells", ma.is_valid(slope))
    show("Explicit invalid cells", ma.is_invalid(slope))

    print(
        "\nUse &, |, and ~ for raster Boolean algebra. Python 'and' and 'or' "
        "perform scalar truth tests and are intentionally rejected."
    )
    print("Use ordinary NumPy operations for arrays that have no spatial grid.")


if __name__ == "__main__":
    main()
