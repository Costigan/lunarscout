"""Make units, dtypes, overflow, casting, and numeric-error policy explicit.

Inputs: none; all values and lunar grid metadata are deterministic.
Outputs: printed results and structured error codes.
Resources: tiny in-memory CPU arrays; no files, SPICE kernels, or GPU.
"""

from __future__ import annotations

import lunarscout as ls
import numpy as np

from _example_support import example_parser, synthetic_georef


ma = ls.map_algebra


def show(label: str, raster: ls.Raster) -> None:
    print(
        f"\n{label}: dtype={raster.dtype}, units={raster.units!r}, "
        f"valid={int(raster.valid.sum())}/{raster.valid.size}"
    )
    print(raster.values)
    if not raster.all_valid:
        print("valid:")
        print(raster.valid)


def main() -> None:
    example_parser(__doc__).parse_args()
    grid = synthetic_georef(width=4, height=2, nodata=None)

    elevation = ma.raster(
        np.array([[100, 102, 104, 106], [108, 110, 112, 114]], dtype=np.float32),
        grid,
        units="metres",
        name="elevation",
    )
    correction = ma.raster(
        np.full((2, 4), 0.5, dtype=np.float32),
        grid,
        units="metres",
        name="vertical correction",
    )
    slope = ma.raster(
        np.array([[0, 5, 10, 15], [20, 25, 30, 35]], dtype=np.float32),
        grid,
        units="degrees",
        name="slope",
    )
    show("Matching-unit addition", elevation + correction)
    show("Scalar threshold interpreted in slope units", slope <= 15.0)
    show(
        "Explicit derived units for multiplication",
        ma.multiply(elevation, elevation, output_units="square metres"),
    )
    show("Sine accepts explicit degree units", ma.sin(slope))

    try:
        ma.add(elevation, slope)
    except ls.MapAlgebraUnitError as error:
        print(f"\nIncompatible units rejected: {error.code}")

    integers = ma.raster(
        np.array([[120, 1, -5, 10], [100, -100, 0, 5]], dtype=np.int8),
        grid,
        name="integer samples",
    )
    try:
        ma.add(integers, 10, overflow="raise")
    except ls.MapAlgebraDTypeError as error:
        print(f"Checked int8 overflow rejected: {error.code}")
    show("int8 overflow='wrap'", ma.add(integers, 10, overflow="wrap"))
    show("overflow='promote' chooses a wider dtype", ma.add(integers, 10, overflow="promote"))

    decimals = ma.raster(
        np.array([[1.2, 2.8, 3.0, 4.5], [5.1, 6.9, 7.0, 8.4]], dtype=np.float32),
        grid,
    )
    try:
        ma.cast(decimals, "int16", casting="safe")
    except ls.MapAlgebraDTypeError as error:
        print(f"Safe float-to-int cast rejected: {error.code}")
    show("Explicit unsafe float-to-int cast", ma.cast(decimals, "int16", casting="unsafe"))

    log_input = ma.raster(
        np.array([[1.0, 0.0, -1.0, 10.0], [2.0, 3.0, 4.0, 5.0]], dtype=np.float32),
        grid,
    )
    show("log numeric_errors='invalid'", ma.log(log_input, numeric_errors="invalid"))
    show("log numeric_errors='keep'", ma.log(log_input, numeric_errors="keep"))
    try:
        ma.log(log_input, numeric_errors="raise")
    except ls.MapAlgebraOperationError as error:
        print(f"log numeric_errors='raise' rejected the domain errors: {error.code}")

    print(
        "\nThese policies are part of the public calculation contract; choose "
        "them explicitly when a silent choice could change an analysis."
    )


if __name__ == "__main__":
    main()
