#!/usr/bin/env python3
"""Level-0 stepping variants on the hierarchy safety matrix (CPU only, no GPU).

Builds four variants of the CPU hierarchy oracle (traverse_hierarchy) from the repo source, in memory:
  current            production stepping: no stop at level-0 cell exits (d543e6e)
  stall-safe clamp   each level-0 step stops at the cell exit, nudged past it by
                     max(1e-6 km, 4 float32 ulps of s) so the ray always advances
  quarter-pixel      quarter-pixel level-0 steps at every range, no 0.8 / 0.5 pixel floor
  clamp + quarter    both
and runs scripts/numba_horizon/validate_phase4_hierarchy_safety.run_matrix plus the C#-derived
production fixture against lunarscout's dense fixed-step reference. Then prints the level-0 samples
around the obstacle node for the worst case (level0_boundary_southeast), with and without the clamp.

    .venv/bin/python review_scripts/s7_stepping_matrix.py
Used for docs/numba-horizon-terrain-interpolation.md (2026-10-09).
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
for path in (REPO, REPO / "src", REPO / "tests" / "numba_horizon"):
    sys.path.insert(0, str(path))
import scripts.numba_horizon.validate_phase4_hierarchy_safety as vm  # noqa: E402
import test_phase4_pyramid as tp  # noqa: E402
from lunarscout._numba_horizon.fixed_step import traverse_level0_fixed_step  # noqa: E402

SOURCE = (REPO / "src/lunarscout/_numba_horizon/hierarchy.py").read_text()
MARK = """                # Like the production CUDA kernel, advance by the adaptive step
                # without stopping at the level-0 cell exit (d543e6e).
"""
CLAMP = """                nudge = max(np.float32(0.000001), np.float32(4.0) * np.spacing(np.float32(s)))
                boundary_advance = (
                    distance_to_exit + nudge if distance_to_exit > 0 else fallback
                )
                advance = min(advance, boundary_advance)
"""
QUARTER_OLD = """                if s < np.float32(0.5):
                    advance *= np.float32(0.25)
"""
QUARTER_NEW = """                advance *= np.float32(0.25)
"""
FLOOR_OLD = "                advance = max(advance, floor)\n"
FLOOR_NEW = "                advance = max(advance, np.float32(1e-6))\n"


def variant(name: str, clamp: bool, quarter: bool):
    source = SOURCE
    for text in (MARK, QUARTER_OLD, FLOOR_OLD):
        if source.count(text) != 1:
            sys.exit("hierarchy.py no longer matches this script; update the patterns")
    if clamp:
        source = source.replace(MARK, CLAMP)
    if quarter:
        source = source.replace(QUARTER_OLD, QUARTER_NEW).replace(FLOOR_OLD, FLOOR_NEW)
    module = types.ModuleType(f"lunarscout._numba_horizon._s7_{name}")
    module.__package__ = "lunarscout._numba_horizon"
    sys.modules[module.__name__] = module
    exec(compile(source, f"hierarchy_{name}.py", "exec"), module.__dict__)
    return module.traverse_hierarchy


VARIANTS = {
    "current": variant("current", False, False),
    "stall-safe clamp": variant("clamp", True, False),
    "quarter-pixel": variant("quarter", False, True),
    "clamp + quarter": variant("both", True, True),
}


def main() -> None:
    _, _, segment, pyramid, observer_z, radius = tp._production_hierarchy_inputs()
    dense = traverse_level0_fixed_step(
        segment, pyramid.level0, observer_z_m=observer_z, radius_m=radius, map_resolution_m=30.0
    )
    dense_degrees = float(np.degrees(np.arctan(dense.maximum_slope)))
    rows = {}
    for name, traverse in VARIANTS.items():
        vm.traverse_hierarchy = traverse
        report = vm.run_matrix()
        gaps = {case["id"]: case["dense_minus_hierarchy_degrees"] for case in report["cases"]}
        result = traverse(segment, pyramid, observer_z_m=observer_z, radius_m=radius,
                          map_resolution_m=30.0, pass_index=1)
        gaps["csharp_production_fixture"] = dense_degrees - float(np.degrees(np.arctan(result.maximum_slope)))
        rows[name] = (gaps, len(result.values))
    ids = list(next(iter(rows.values()))[0])
    print(f"{'case (dense - hierarchy, deg)':34s}" + "".join(f"{name:>18s}" for name in rows))
    for case in ids:
        print(f"{case:34s}" + "".join(f"{rows[name][0][case]:>18.4f}" for name in rows))
    print(f"{'max |gap|':34s}" + "".join(f"{max(abs(v) for v in rows[n][0].values()):>18.4f}" for n in rows))
    print(f"{'steps, production fixture ray':34s}" + "".join(f"{rows[n][1]:>18d}" for n in rows))
    print(f"dense reference for the production fixture: {dense_degrees:.4f} deg")

    size, center = 129, 64
    elevation = np.zeros((size, size), np.float32)
    elevation[center + 20, center + 20] = 150.0
    segment = vm._straight_segment(center + 0.25, center + 0.25, 1.0, 1.0, 1.7)
    pyramid = vm._pyramid(elevation)
    for name in ("current", "stall-safe clamp"):
        trace = VARIANTS[name](segment, pyramid, observer_z_m=0.0, radius_m=vm.RADIUS_M,
                               map_resolution_m=vm.MAP_RESOLUTION_M, pass_index=0)
        near = trace.values[(trace.values[:, 11] == 4.0) & (np.abs(trace.values[:, 5] - 84) < 1.6)]
        print(f"\nlevel0_boundary_southeast, {name}: level-0 samples near the node (84,84); "
              f"horizon {np.degrees(np.arctan(trace.maximum_slope)):.2f} deg")
        for row in near:
            print(f"  pixel ({row[5]:7.3f}, {row[6]:7.3f})  height {row[8]:7.2f} m  "
                  f"distance {row[1]:6.1f} m  step {row[10] * 1000:5.1f} m")


if __name__ == "__main__":
    main()
