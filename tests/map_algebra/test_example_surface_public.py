"""Fresh-process contracts for the public map-algebra example surface."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest


def _run_fresh_process(tmp_path: Path, code: str) -> None:
    environment = os.environ.copy()
    result = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path)],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert result.returncode == 0, (
        f"fresh-process contract failed\nstdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )


@pytest.mark.parametrize(
    ("script_name", "expected_output"),
    [
        ("18_map_algebra_basics.py", "Gentle AND nonzero"),
        ("19_map_algebra_validity.py", "Canonical validity is separate"),
        ("20_map_algebra_grids.py", "Same grid after explicit alignment: True"),
        ("21_map_algebra_numerics.py", "overflow='promote' chooses a wider dtype"),
    ],
)
def test_introductory_examples_run_as_fresh_public_programs(
    tmp_path,
    script_name,
    expected_output,
):
    repository = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [
            sys.executable,
            str(repository / "examples" / script_name),
            "--workspace",
            str(tmp_path),
        ],
        check=False,
        capture_output=True,
        text=True,
        env=os.environ.copy(),
        cwd=repository,
    )
    assert result.returncode == 0, result.stderr
    assert expected_output in result.stdout


_GRID_HELPER = r'''
MOON_WKT = (
    'PROJCS["ESRI:103878",'
    'GEOGCS["Moon_2000",DATUM["D_Moon_2000",'
    'SPHEROID["Moon_2000_IAU_IAG",1737400,0]],'
    'PRIMEM["Reference_Meridian",0],UNIT["Degree",0.0174532925199433]],'
    'PROJECTION["Polar_Stereographic"],'
    'PARAMETER["latitude_of_origin",-90],'
    'PARAMETER["central_meridian",0],'
    'PARAMETER["scale_factor",1],'
    'PARAMETER["false_easting",0],'
    'PARAMETER["false_northing",0],UNIT["Meter",1]]'
)
MOON_PROJ4 = (
    "+proj=stere +lat_0=-90 +lon_0=0 +k=1 +x_0=0 +y_0=0 "
    "+R=1737400 +units=m +no_defs"
)

def grid(*, origin_x=0.0):
    return ls.GeoReference(
        projection_wkt=MOON_WKT,
        projection_proj4=MOON_PROJ4,
        affine_transform=(origin_x, 10.0, 0.0, 20.0, 0.0, -10.0),
        width=3,
        height=2,
        pixel_size_x=10.0,
        pixel_size_y=-10.0,
        nodata=None,
    )
'''


def test_fresh_public_geotiff_preserves_valid_zero_and_qgis_mask(tmp_path):
    _run_fresh_process(
        tmp_path,
        r'''
from pathlib import Path
import sys
import numpy as np
import rasterio
import lunarscout as ls

ma = ls.map_algebra
root = Path(sys.argv[1])
''' + _GRID_HELPER + r'''
values = np.array(
    [[False, True, False], [True, False, True]],
    dtype=np.bool_,
)
valid = np.array(
    [[True, True, False], [True, True, True]],
    dtype=np.bool_,
)
raster = ma.raster(values, grid(), valid=valid, name="candidate")
output = ma.write(
    root / "candidate.tif",
    raster.expression(),
    invalid_value=0,
)

with rasterio.open(output) as dataset:
    stored = dataset.read(1)
    stored_valid = dataset.read_masks(1) > 0
    assert dataset.dtypes == ("uint8",)
    assert stored[0, 0] == 0 and stored_valid[0, 0]
    assert stored[0, 2] == 0 and not stored_valid[0, 2]
    np.testing.assert_array_equal(stored_valid, valid)

roundtrip = ma.read(output)
np.testing.assert_array_equal(roundtrip.valid, valid)
assert roundtrip.values[0, 0] == 0
assert roundtrip.valid[0, 0]
assert not roundtrip.valid[0, 2]
''',
    )


def test_fresh_public_grid_and_unit_mismatches_are_structured(tmp_path):
    _run_fresh_process(
        tmp_path,
        r'''
import sys
import numpy as np
import lunarscout as ls

ma = ls.map_algebra
''' + _GRID_HELPER + r'''
values = np.arange(6, dtype=np.float32).reshape(2, 3)
reference = ma.raster(values, grid(), units="metres")
shifted = ma.raster(values, grid(origin_x=5.0), units="metres")

try:
    ma.add(reference, shifted)
except ls.MapAlgebraGridError as error:
    assert error.code == "map_algebra_grid_mismatch"
    assert error.details
else:
    raise AssertionError("shifted grids were accepted")

degrees = ma.raster(values, grid(), units="degrees")
try:
    ma.add(reference, degrees)
except ls.MapAlgebraUnitError as error:
    assert error.code == "map_algebra_unit_mismatch"
    assert error.details["left_units"] == "metres"
    assert error.details["right_units"] == "degrees"
else:
    raise AssertionError("incompatible units were accepted")
''',
    )


def test_fresh_public_integer_nonfinite_and_all_invalid_contracts(tmp_path):
    _run_fresh_process(
        tmp_path,
        r'''
import sys
import numpy as np
import lunarscout as ls

ma = ls.map_algebra
''' + _GRID_HELPER + r'''
maximum = np.iinfo(np.uint64).max
integers = ma.raster(
    np.array([[maximum - 1, 2, 3], [4, 5, 6]], dtype=np.uint64),
    grid(),
)
eager_integer = ma.add(integers, np.uint64(1))
expression_integer = ma.compute(ma.add(integers.expression(), np.uint64(1)))
assert eager_integer.dtype == expression_integer.dtype == np.dtype(np.uint64)
assert eager_integer.values[0, 0] == expression_integer.values[0, 0] == maximum

try:
    ma.add(
        ma.raster(
            np.array([[maximum, 0, 0], [0, 0, 0]], dtype=np.uint64),
            grid(),
        ),
        np.uint64(1),
    )
except ls.MapAlgebraDTypeError as error:
    assert error.code == "map_algebra_overflow"
else:
    raise AssertionError("uint64 overflow was accepted")

floating = ma.raster(
    np.array([[4.0, -1.0, np.inf], [0.0, 9.0, np.nan]], dtype=np.float32),
    grid(),
)
eager_sqrt = ma.sqrt(floating, numeric_errors="invalid")
expression_sqrt = ma.compute(
    ma.sqrt(floating.expression(), numeric_errors="invalid"),
)
np.testing.assert_array_equal(eager_sqrt.valid, expression_sqrt.valid)
np.testing.assert_array_equal(
    eager_sqrt.valid,
    np.array([[True, False, False], [True, True, False]], dtype=np.bool_),
)
np.testing.assert_array_equal(
    eager_sqrt.values[eager_sqrt.valid],
    expression_sqrt.values[expression_sqrt.valid],
)

all_invalid = ma.raster(
    np.full((2, 3), maximum, dtype=np.uint64),
    grid(),
    valid=np.zeros((2, 3), dtype=np.bool_),
)
ignored_payload = ma.add(all_invalid, np.uint64(1), overflow="raise")
ignored_expression = ma.compute(
    ma.add(all_invalid.expression(), np.uint64(1), overflow="raise"),
)
assert not ignored_payload.valid.any()
np.testing.assert_array_equal(ignored_payload.valid, ignored_expression.valid)
''',
    )


def test_fresh_public_eager_expression_parity_for_introductory_operations(tmp_path):
    _run_fresh_process(
        tmp_path,
        r'''
import sys
import numpy as np
import lunarscout as ls

ma = ls.map_algebra
''' + _GRID_HELPER + r'''
valid = np.array(
    [[True, True, False], [True, False, True]],
    dtype=np.bool_,
)
slope = ma.raster(
    np.array([[2.0, 9.0, 4.0], [7.0, 6.0, 10.0]], dtype=np.float32),
    grid(),
    valid=valid,
    units="degrees",
)
fallback = ma.raster(
    np.array([[5.0, 5.0, 5.0], [5.0, 5.0, 5.0]], dtype=np.float32),
    grid(),
    units="degrees",
)

eager_candidate = (slope <= 8.0) & ma.is_valid(slope)
expression_candidate = (
    (slope.expression() <= 8.0) & ma.is_valid(slope.expression())
)
computed_candidate = ma.compute(expression_candidate)
np.testing.assert_array_equal(eager_candidate.values, computed_candidate.values)
np.testing.assert_array_equal(eager_candidate.valid, computed_candidate.valid)
assert eager_candidate.units is computed_candidate.units is None

eager_selected = ma.where(
    eager_candidate,
    ma.clip(slope, lower=0.0, upper=8.0),
    ma.invalid,
)
expression_selected = ma.where(
    expression_candidate,
    ma.clip(slope.expression(), lower=0.0, upper=8.0),
    ma.invalid,
)
computed_selected = ma.compute(expression_selected)
np.testing.assert_array_equal(eager_selected.values, computed_selected.values)
np.testing.assert_array_equal(eager_selected.valid, computed_selected.valid)

eager_filled = ma.coalesce(eager_selected, fallback)
computed_filled = ma.compute(
    ma.coalesce(expression_selected, fallback.expression()),
)
np.testing.assert_array_equal(eager_filled.values, computed_filled.values)
np.testing.assert_array_equal(eager_filled.valid, computed_filled.valid)
assert eager_filled.units == computed_filled.units == "degrees"
''',
    )


def test_fresh_public_review_is_read_only_and_rejects_unsupported_plan(tmp_path):
    _run_fresh_process(
        tmp_path,
        r'''
from pathlib import Path
import json
import sys
import numpy as np
import lunarscout as ls

ma = ls.map_algebra
root = Path(sys.argv[1])
''' + _GRID_HELPER + r'''
source_path = root / "slope.tif"
ls.write_geotiff(
    source_path,
    np.arange(6, dtype=np.float32).reshape(2, 3),
    grid(),
)
source_stat = source_path.stat()
candidate_output = root / "planned" / "candidate.tif"
expression = ma.source(source_path, units="degrees") <= 8.0

explanation = ma.explain(expression)
plan = ma.plan(expression, output=candidate_output)
json.dumps(plan)
assert "local.less_equal" in explanation
assert plan["validated"] is True
assert plan["output_preflight"]["created_or_modified"] is False
assert not candidate_output.exists()
assert not candidate_output.parent.exists()
assert source_path.stat() == source_stat

unsupported_output = root / "unsupported" / "mean.tif"
unsupported = ma.focal_mean(ma.source(source_path, units="degrees"), size=3)
try:
    ma.plan(unsupported, output=unsupported_output)
except ls.MapAlgebraExpressionError as error:
    assert error.code == "map_algebra_unsupported_windowed_operation"
    assert error.details["operation_id"] == "focal.mean"
else:
    raise AssertionError("unsupported file-backed focal operation was planned")
assert not unsupported_output.exists()
assert not unsupported_output.parent.exists()
assert source_path.stat() == source_stat
''',
    )
