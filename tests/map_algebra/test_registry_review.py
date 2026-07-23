from __future__ import annotations

import inspect
import json
import subprocess
import sys

import numpy as np
import pytest

import lunarscout as ls
from lunarscout.errors import GeoTiffOpenError, MapAlgebraError
from lunarscout.map_algebra import describe_operation, list_operations
from lunarscout.map_algebra._registry import (
    _OPERAND_PARAMETER_NAMES,
    OperationSpec,
    _public_function_name,
)
from tests.map_algebra.conftest import MOON_PROJ4, MOON_WKT


ma = ls.map_algebra


def _grid(height: int = 2, width: int = 3) -> ls.GeoReference:
    return ls.GeoReference(
        projection_wkt=MOON_WKT,
        projection_proj4=MOON_PROJ4,
        affine_transform=(1000.0, 20.0, 0.0, 2000.0, 0.0, -20.0),
        width=width,
        height=height,
        pixel_size_x=20.0,
        pixel_size_y=-20.0,
        nodata=None,
    )


def _write_source(path, values: np.ndarray) -> None:
    ls.write_geotiff(path, values, _grid(*values.shape))


@pytest.mark.parametrize("version", [0, -1, True, 1.5])
def test_operation_spec_rejects_invalid_versions(version):
    with pytest.raises(ValueError):
        OperationSpec("test.operation", version, 1, "test", "Test operation.")


def test_registry_builder_rejects_duplicate_ids(monkeypatch):
    from lunarscout.map_algebra import _registry

    duplicate = OperationSpec(
        "test.duplicate", 1, 1, "test", "Duplicate test operation.",
    )
    monkeypatch.setattr(_registry, "_SPECS", (duplicate, duplicate))
    with pytest.raises(RuntimeError, match="Duplicate built-in operation id"):
        _registry._build_registry()


def test_public_registry_parameters_follow_callable_signatures():
    for description in list_operations():
        operation_id = description["id"]
        function_name = _public_function_name(operation_id)
        if function_name is None or not hasattr(ma, function_name):
            continue
        function = getattr(ma, function_name)
        signature = inspect.signature(function)
        ignored = _OPERAND_PARAMETER_NAMES.get(operation_id)
        if ignored is None:
            positional = [
                parameter.name
                for parameter in signature.parameters.values()
                if parameter.kind in {
                    inspect.Parameter.POSITIONAL_ONLY,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    inspect.Parameter.VAR_POSITIONAL,
                }
            ]
            arity = description["arity"]
            ignored = frozenset(positional[: arity or len(positional)])
        expected = [
            parameter.name
            for parameter in signature.parameters.values()
            if parameter.name not in ignored
            and parameter.kind is not inspect.Parameter.VAR_POSITIONAL
        ]
        actual = [parameter["name"] for parameter in description["parameters"]]
        assert actual == expected, operation_id


def test_registry_descriptions_are_complete_and_json_serializable():
    descriptions = list_operations()
    json.dumps(descriptions)
    for description in descriptions:
        assert description["output_dtype_rule"] != "operation-specific"
        assert description["output_units_rule"] != "operation-specific"
        assert description["validity_rule"] != "operation-specific"
        assert description["execution_modes"]
        for parameter in description["parameters"]:
            assert parameter["type"]
            assert isinstance(parameter["required"], bool)


def test_registry_reports_distinct_execution_modes():
    assert describe_operation("source")["execution_modes"] == [
        "expression_compute", "windowed_write",
    ]
    assert describe_operation("focal.mean")["execution_modes"] == [
        "eager", "expression_compute",
    ]
    assert describe_operation("local.sum_layers")["execution_modes"] == [
        "eager", "expression_compute", "composed_windowed_write",
    ]
    assert describe_operation("temporal.mean")["execution_modes"] == [
        "eager", "expression_compute", "temporal_streaming",
    ]


@pytest.mark.parametrize(
    "execution_mode",
    [
        "eager", "expression_compute", "windowed_write",
        "composed_windowed_write", "temporal_streaming",
    ],
)
def test_registry_execution_mode_filters_match_operation_claims(execution_mode):
    descriptions = list_operations(execution_mode=execution_mode)
    assert descriptions
    assert all(
        execution_mode in description["execution_modes"]
        for description in descriptions
    )


def test_legacy_file_backed_filter_aliases_windowed_write():
    assert list_operations(execution_mode="file_backed") == list_operations(
        execution_mode="windowed_write",
    )


def test_registry_exposes_scientific_choices_and_defaults():
    focal = {parameter["name"]: parameter for parameter in describe_operation("focal.mean")["parameters"]}
    assert focal["edge"]["default"] == "invalid"
    assert focal["edge"]["choices"] == ["invalid", "constant", "nearest", "reflect", "wrap"]
    assert focal["valid_neighbor"]["default"] == "require_all"

    distance = {parameter["name"]: parameter for parameter in describe_operation("distance.to")["parameters"]}
    assert distance["metric"]["choices"] == ["euclidean", "taxicab", "chessboard"]
    assert distance["units"]["choices"] == ["pixels", "physical"]
    assert distance["invalid_output"]["choices"] == ["preserve", "compute"]

    region = {parameter["name"]: parameter for parameter in describe_operation("region.label_regions")["parameters"]}
    assert region["connectivity"]["default"] == 8
    assert region["connectivity"]["choices"] == [4, 8]


def test_explain_includes_threshold_units_validity_versions_and_source_identity(tmp_path):
    source_path = tmp_path / "illumination.tif"
    _write_source(source_path, np.array([[0.4, 0.8, 0.7]], dtype=np.float32))
    illumination = ma.source(source_path, units="fraction")
    expression = ma.where(illumination >= 0.6, illumination, ma.invalid)

    explanation = ma.explain(expression)
    assert "Scientific identity: sha256:" in explanation
    assert str(source_path.resolve()) in explanation
    assert "identity(mode=stat" in explanation
    assert "local.greater_equal v1" in explanation
    assert "0.6" in explanation
    assert "units='fraction'" in explanation
    assert "condition validity and selected-branch validity" in explanation
    assert "storage dtype=float32" in explanation
    assert "separate GDAL dataset mask" in explanation
    assert "human scientific review" in explanation


def test_plan_is_machine_readable_complete_and_read_only(tmp_path):
    source_path = tmp_path / "slope.tif"
    output_path = tmp_path / "new" / "candidate.tif"
    _write_source(source_path, np.array([[3.0, 9.0, 6.0]], dtype=np.float32))
    expression = ma.source(source_path, units="degrees") <= 8.0

    before = set(tmp_path.rglob("*"))
    result = ma.plan(expression, output=output_path)
    after = set(tmp_path.rglob("*"))

    assert before == after
    json.dumps(result)
    assert result["validated"] is True
    assert result["scientific_identity"] == expression.scientific_identity()
    assert json.loads(result["canonical_expression_json"])["schema_version"] == 3
    assert result["output_grid"]["width"] == 3
    assert result["output_contract"] == {
        "scientific_dtype": "bool",
        "storage_dtype": "uint8",
        "units": None,
        "invalid_fill": 0,
        "validity_encoding": "gdal_dataset_mask",
        "estimated_payload_bytes": 6,
    }
    assert result["output_preflight"]["resolved_path"] == str(output_path.resolve())
    assert result["output_preflight"]["parent_exists"] is False
    assert result["output_preflight"]["created_or_modified"] is False
    assert result["planner"]["execution_mode"] == "bounded_windowed_write"
    assert result["planner"]["backend"] == "cpu"
    assert result["planner"]["backend_availability"] == {
        "cpu": True, "cuda": False,
    }
    assert result["planner"]["estimated_temporary_bytes"] == 0
    assert result["planner"]["unsupported_nodes"] == []
    comparison = next(
        node for node in result["operations"]
        if node["operation_id"] == "local.less_equal"
    )
    assert comparison["operands"][1] == {
        "scalar": {"type": "float", "value": 8.0},
    }
    assert comparison["output_dtype_rule"] == "bool"


@pytest.mark.parametrize("function", [ma.explain, ma.plan])
def test_review_helpers_reject_non_expressions(function):
    raster = ma.raster(np.ones((2, 3), dtype=np.float32), _grid())
    with pytest.raises(MapAlgebraError) as error:
        function(raster)
    assert error.value.code == "map_algebra_invalid_expression"


def test_source_preflight_rejects_invalid_identity_band_and_units(tmp_path):
    source_path = tmp_path / "source.tif"
    _write_source(source_path, np.ones((2, 3), dtype=np.float32))
    with pytest.raises(MapAlgebraError) as identity_error:
        ma.source(source_path, identity="mtime")  # type: ignore[arg-type]
    assert identity_error.value.code == "map_algebra_invalid_source_identity"
    with pytest.raises(GeoTiffOpenError) as band_error:
        ma.source(source_path, band=0)
    assert band_error.value.code == "geotiff_band_out_of_range"
    with pytest.raises(MapAlgebraError) as units_error:
        ma.source(source_path, units="  ")
    assert units_error.value.code == "map_algebra_invalid_units"


def test_review_public_api_in_fresh_process():
    code = """
import json
import numpy as np
import lunarscout as ls
ma = ls.map_algebra
g = ls.GeoReference(
    projection_wkt='LOCAL_CS["Moon",UNIT["metre",1]]',
    projection_proj4=None,
    affine_transform=(0, 1, 0, 2, 0, -1),
    width=2, height=2, pixel_size_x=1, pixel_size_y=-1, nodata=None,
)
expr = (ma.raster(np.ones((2, 2), dtype=np.float32), g).expression() >= 0.5)
assert 'local.greater_equal' in ma.explain(expr)
try:
    plan = ma.plan(expr)
except ls.MapAlgebraExpressionError as error:
    assert error.code == 'map_algebra_unsupported_windowed_operation'
else:
    json.dumps(plan)
print('ok')
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == "ok"
