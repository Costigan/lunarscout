from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from ..errors import MapAlgebraExpressionError


@dataclass(frozen=True, slots=True)
class OperationSpec:
    """Immutable metadata for one built-in map-algebra operation."""

    id: str
    version: int
    arity: int | None
    category: str
    summary: str
    operand_kinds: tuple[str, ...] = ("raster",)
    parameters: tuple[tuple[str, str], ...] = ()
    eager_available: bool = True
    file_backed_available: bool = False
    output_dtype_rule: str = "operation-specific"
    output_units_rule: str = "operation-specific"
    validity_rule: str = "strict"
    cost_class: str = "linear"
    examples: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.id or "." not in self.id and self.id not in {"source", "constant"}:
            raise ValueError("Operation ids must be non-empty stable identifiers.")
        if not isinstance(self.version, int) or isinstance(self.version, bool) or self.version < 1:
            raise ValueError("Operation versions must be positive integers.")
        if self.arity is not None and self.arity < 0:
            raise ValueError("Operation arity must be non-negative or None.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "version": self.version,
            "arity": self.arity,
            "category": self.category,
            "summary": self.summary,
            "operand_kinds": list(self.operand_kinds),
            "parameters": [
                {"name": name, "description": description}
                for name, description in self.parameters
            ],
            "eager_available": self.eager_available,
            "file_backed_available": self.file_backed_available,
            "output_dtype_rule": self.output_dtype_rule,
            "output_units_rule": self.output_units_rule,
            "validity_rule": self.validity_rule,
            "cost_class": self.cost_class,
            "examples": list(self.examples),
        }


def _spec(
    operation_id: str,
    arity: int | None,
    category: str,
    summary: str,
    **kwargs: Any,
) -> OperationSpec:
    return OperationSpec(operation_id, 1, arity, category, summary, **kwargs)


_SPECS = (
    _spec("source", 0, "source", "Read a registered GeoTIFF source.", file_backed_available=True),
    _spec("constant", 1, "source", "Use an in-memory raster constant.", file_backed_available=True),
    *(
        _spec(f"local.{name}", 2, "local", summary, file_backed_available=True)
        for name, summary in (
            ("add", "Add corresponding cells."),
            ("subtract", "Subtract corresponding cells."),
            ("multiply", "Multiply corresponding cells."),
            ("divide", "Divide corresponding cells."),
            ("floor_divide", "Floor-divide corresponding cells."),
            ("remainder", "Calculate the corresponding-cell remainder."),
            ("power", "Raise corresponding cells to a power."),
            ("minimum", "Select the corresponding-cell minimum."),
            ("maximum", "Select the corresponding-cell maximum."),
            ("less", "Compare corresponding cells with less-than."),
            ("less_equal", "Compare corresponding cells with less-than-or-equal."),
            ("greater", "Compare corresponding cells with greater-than."),
            ("greater_equal", "Compare corresponding cells with greater-than-or-equal."),
            ("equal", "Compare corresponding cells for equality."),
            ("not_equal", "Compare corresponding cells for inequality."),
            ("logical_and", "Apply Boolean AND."),
            ("logical_or", "Apply Boolean OR."),
            ("logical_xor", "Apply Boolean XOR."),
            ("hypot", "Calculate the corresponding-cell hypotenuse."),
            ("arctan2", "Calculate the two-argument arctangent."),
        )
    ),
    *(
        _spec(f"local.{name}", 1, "local", summary, file_backed_available=True)
        for name, summary in (
            ("negative", "Negate cells."), ("absolute", "Calculate absolute values."),
            ("sqrt", "Calculate square roots."), ("square", "Square cells."),
            ("exp", "Calculate exponentials."), ("log", "Calculate natural logarithms."),
            ("log10", "Calculate base-10 logarithms."), ("sin", "Calculate sine."),
            ("cos", "Calculate cosine."), ("tan", "Calculate tangent."),
            ("arcsin", "Calculate arcsine."), ("arccos", "Calculate arccosine."),
            ("arctan", "Calculate arctangent."), ("logical_not", "Apply Boolean NOT."),
            ("floor", "Round down."), ("ceil", "Round up."),
            ("trunc", "Truncate fractional values."),
            ("degrees", "Convert radians to degrees."), ("radians", "Convert degrees to radians."),
            ("is_valid", "Return the validity mask."), ("is_invalid", "Return the invalidity mask."),
        )
    ),
    _spec("local.where", 3, "local", "Select between branches by a Boolean condition.", file_backed_available=True),
    _spec("local.round", 1, "local", "Round half to even.",
          parameters=(("ndigits", "Number of decimal digits."),), file_backed_available=True),
    _spec("local.isclose", 2, "local", "Compare corresponding cells within tolerances.",
          parameters=(("rtol", "Relative tolerance."), ("atol", "Absolute tolerance."),
                      ("equal_nan", "Whether NaN values compare equal.")), file_backed_available=True),
    _spec("local.coalesce", None, "local", "Select the first valid operand.", file_backed_available=True),
    _spec("local.clip", 1, "local", "Clip values to an interval.", file_backed_available=True),
    _spec("local.cast", 1, "local", "Cast values to a requested dtype.", file_backed_available=True),
    _spec("local.set_invalid", 2, "local", "Invalidate cells selected by a mask.", file_backed_available=True),
    _spec("local.fill_invalid", 2, "local", "Fill and validate invalid cells.", file_backed_available=True),
    _spec("local.reclassify_values", 1, "classification", "Map exact input values to classes.",
          parameters=(("mapping", "Exact input-to-output mapping."), ("default", "Unmatched-cell behavior.")), file_backed_available=True),
    _spec("local.reclassify_ranges", 1, "classification", "Map half-open input ranges to classes.",
          parameters=(("ranges", "Half-open lower, upper, output triples."), ("default", "Unmatched-cell behavior.")), file_backed_available=True),
    _spec("local.digitize", 1, "classification", "Assign values to ordered bins.",
          parameters=(("bins", "Monotonically increasing bin edges."), ("right", "Use right-closed bins.")), file_backed_available=True),
    _spec("local.one_hot", 1, "classification", "Create one Boolean raster per requested class.",
          parameters=(("classes", "Class values in output order."),), file_backed_available=True),
    _spec("local.normalize_minmax", 1, "normalization", "Scale values by a minimum and maximum; file-backed execution requires both statistics.",
          parameters=(("minimum", "Supplied or measured minimum."), ("maximum", "Supplied or measured maximum.")), file_backed_available=True),
    _spec("local.standardize", 1, "normalization", "Center and scale values by mean and standard deviation; file-backed execution requires both statistics.",
          parameters=(("mean", "Supplied or measured mean."), ("std", "Supplied or measured standard deviation."),
                      ("ddof", "Delta degrees of freedom for a measured standard deviation.")), file_backed_available=True),
    *(
        _spec(f"coordinate.{name}", 0, "coordinate", summary, file_backed_available=True)
        for name, summary in (
            ("row_indices", "Generate zero-based row indices."),
            ("column_indices", "Generate zero-based column indices."),
            ("projected_x", "Generate x coordinates in the grid CRS."),
            ("projected_y", "Generate y coordinates in the grid CRS."),
            ("longitude", "Generate longitudes in the grid geodetic CRS."),
            ("latitude", "Generate latitudes in the grid geodetic CRS."),
        )
    ),
    *(
        _spec(f"focal.{name}", 1, "focal", f"Apply focal {name}.", cost_class="neighborhood")
        for name in ("sum", "mean", "min", "max", "range", "std", "count", "median",
                     "dilate", "erode", "opening", "closing", "majority", "convolve")
    ),
    _spec("global.statistics", 1, "global", "Calculate global descriptive statistics.", cost_class="global"),
    _spec("global.histogram", 1, "global", "Calculate a global histogram.", cost_class="global"),
    _spec("global.percentile", 1, "global", "Calculate global percentiles.", cost_class="global"),
    _spec("global.unique_counts", 1, "global", "Count unique valid values.", cost_class="global"),
    _spec("zonal.stats", 2, "zonal", "Calculate statistics grouped by zone.", cost_class="global"),
    _spec("zonal.raster", 2, "zonal", "Broadcast a zonal statistic to zone cells.", cost_class="global"),
    _spec("distance.to", 1, "distance", "Calculate distance to Boolean seed cells.", cost_class="global"),
    _spec("distance.signed", 1, "distance", "Calculate a signed distance field.", cost_class="global"),
    *(
        _spec(f"temporal.{name}", 1, "temporal", f"Apply temporal {name}.", cost_class="temporal")
        for name in ("mean", "min", "max", "std", "sum", "count")
    ),
    _spec("temporal.source", 0, "source", "Read a file-backed temporal series.", cost_class="temporal"),
    _spec("temporal.constant", 1, "source", "Use an in-memory temporal constant.", cost_class="temporal"),
    _spec("temporal.broadcast", 1, "temporal", "Broadcast a spatial raster over time.", cost_class="temporal"),
)


def _build_registry() -> dict[str, OperationSpec]:
    registry: dict[str, OperationSpec] = {}
    for spec in _SPECS:
        if spec.id in registry:
            raise RuntimeError(f"Duplicate built-in operation id: {spec.id}")
        registry[spec.id] = spec
    return registry


_REGISTRY = _build_registry()


def get_operation_spec(operation_id: str) -> OperationSpec:
    try:
        return _REGISTRY[operation_id]
    except KeyError as exc:
        raise MapAlgebraExpressionError(
            f"Unknown map-algebra operation: {operation_id}",
            code="map_algebra_unknown_operation",
            details={"operation_id": operation_id},
        ) from exc


def describe_operation(operation_id: str) -> dict[str, Any]:
    return get_operation_spec(operation_id).to_dict()


def list_operations(
    *,
    category: str | None = None,
    execution_mode: Literal["eager", "file_backed"] | None = None,
) -> list[dict[str, Any]]:
    specs = _REGISTRY.values()
    if category is not None:
        specs = (spec for spec in specs if spec.category == category)
    if execution_mode == "eager":
        specs = (spec for spec in specs if spec.eager_available)
    elif execution_mode == "file_backed":
        specs = (spec for spec in specs if spec.file_backed_available)
    elif execution_mode is not None:
        raise ValueError("execution_mode must be 'eager', 'file_backed', or None.")
    return [spec.to_dict() for spec in sorted(specs, key=lambda item: item.id)]
