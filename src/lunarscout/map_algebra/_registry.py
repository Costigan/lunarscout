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
    version = kwargs.pop("version", 1)
    return OperationSpec(operation_id, version, arity, category, summary, **kwargs)


_LOCAL_BINARY_PARAMETERS: dict[str, tuple[tuple[str, str], ...]] = {
    "add": (
        ("overflow", "Integer overflow policy: raise, wrap, or promote."),
        ("numeric_errors", "Non-finite policy: invalid, keep, or raise."),
    ),
    "subtract": (
        ("overflow", "Integer overflow policy: raise, wrap, or promote."),
        ("numeric_errors", "Non-finite policy: invalid, keep, or raise."),
    ),
    "multiply": (
        ("output_units", "Required output units for two unit-bearing rasters."),
        ("overflow", "Integer overflow policy: raise, wrap, or promote."),
        ("numeric_errors", "Non-finite policy: invalid, keep, or raise."),
    ),
    "divide": (
        ("output_units", "Required output units for two unit-bearing rasters."),
        ("numeric_errors", "Non-finite/domain policy: invalid, keep, or raise."),
    ),
    "floor_divide": (
        ("overflow", "Integer overflow policy: raise, wrap, or promote."),
        ("numeric_errors", "Division-by-zero policy: invalid, keep, or raise."),
    ),
    "remainder": (
        ("overflow", "Integer overflow policy: raise, wrap, or promote."),
        ("numeric_errors", "Division-by-zero policy: invalid, keep, or raise."),
    ),
    "power": (
        ("output_units", "Required for a unit-bearing base unless exponent is one."),
        ("overflow", "Integer overflow policy: raise, wrap, or promote."),
        ("numeric_errors", "Non-finite/negative-integer-exponent policy."),
    ),
    **{
        name: (("numeric_errors", "Non-finite policy: invalid, keep, or raise."),)
        for name in ("minimum", "maximum", "hypot", "arctan2")
    },
}

_LOCAL_UNARY_PARAMETERS: dict[str, tuple[tuple[str, str], ...]] = {
    "negative": (
        ("overflow", "Integer overflow policy: raise, wrap, or promote."),
        ("numeric_errors", "Non-finite policy: invalid, keep, or raise."),
    ),
    "absolute": (
        ("overflow", "Integer overflow policy: raise, wrap, or promote."),
        ("numeric_errors", "Non-finite policy: invalid, keep, or raise."),
    ),
    "square": (
        ("overflow", "Integer overflow policy: raise, wrap, or promote."),
        ("numeric_errors", "Non-finite policy: invalid, keep, or raise."),
    ),
    **{
        name: (("numeric_errors", "Non-finite/domain policy: invalid, keep, or raise."),)
        for name in (
            "sqrt", "exp", "log", "log10", "sin", "cos", "tan",
            "arcsin", "arccos", "arctan", "floor", "ceil", "trunc",
            "degrees", "radians",
        )
    },
}


_SPECS = (
    _spec("source", 0, "source", "Read a registered GeoTIFF source.", file_backed_available=True),
    _spec("constant", 1, "source", "Use an in-memory raster constant.", file_backed_available=True),
    *(
        _spec(
            f"local.{name}", 2, "local", summary,
            parameters=_LOCAL_BINARY_PARAMETERS.get(name, ()),
            version=(3 if name == "power" else 2 if name in _LOCAL_BINARY_PARAMETERS else 1),
            output_units_rule=(
                "unit-bearing base requires scalar exponent and explicit output_units unless exponent is one"
                if name == "power" else "operation-specific"
            ),
            file_backed_available=True,
        )
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
        _spec(
            f"local.{name}", 1, "local", summary,
            parameters=_LOCAL_UNARY_PARAMETERS.get(name, ()),
            version=2 if name in _LOCAL_UNARY_PARAMETERS else 1,
            file_backed_available=True,
        )
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
    _spec(
        "local.sum_layers", None, "local",
        "Add one or more layers by composing ordinary local addition.",
        parameters=(
            ("overflow", "Integer overflow policy: raise, wrap, or promote."),
            ("numeric_errors", "Non-finite policy: invalid, keep, or raise."),
        ),
        output_dtype_rule="left-associated local.add promotion",
        output_units_rule="all layers must have matching units",
        validity_rule="strict intersection",
    ),
    _spec(
        "local.mean_layers", None, "local",
        "Calculate the arithmetic mean of one or more layers.",
        parameters=(
            ("overflow", "Integer sum overflow policy: raise, wrap, or promote."),
            ("numeric_errors", "Non-finite policy: invalid, keep, or raise."),
        ),
        output_dtype_rule="sum_layers followed by true division",
        output_units_rule="all layers must have matching units",
        validity_rule="strict intersection",
    ),
    *(
        _spec(
            f"local.{name}_layers", None, "local",
            f"Calculate the cell-wise {name} of one or more layers.",
            parameters=((
                "numeric_errors", "Non-finite policy: invalid, keep, or raise.",
            ),),
            output_dtype_rule=f"left-associated local.{name}imum promotion",
            output_units_rule="all layers must have matching units",
            validity_rule="strict intersection",
        )
        for name in ("min", "max")
    ),
    _spec(
        "local.where", 3, "local",
        "Select between branches by a Boolean condition.",
        version=2,
        output_dtype_rule="exact common selection dtype",
        output_units_rule="raster branches must have matching units",
        validity_rule="condition validity and selected-branch validity",
        file_backed_available=True,
    ),
    _spec("local.round", 1, "local", "Round half to even.",
          version=2,
          parameters=(
              ("ndigits", "Number of decimal digits."),
              ("numeric_errors", "Non-finite policy: invalid, keep, or raise."),
          ), file_backed_available=True),
    _spec("local.isclose", 2, "local", "Compare corresponding cells within tolerances.",
          parameters=(("rtol", "Relative tolerance."), ("atol", "Absolute tolerance."),
                      ("equal_nan", "Whether NaN values compare equal.")), file_backed_available=True),
    _spec(
        "local.coalesce", None, "local", "Select the first valid operand.",
        version=2,
        output_dtype_rule="exact common selection dtype",
        output_units_rule="raster operands must have matching units",
        validity_rule="first valid operand",
        file_backed_available=True,
    ),
    _spec("local.clip", 1, "local", "Clip values to an interval.", file_backed_available=True),
    _spec(
        "local.cast", 1, "local", "Cast values to a requested dtype.",
        version=2,
        parameters=(
            ("casting", "NumPy type-level rule: safe, same_kind, or unsafe."),
            ("overflow", "Value overflow policy: raise or integer wrap."),
        ),
        file_backed_available=True,
    ),
    _spec("local.set_invalid", 2, "local", "Invalidate cells selected by a mask.", file_backed_available=True),
    _spec(
        "local.fill_invalid", 2, "local", "Fill and validate invalid cells.",
        version=2,
        output_dtype_rule="input dtype; fill must be exactly representable",
        validity_rule="all cells valid after exact fill",
        file_backed_available=True,
    ),
    _spec("local.reclassify_values", 1, "classification", "Map exact input values to classes.",
          version=2,
          output_dtype_rule="smallest supported dtype representing every output; preserve also includes the source dtype",
          parameters=(("mapping", "Exact input-to-output mapping."), ("default", "Unmatched-cell behavior.")), file_backed_available=True),
    _spec("local.reclassify_ranges", 1, "classification", "Map half-open input ranges to classes.",
          version=2,
          output_dtype_rule="smallest supported dtype representing every output; preserve also includes the source dtype",
          parameters=(("ranges", "Half-open lower, upper, output triples."), ("default", "Unmatched-cell behavior.")), file_backed_available=True),
    _spec("local.digitize", 1, "classification", "Assign values to ordered bins.",
          parameters=(("bins", "Monotonically increasing bin edges."), ("right", "Use right-closed bins.")), file_backed_available=True),
    _spec("local.one_hot", 1, "classification", "Create one Boolean raster per requested class.",
          parameters=(("classes", "Class values in output order."),), file_backed_available=True),
    _spec("local.normalize_minmax", 1, "normalization", "Scale values by a minimum and maximum; file-backed execution requires both statistics.", version=2,
          output_dtype_rule="FP32 for FP32 and Boolean/8/16-bit inputs unless typed FP64 statistics require FP64; otherwise FP64",
          parameters=(("minimum", "Supplied or measured minimum."), ("maximum", "Supplied or measured maximum.")), file_backed_available=True),
    _spec("local.standardize", 1, "normalization", "Center and scale values by mean and standard deviation; file-backed execution requires both statistics.", version=2,
          output_dtype_rule="FP32 for FP32 and Boolean/8/16-bit inputs unless typed FP64 statistics require FP64; otherwise FP64",
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
        _spec(
            f"focal.{name}", 1, "focal", f"Apply focal {name}.",
            version=(
                4 if name == "range"
                else 3 if name in {"sum", "mean", "min", "max", "std", "median"}
                else 2
            ),
            output_dtype_rule=(
                "accumulator_dtype(source_dtype)"
                if name in {"sum", "mean", "min", "max", "std", "count"}
                else "operation-specific"
            ),
            output_units_rule="None" if name == "count" else "source units",
            parameters=(
                ("size", "Odd scalar or rectangular neighborhood dimensions."),
                ("footprint", "Explicit odd two-dimensional Boolean footprint."),
                ("edge", "Edge mode: invalid, constant, nearest, reflect, or wrap."),
                ("valid_neighbor", "Validity policy for neighborhood cells."),
                ("min_valid_count", "Minimum valid cells when ignoring invalid neighbors."),
                ("cval", "Constant padding value."),
                *((
                    ("ddof", "Delta degrees of freedom."),
                ) if name == "std" else ()),
            ),
            cost_class="neighborhood",
        )
        for name in ("sum", "mean", "min", "max", "range", "std", "count", "median")
    ),
    *(
        _spec(
            f"focal.{name}", 1, "focal", f"Apply focal {name}.",
            cost_class="neighborhood",
        )
        for name in ("dilate", "erode", "opening", "closing", "majority")
    ),
    _spec(
        "focal.convolve", 2, "focal", "Apply a finite convolution kernel.",
        version=3,
        parameters=(
            ("normalize", "Normalize by the sum of absolute kernel weights."),
            ("edge", "Edge mode: invalid, constant, nearest, reflect, or wrap."),
            ("valid_neighbor", "Validity policy for neighborhood cells."),
            ("min_valid_count", "Minimum valid cells when ignoring invalid neighbors."),
            ("cval", "Constant padding value."),
        ),
        cost_class="neighborhood",
    ),
    _spec("terrain.slope", 1, "terrain", "Calculate terrain slope from an elevation raster.",
          parameters=(("output_nodata", "Sentinel value at invalid cells (default NaN)."),
                      ("units", "'degrees' (default) or 'percent'."),
                      ("compute_edges", "When True, compute valid edges; when False, border cells are invalid."),
                      ("scale", "Horizontal-to-vertical ratio (default 1.0).")),
          file_backed_available=True, cost_class="neighborhood",
          output_dtype_rule="float32", output_units_rule="degrees or percent parameter",
          validity_rule="canonical gradient validity independent of output_nodata",
          examples=("slope(dem)  # default degrees", "slope(dem, units='percent', scale=2.0)")),
    _spec("terrain.aspect", 1, "terrain", "Calculate terrain aspect azimuth from an elevation raster.",
          parameters=(("output_nodata", "Sentinel value at invalid cells (default NaN)."),
                      ("compute_edges", "When True, compute valid edges; when False, border cells are invalid.")),
          file_backed_available=True, cost_class="neighborhood",
          output_dtype_rule="float32", output_units_rule="degrees",
          validity_rule="canonical gradient validity independent of output_nodata; flat cells are invalid",
          examples=("aspect(dem)", "aspect(dem, output_nodata=270.0)")),
    _spec("terrain.hillshade", 1, "terrain", "Calculate shaded terrain relief from an elevation raster.",
          parameters=(("output_nodata", "Sentinel value at invalid cells (default 0)."),
                      ("azimuth", "Illumination azimuth degrees clockwise from north (default 315)."),
                      ("altitude", "Illumination altitude degrees above horizon (default 45)."),
                      ("compute_edges", "When True, compute valid edges; when False, border cells are invalid."),
                      ("scale", "Horizontal-to-vertical ratio (default 1.0)."),
                      ("z_factor", "Vertical exaggeration factor (default 1.0).")),
          file_backed_available=True, cost_class="neighborhood",
          output_dtype_rule="uint8", output_units_rule="None",
          validity_rule="canonical neighbourhood gradient validity independent of output_nodata",
          examples=("hillshade(dem)", "hillshade(dem, azimuth=180.0, altitude=30.0)")),
    _spec("alignment.resample_to", 1, "alignment", "Resample onto an explicit destination grid.",
          parameters=(("resampling", "Algorithm name: nearest, bilinear, cubic, etc. (default 'nearest')."),
                      ("output_dtype", "Explicit output dtype or None to preserve source dtype."),
                      ("validity_coverage_threshold", "Minimum valid source fraction per output pixel or None."),
                      ("categorical", "Explicit categorical/continuous flag or None for auto-inference."),
                      ("allow_unsafe", "Suppress categorical-safety rejection.")),
          file_backed_available=True, cost_class="resampling",
          output_dtype_rule="source or explicit", output_units_rule="preserve source",
          validity_rule="nearest-neighbour categorical validity by default; coverage threshold optional",
          examples=("resample_to(src, dst_grid)", "resample_to(src, dst_grid, resampling='bilinear')")),
    _spec("global.statistics", 1, "global", "Calculate global descriptive statistics.",
          version=2, output_dtype_rule="exact integer sum/extrema/range; fractional moments are float64",
          cost_class="global"),
    _spec("global.histogram", 1, "global", "Calculate a global histogram.",
          version=2, parameters=(
              ("bins", "Bin count or explicit monotonically increasing edges."),
              ("range", "Optional lower and upper histogram range."),
          ), cost_class="global"),
    _spec("global.percentile", 1, "global", "Calculate global percentiles.",
          version=2, parameters=(
              ("q", "Finite percentile or percentiles from 0 through 100."),
              ("method", "Exact linear interpolation or approximate nearest selection."),
          ), output_dtype_rule="observed integer selections preserve source dtype; interpolated results are float64",
          cost_class="global"),
    _spec("global.unique_counts", 1, "global", "Count unique valid values.", cost_class="global"),
    *(
        _spec(
            f"region.{name}", 1, "region", summary,
            parameters=(
                ("cleanup", "Pre-label cleanup: none, erosion, or opening."),
                ("iterations", "Non-negative cleanup iteration count."),
                ("connectivity", "Connected-neighbor rule: 4 or 8."),
            ),
            output_dtype_rule="int32",
            output_units_rule="None",
            validity_rule="preserve canonical input validity",
            cost_class="global",
        )
        for name, summary in (
            ("label_regions", "Label connected valid true cells."),
            ("region_sizes", "Broadcast connected-region sizes to true cells."),
        )
    ),
    _spec(
        "region.filter_regions_by_size", 1, "region",
        "Keep connected regions selected by pixel count.",
        parameters=(
            ("threshold", "Finite non-negative region-size threshold."),
            ("comparator", "Threshold comparison: >= or <=."),
            ("cleanup", "Pre-label cleanup: none, erosion, or opening."),
            ("iterations", "Non-negative cleanup iteration count."),
            ("connectivity", "Connected-neighbor rule: 4 or 8."),
        ),
        output_dtype_rule="bool",
        output_units_rule="None",
        validity_rule="preserve canonical input validity",
        cost_class="global",
    ),
    _spec(
        "region.find_borders", 1, "region",
        "Return internal border cells of valid true regions.",
        parameters=(("connectivity", "Border-neighbor rule: 4 or 8."),),
        output_dtype_rule="bool",
        output_units_rule="None",
        validity_rule="preserve canonical input validity",
        cost_class="global",
    ),
    _spec("zonal.stats", 2, "zonal", "Calculate statistics grouped by zone.",
          version=2, output_dtype_rule="counts int64; integer sums/extrema/ranges remain integer; fractional statistics float64",
          cost_class="global"),
    _spec("zonal.raster", 2, "zonal", "Broadcast a zonal statistic to zone cells.",
          version=2, output_dtype_rule="selected zonal statistic dtype",
          cost_class="global"),
    _spec("distance.to", 1, "distance", "Calculate distance to Boolean seed cells.", cost_class="global"),
    _spec("distance.signed", 1, "distance", "Calculate a signed distance field.", cost_class="global"),
    *(
        _spec(
            f"temporal.{name}",
            1,
            "temporal",
            f"Apply temporal {name}.",
            version=2 if name in {"mean", "std", "sum"} else 1,
            parameters=(
                (("ddof", "Non-negative finite delta degrees of freedom."),)
                if name == "std" else ()
            ),
            output_dtype_rule=(
                "accumulator_dtype(source_dtype)"
                if name in {"mean", "std", "sum", "count"}
                else "source dtype"
            ),
            output_units_rule="None" if name == "count" else "source units",
            cost_class="temporal",
        )
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
