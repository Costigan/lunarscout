from __future__ import annotations

from functools import wraps
from inspect import signature
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from ..errors import (
    GeoTiffOpenError,
    MapAlgebraError,
    RasterValidationError,
)
from ..georeference import GeoReference
from ..geotiff import read_geotiff as _read_geotiff
from ..raster import (
    Raster,
    _fill_invalid_exact,
    _validate_nodata_representable,
    _valid_from_nodata,
)

from .local import (
    absolute,
    add,
    arccos,
    arcsin,
    arctan,
    arctan2,
    cast,
    ceil,
    clip,
    coalesce,
    cos,
    degrees,
    digitize,
    divide,
    equal,
    exp,
    fill_invalid,
    floor,
    floor_divide,
    greater,
    greater_equal,
    hypot,
    invalid,
    is_invalid,
    is_valid,
    isclose,
    less,
    less_equal,
    log,
    log10,
    logical_and,
    logical_not,
    logical_or,
    logical_xor,
    max_layers as _max_layers_eager,
    maximum,
    mean_layers as _mean_layers_eager,
    min_layers as _min_layers_eager,
    minimum,
    multiply,
    negative,
    normalize_minmax,
    not_equal,
    one_hot,
    positive,
    power,
    radians,
    reclassify_ranges,
    reclassify_values,
    remainder,
    round_half_even as round,
    standardize,
    sum_layers as _sum_layers_eager,
    set_invalid,
    sin,
    sqrt,
    square,
    subtract,
    tan,
    trunc,
    where,
)

from ._model import RasterExpression
from .expression import (
    compute,
    explain,
    plan,
)
from ._sources import source
from ._writer import write
from ._temporal_model import TemporalRaster, TemporalRasterExpression
from .temporal import (
    compute_temporal,
    explain_temporal,
    temporal_count,
    temporal_max,
    temporal_mean,
    temporal_min,
    temporal_source,
    temporal_std,
    temporal_sum,
    from_temporal_cube,
)
from .focal import (
    closing,
    convolve,
    dilate,
    erode,
    focal_count,
    focal_max,
    focal_mean,
    focal_median,
    focal_min,
    focal_range,
    focal_std,
    focal_sum,
    majority,
    opening,
)
from .reductions import (
    RasterStatistics,
    histogram,
    percentile,
    statistics,
    unique_counts,
)
from .zonal import (
    ZonalStatistics,
    zonal_raster,
    zonal_stats,
)
from .distance import (
    distance_to,
    signed_distance,
)
from .regions import (
    filter_regions_by_size,
    find_borders,
    label_regions,
    region_sizes,
)
from .coordinates import (
    column_indices,
    latitude,
    longitude,
    projected_x,
    projected_y,
    row_indices,
)
from ._registry import (
    describe_operation,
    list_operations,
)
from ._spatial import make_resample_expression, make_terrain_expression


# ---------------------------------------------------------------------------
# Expression-dispatch wrappers for unary & binary functions
# ---------------------------------------------------------------------------

def _has_expr(*args: Any) -> bool:
    return any(isinstance(a, (RasterExpression, TemporalRasterExpression)) for a in args)


def _is_temporal_expr(a: Any) -> bool:
    return isinstance(a, TemporalRasterExpression)


def _normalized_numeric_params(kwargs: dict[str, Any]) -> dict[str, Any]:
    params = dict(kwargs)
    if "overflow" in params:
        from ._dtypes import normalize_overflow

        params["overflow"] = normalize_overflow(params["overflow"])
    if "numeric_errors" in params:
        from ._validity import normalize_numeric_errors

        params["numeric_errors"] = normalize_numeric_errors(params["numeric_errors"])
    return params


def _wrap_unary(fn: Any, op_id: str) -> Any:
    fn_signature = signature(fn)

    @wraps(fn)
    def _wrapper(a: Any, **kwargs: Any) -> Any:
        fn_signature.bind(a, **kwargs)
        if isinstance(a, TemporalRasterExpression):
            if kwargs:
                raise TypeError(f"{fn.__name__}() numeric policies are not supported for temporal expressions yet.")
            from ._temporal_model import _temporal_local_op
            return _temporal_local_op(op_id, a)
        if isinstance(a, RasterExpression):
            from ._model import _new_expr_unary
            return _new_expr_unary(a, op_id, params=_normalized_numeric_params(kwargs))
        return fn(a, **kwargs)
    return _wrapper


sqrt = _wrap_unary(sqrt, "local.sqrt")
square = _wrap_unary(square, "local.square")
exp = _wrap_unary(exp, "local.exp")
log = _wrap_unary(log, "local.log")
log10 = _wrap_unary(log10, "local.log10")
sin = _wrap_unary(sin, "local.sin")
cos = _wrap_unary(cos, "local.cos")
tan = _wrap_unary(tan, "local.tan")
arcsin = _wrap_unary(arcsin, "local.arcsin")
arccos = _wrap_unary(arccos, "local.arccos")
arctan = _wrap_unary(arctan, "local.arctan")
negative = _wrap_unary(negative, "local.negative")
absolute = _wrap_unary(absolute, "local.absolute")
logical_not = _wrap_unary(logical_not, "local.logical_not")
floor = _wrap_unary(floor, "local.floor")
ceil = _wrap_unary(ceil, "local.ceil")
trunc = _wrap_unary(trunc, "local.trunc")
_round_eager = round


def round(
    a: Any,
    ndigits: int = 0,
    *,
    numeric_errors: Literal["invalid", "keep", "raise"] = "invalid",
) -> Any:
    if not isinstance(a, RasterExpression):
        return _round_eager(a, ndigits, numeric_errors=numeric_errors)
    if not isinstance(ndigits, int) or isinstance(ndigits, bool):
        raise TypeError("ndigits must be an integer.")
    from ._model import _make_expr_node
    from ._validity import normalize_numeric_errors
    return _make_expr_node(
        "local.round", (a,), grid=a.grid, dtype=a.dtype, units=a.units,
        params={
            "ndigits": ndigits,
            "numeric_errors": normalize_numeric_errors(numeric_errors),
        },
    )


degrees = _wrap_unary(degrees, "local.degrees")
radians = _wrap_unary(radians, "local.radians")


_positive_eager = positive


def positive(a: Any) -> Any:
    if isinstance(a, RasterExpression):
        return a
    return _positive_eager(a)


def _wrap_binary(fn: Any, op_id: str) -> Any:
    fn_signature = signature(fn)

    @wraps(fn)
    def _wrapper(a: Any, b: Any, **kwargs: Any) -> Any:
        fn_signature.bind(a, b, **kwargs)
        if isinstance(a, (TemporalRasterExpression, TemporalRaster)) or isinstance(b, (TemporalRasterExpression, TemporalRaster)):
            if kwargs:
                raise TypeError(f"{fn.__name__}() keyword policies are not supported for temporal expressions yet.")
            from ._temporal_model import _temporal_local_op
            return _temporal_local_op(op_id, a, b)
        if isinstance(a, RasterExpression) or isinstance(b, RasterExpression):
            from ._model import _new_expr_op
            params = _normalized_numeric_params(kwargs)
            if op_id == "local.power" and "output_units" in params:
                from ._units import normalize_output_units

                normalized_units = normalize_output_units(params["output_units"])
                if normalized_units is None:
                    params.pop("output_units")
                else:
                    params["output_units"] = normalized_units
            return _new_expr_op(a, b, op_id, params=params)
        return fn(a, b, **kwargs)
    return _wrapper


add = _wrap_binary(add, "local.add")
subtract = _wrap_binary(subtract, "local.subtract")
multiply = _wrap_binary(multiply, "local.multiply")
divide = _wrap_binary(divide, "local.divide")
minimum = _wrap_binary(minimum, "local.minimum")
maximum = _wrap_binary(maximum, "local.maximum")
power = _wrap_binary(power, "local.power")
floor_divide = _wrap_binary(floor_divide, "local.floor_divide")
remainder = _wrap_binary(remainder, "local.remainder")
less = _wrap_binary(less, "local.less")
less_equal = _wrap_binary(less_equal, "local.less_equal")
greater = _wrap_binary(greater, "local.greater")
greater_equal = _wrap_binary(greater_equal, "local.greater_equal")
equal = _wrap_binary(equal, "local.equal")
not_equal = _wrap_binary(not_equal, "local.not_equal")
logical_and = _wrap_binary(logical_and, "local.logical_and")
logical_or = _wrap_binary(logical_or, "local.logical_or")
logical_xor = _wrap_binary(logical_xor, "local.logical_xor")
hypot = _wrap_binary(hypot, "local.hypot")
arctan2 = _wrap_binary(arctan2, "local.arctan2")


def _validate_nonempty_layers(layers: tuple[Any, ...], *, operation: str) -> None:
    if not layers:
        raise MapAlgebraError(
            f"{operation}() requires at least one Raster or RasterExpression.",
            code="map_algebra_empty_layers",
            details={"operation": operation},
        )
    for index, layer in enumerate(layers):
        if not isinstance(layer, (Raster, RasterExpression)):
            raise MapAlgebraError(
                f"{operation}() layers must be Raster or RasterExpression values.",
                code="map_algebra_invalid_layer",
                details={
                    "operation": operation,
                    "layer_index": index,
                    "type": type(layer).__name__,
                },
            )


def sum_layers(
    *layers: Any,
    overflow: Literal["raise", "wrap", "promote"] = "raise",
    numeric_errors: Literal["invalid", "keep", "raise"] = "invalid",
) -> Any:
    _validate_nonempty_layers(layers, operation="sum_layers")
    if not _has_expr(*layers):
        return _sum_layers_eager(
            *layers, overflow=overflow, numeric_errors=numeric_errors,
        )
    result = add(
        layers[0], 0, overflow=overflow, numeric_errors=numeric_errors,
    )
    for layer in layers[1:]:
        result = add(
            result, layer, overflow=overflow, numeric_errors=numeric_errors,
        )
    return result


def mean_layers(
    *layers: Any,
    overflow: Literal["raise", "wrap", "promote"] = "raise",
    numeric_errors: Literal["invalid", "keep", "raise"] = "invalid",
) -> Any:
    _validate_nonempty_layers(layers, operation="mean_layers")
    if not _has_expr(*layers):
        return _mean_layers_eager(
            *layers, overflow=overflow, numeric_errors=numeric_errors,
        )
    total = sum_layers(
        *layers, overflow=overflow, numeric_errors=numeric_errors,
    )
    return divide(total, len(layers), numeric_errors=numeric_errors)


def min_layers(
    *layers: Any,
    numeric_errors: Literal["invalid", "keep", "raise"] = "invalid",
) -> Any:
    _validate_nonempty_layers(layers, operation="min_layers")
    if not _has_expr(*layers):
        return _min_layers_eager(*layers, numeric_errors=numeric_errors)
    result = minimum(
        layers[0], layers[0], numeric_errors=numeric_errors,
    )
    for layer in layers[1:]:
        result = minimum(result, layer, numeric_errors=numeric_errors)
    return result


def max_layers(
    *layers: Any,
    numeric_errors: Literal["invalid", "keep", "raise"] = "invalid",
) -> Any:
    _validate_nonempty_layers(layers, operation="max_layers")
    if not _has_expr(*layers):
        return _max_layers_eager(*layers, numeric_errors=numeric_errors)
    result = maximum(
        layers[0], layers[0], numeric_errors=numeric_errors,
    )
    for layer in layers[1:]:
        result = maximum(result, layer, numeric_errors=numeric_errors)
    return result


_isclose_eager = isclose


def isclose(
    a: Any,
    b: Any,
    *,
    rtol: float = 1e-5,
    atol: float = 1e-8,
    equal_nan: bool = False,
) -> Any:
    if not _has_expr(a, b):
        return _isclose_eager(
            a, b, rtol=rtol, atol=atol, equal_nan=equal_nan,
        )
    from ._model import _make_expr_node, _to_expr_or_scalar
    left = _to_expr_or_scalar(a)
    right = _to_expr_or_scalar(b)
    return _make_expr_node(
        "local.isclose",
        (left, right),
        grid=_expression_grid(left, right),
        dtype=np.dtype(np.bool_),
        units=None,
        params={"rtol": rtol, "atol": atol, "equal_nan": equal_nan},
    )


# ---------------------------------------------------------------------------
# Expression-dispatch wrappers for conditional and validity operations
# ---------------------------------------------------------------------------

_where_eager = where
_coalesce_eager = coalesce
_is_valid_eager = is_valid
_is_invalid_eager = is_invalid
_set_invalid_eager = set_invalid
_fill_invalid_eager = fill_invalid
_clip_eager = clip
_cast_eager = cast


def _expression_grid(*operands: Any) -> GeoReference | None:
    from ._model import _infer_common_grid

    return _infer_common_grid(
        [operand for operand in operands if isinstance(operand, RasterExpression)]
    )


def where(condition: Any, x: Any, y: Any) -> Any:
    """Select branch values with selected-branch validity.

    Raster branches must share a grid and matching units. Dtype inference is
    identical for eager and expression execution and never silently wraps a
    Python integer branch.
    """
    if not _has_expr(condition, x, y):
        return _where_eager(condition, x, y)
    from ._model import _make_expr_node, _to_expr_or_scalar
    from ..errors import MapAlgebraDTypeError

    condition_expr = _to_expr_or_scalar(condition)
    if not isinstance(condition_expr, RasterExpression):
        raise MapAlgebraDTypeError(
            "where() condition must be a Boolean raster expression.",
            code="map_algebra_requires_boolean",
        )
    if condition_expr.dtype != np.dtype(np.bool_):
        raise MapAlgebraDTypeError(
            "where() condition must have boolean dtype.",
            code="map_algebra_requires_boolean",
            details={"dtype": str(condition_expr.dtype)},
        )
    branches = tuple(
        "invalid" if branch is invalid else _to_expr_or_scalar(branch)
        for branch in (x, y)
    )
    from ._dtypes import result_dtype
    from ._units import require_matching_units

    branch_expressions = [
        branch for branch in branches if isinstance(branch, RasterExpression)
    ]
    if len(branch_expressions) == 2:
        require_matching_units(
            units_a=branch_expressions[0].units,
            units_b=branch_expressions[1].units,
        )
    non_invalid_branches = tuple(
        branch for branch in branches
        if not (isinstance(branch, str) and branch == "invalid")
    )
    output_dtype = (
        result_dtype(
            tuple(branch.dtype for branch in branch_expressions),
            operation="where",
            scalars=tuple(
                branch for branch in non_invalid_branches
                if not isinstance(branch, RasterExpression)
            ),
        )
        if non_invalid_branches
        else condition_expr.dtype
    )
    return _make_expr_node(
        "local.where",
        (condition_expr, *branches),
        grid=_expression_grid(condition_expr, *branches),
        dtype=output_dtype,
        units=branch_expressions[0].units if branch_expressions else None,
    )


def coalesce(*operands: Any) -> Any:
    """Select each cell's first valid operand without FP64 intermediates.

    Raster operands must share a grid and matching units. Dtype inference is
    identical for eager and expression execution.
    """
    if not _has_expr(*operands):
        return _coalesce_eager(*operands)
    from ._model import _make_expr_node, _to_expr_or_scalar

    normalized = tuple(_to_expr_or_scalar(operand) for operand in operands)
    from ._dtypes import result_dtype
    from ._units import require_matching_units

    expression_operands = [
        operand for operand in normalized if isinstance(operand, RasterExpression)
    ]
    for operand in expression_operands[1:]:
        require_matching_units(
            units_a=expression_operands[0].units,
            units_b=operand.units,
        )
    return _make_expr_node(
        "local.coalesce",
        normalized,
        grid=_expression_grid(*normalized),
        dtype=result_dtype(
            tuple(operand.dtype for operand in expression_operands),
            operation="coalesce",
            scalars=tuple(
                operand for operand in normalized
                if not isinstance(operand, RasterExpression)
            ),
        ),
        units=expression_operands[0].units if expression_operands else None,
    )


def is_valid(raster: Any) -> Any:
    if not isinstance(raster, RasterExpression):
        return _is_valid_eager(raster)
    from ._model import _make_expr_node
    return _make_expr_node(
        "local.is_valid", (raster,), grid=raster.grid,
        dtype=np.dtype(np.bool_), units=None,
    )


def is_invalid(raster: Any) -> Any:
    if not isinstance(raster, RasterExpression):
        return _is_invalid_eager(raster)
    from ._model import _make_expr_node
    return _make_expr_node(
        "local.is_invalid", (raster,), grid=raster.grid,
        dtype=np.dtype(np.bool_), units=None,
    )


def set_invalid(raster: Any, mask: Any) -> Any:
    if not _has_expr(raster, mask):
        return _set_invalid_eager(raster, mask)
    from ._model import _make_expr_node, _to_expr_or_scalar
    raster_expr = _to_expr_or_scalar(raster)
    mask_expr = _to_expr_or_scalar(mask)
    if not isinstance(raster_expr, RasterExpression) or not isinstance(mask_expr, RasterExpression):
        raise TypeError("set_invalid() requires raster operands.")
    return _make_expr_node(
        "local.set_invalid", (raster_expr, mask_expr),
        grid=_expression_grid(raster_expr, mask_expr), dtype=raster_expr.dtype,
        units=raster_expr.units,
    )


def fill_invalid(raster: Any, value: Any) -> Any:
    """Fill invalid cells with an exactly representable value."""
    if not isinstance(raster, RasterExpression):
        return _fill_invalid_eager(raster, value)
    from ._model import _make_expr_node, _to_expr_or_scalar
    validated = _validate_nodata_representable(value, raster.dtype)
    return _make_expr_node(
        "local.fill_invalid", (raster, _to_expr_or_scalar(validated)),
        grid=raster.grid, dtype=raster.dtype, units=raster.units,
    )


def clip(raster: Any, *, lower: Any = None, upper: Any = None) -> Any:
    if not isinstance(raster, RasterExpression):
        return _clip_eager(raster, lower=lower, upper=upper)
    from ._model import _make_expr_node
    return _make_expr_node(
        "local.clip", (raster,), grid=raster.grid, dtype=raster.dtype,
        units=raster.units, params={"lower": lower, "upper": upper},
    )


def cast(
    raster: Any,
    dtype: Any,
    *,
    casting: str = "safe",
    overflow: Literal["raise", "wrap"] = "raise",
) -> Any:
    if not isinstance(raster, RasterExpression):
        return _cast_eager(raster, dtype, casting=casting, overflow=overflow)
    from ._model import _make_expr_node
    from ..errors import MapAlgebraDTypeError
    from ._dtypes import normalize_cast_overflow, normalize_dtype
    target_dtype = normalize_dtype(dtype, operation="cast")
    overflow = normalize_cast_overflow(overflow)
    if casting not in {"safe", "same_kind", "unsafe"}:
        raise MapAlgebraDTypeError(
            f"Unknown casting policy: {casting}",
            code="map_algebra_invalid_casting",
            details={"casting": casting},
        )
    if raster.dtype is not None and not np.can_cast(raster.dtype, target_dtype, casting=casting):
        raise MapAlgebraDTypeError(
            f"Cannot cast {raster.dtype} to {target_dtype} with casting='{casting}'.",
            code="map_algebra_unsafe_cast",
            details={"source_dtype": str(raster.dtype), "target_dtype": str(target_dtype)},
        )
    return _make_expr_node(
        "local.cast", (raster, target_dtype), grid=raster.grid,
        dtype=target_dtype, units=raster.units,
        params={"casting": casting, "overflow": overflow},
    )


# ---------------------------------------------------------------------------
# Expression-dispatch wrappers for classification and normalization
# ---------------------------------------------------------------------------

_reclassify_values_eager = reclassify_values
_reclassify_ranges_eager = reclassify_ranges
_digitize_eager = digitize
_one_hot_eager = one_hot
_normalize_minmax_eager = normalize_minmax
_standardize_eager = standardize


def _classification_expression_dtype(
    raster: RasterExpression,
    output_values: list[Any],
    *,
    preserve: bool,
) -> np.dtype[Any] | None:
    from .local import _classification_dtype

    if raster.dtype is None:
        return None
    fallback = raster.dtype
    return _classification_dtype(
        output_values, fallback=fallback, preserve=preserve,
    )


def reclassify_values(
    raster: Any,
    mapping: Any,
    *,
    default: Any = "invalidate",
) -> Any:
    """Map exact input values with an exact common output dtype.

    ``default="preserve"`` includes the complete source dtype domain when
    inferring the result, rather than inspecting one payload value.
    """
    if isinstance(raster, RasterExpression):
        from ._model import _make_expr_node

        default = "invalidate" if default is None else default
        mapping_items = tuple(mapping.items())
        output_values = [value for _, value in mapping_items]
        if not isinstance(default, str):
            output_values.append(default)
        return _make_expr_node(
            "local.reclassify_values",
            (raster,),
            grid=raster.grid,
            dtype=_classification_expression_dtype(
                raster, output_values, preserve=default == "preserve"
            ),
            units=raster.units,
            params={"mapping": mapping_items, "default": default},
        )
    return _reclassify_values_eager(raster, mapping, default=default)


def reclassify_ranges(
    raster: Any,
    ranges: Any,
    *,
    default: Any = "invalidate",
) -> Any:
    """Map half-open ranges with an exact common output dtype.

    ``default="preserve"`` includes the complete source dtype domain when
    inferring the result, rather than inspecting one payload value.
    """
    if isinstance(raster, RasterExpression):
        from ._model import _make_expr_node

        default = "invalidate" if default is None else default
        normalized_ranges = tuple(tuple(item) for item in ranges)
        for index, item in enumerate(normalized_ranges):
            if len(item) != 3 or not item[0] < item[1]:
                from ..errors import MapAlgebraError
                raise MapAlgebraError(
                    "Every reclassification range must have lower < upper.",
                    code="map_algebra_invalid_reclassification_range",
                    details={"index": index, "range": item},
                )
        output_values = [item[2] for item in normalized_ranges]
        if not isinstance(default, str):
            output_values.append(default)
        return _make_expr_node(
            "local.reclassify_ranges",
            (raster,),
            grid=raster.grid,
            dtype=_classification_expression_dtype(
                raster, output_values, preserve=default == "preserve"
            ),
            units=raster.units,
            params={"ranges": normalized_ranges, "default": default},
        )
    return _reclassify_ranges_eager(raster, ranges, default=default)


def digitize(raster: Any, bins: Any, *, right: bool = False) -> Any:
    if isinstance(raster, RasterExpression):
        from ._model import _make_expr_node

        normalized_bins = tuple(bins)
        bins_array = np.asarray(normalized_bins)
        if (
            bins_array.ndim != 1
            or bins_array.dtype.kind not in "iuf"
            or not np.all(bins_array[:-1] <= bins_array[1:])
        ):
            from ..errors import MapAlgebraError
            raise MapAlgebraError(
                "bins must be a one-dimensional monotonically increasing numeric sequence.",
                code="map_algebra_invalid_bins",
            )
        return _make_expr_node(
            "local.digitize",
            (raster,),
            grid=raster.grid,
            dtype=np.dtype(np.int64),
            units=None,
            params={"bins": normalized_bins, "right": right},
        )
    return _digitize_eager(raster, bins, right=right)


def one_hot(raster: Any, classes: Any) -> tuple[Any, ...]:
    normalized_classes = tuple(classes)
    if not normalized_classes:
        from ..errors import MapAlgebraError

        raise MapAlgebraError(
            "classes must contain at least one value.",
            code="map_algebra_empty_classes",
        )
    if isinstance(raster, RasterExpression):
        from ._model import _make_expr_node

        return tuple(
            _make_expr_node(
                "local.one_hot",
                (raster,),
                grid=raster.grid,
                dtype=np.dtype(np.bool_),
                units=None,
                params={"class_value": class_value},
            )
            for class_value in normalized_classes
        )
    return _one_hot_eager(raster, normalized_classes)


def normalize_minmax(
    raster: Any,
    *,
    minimum: Any = None,
    maximum: Any = None,
) -> Any:
    """Normalize valid values to [0, 1] using the shared precision policy."""
    if isinstance(raster, RasterExpression):
        from ._model import _make_expr_node
        from ._dtypes import result_dtype

        return _make_expr_node(
            "local.normalize_minmax",
            (raster,),
            grid=raster.grid,
            dtype=(
                None if raster.dtype is None else result_dtype(
                    (raster.dtype,), operation="normalize_minmax",
                    scalars=(minimum, maximum),
                )
            ),
            units=None,
            params={"minimum": minimum, "maximum": maximum},
        )
    return _normalize_minmax_eager(raster, minimum=minimum, maximum=maximum)


def standardize(
    raster: Any,
    *,
    mean: Any = None,
    std: Any = None,
    ddof: float = 0,
) -> Any:
    """Center and scale valid values using the shared precision policy."""
    if isinstance(raster, RasterExpression):
        from ._model import _make_expr_node
        from ._dtypes import result_dtype

        return _make_expr_node(
            "local.standardize",
            (raster,),
            grid=raster.grid,
            dtype=(
                None if raster.dtype is None else result_dtype(
                    (raster.dtype,), operation="standardize",
                    scalars=(mean, std),
                )
            ),
            units=None,
            params={"mean": mean, "std": std, "ddof": ddof},
        )
    return _standardize_eager(raster, mean=mean, std=std, ddof=ddof)


# ---------------------------------------------------------------------------
# Expression-dispatch wrappers for focal functions
# ---------------------------------------------------------------------------

def _wrap_focal(fn: Any, op_id: str) -> Any:
    fn_signature = signature(fn)

    @wraps(fn)
    def _wrapper(raster: Any, *args: Any, **kwargs: Any) -> Any:
        bound = fn_signature.bind(raster, *args, **kwargs)
        if isinstance(raster, RasterExpression):
            from .focal import _validate_focal_expression_parameters
            from ._model import _make_expr_node

            bound.apply_defaults()
            _validate_focal_expression_parameters(dict(bound.arguments))
            all_params = dict(kwargs)
            if args:
                all_params["_args"] = args
            return _make_expr_node(
                op_id, (raster,),
                grid=raster.grid,
                dtype=raster.dtype,
                units=raster.units,
                halo=1,
                params=all_params,
            )
        return fn(raster, *args, **kwargs)
    return _wrapper


focal_sum = _wrap_focal(focal_sum, "focal.sum")
focal_mean = _wrap_focal(focal_mean, "focal.mean")
focal_min = _wrap_focal(focal_min, "focal.min")
focal_max = _wrap_focal(focal_max, "focal.max")
focal_range = _wrap_focal(focal_range, "focal.range")
focal_std = _wrap_focal(focal_std, "focal.std")
focal_count = _wrap_focal(focal_count, "focal.count")
focal_median = _wrap_focal(focal_median, "focal.median")
dilate = _wrap_focal(dilate, "focal.dilate")
erode = _wrap_focal(erode, "focal.erode")
opening = _wrap_focal(opening, "focal.opening")
closing = _wrap_focal(closing, "focal.closing")
majority = _wrap_focal(majority, "focal.majority")
convolve = _wrap_focal(convolve, "focal.convolve")


def raster(
    values: NDArray[Any],
    georef: GeoReference,
    *,
    valid: NDArray[np.bool_] | None = None,
    nodata: int | float | None | Literal["auto"] = "auto",
    units: str | None = None,
    name: str | None = None,
    validity_provenance: str | None = None,
) -> Raster:
    """Construct a ``Raster`` from explicit values and georeferencing.

    When ``valid`` is provided it is used directly.  Otherwise validity is
    derived from ``nodata``: ``"auto"`` uses ``georef.nodata``, an explicit
    exactly representable value uses that nodata, and ``None`` marks every
    pixel valid.
    """
    if np.ma.isMaskedArray(values):
        masked = np.ma.asarray(values)
        data = np.ma.getdata(masked)
        mask = np.ma.getmaskarray(masked)
        if valid is not None:
            raise RasterValidationError(
                "Cannot supply both a masked array and an explicit valid mask.",
                code="raster_conflicting_validity",
                details={},
            )
        resolved_nodata = georef.nodata if nodata == "auto" else nodata
        nodata_valid = _valid_from_nodata(data, resolved_nodata)
        valid = nodata_valid & ~mask
        if validity_provenance is None:
            validity_provenance = "masked-array+nodata"
        return Raster(
            values=data,
            georef=georef,
            valid=valid,
            units=units,
            name=name,
            validity_provenance=validity_provenance,
        )

    values = np.asarray(values)
    if valid is not None:
        valid = np.asarray(valid, dtype=np.bool_)
        if validity_provenance is None:
            validity_provenance = "explicit-caller"
    else:
        resolved_nodata = georef.nodata if nodata == "auto" else nodata
        resolved_nodata = _validate_nodata_representable(resolved_nodata, values.dtype)
        valid = _valid_from_nodata(values, resolved_nodata)
        if validity_provenance is None:
            if resolved_nodata is not None:
                validity_provenance = "nodata"
            else:
                validity_provenance = "all_valid"
    return Raster(
        values=values,
        georef=georef,
        valid=valid,
        units=units,
        name=name,
        validity_provenance=validity_provenance,
    )


def from_masked_array(
    values: np.ma.MaskedArray[Any, Any],
    georef: GeoReference,
    *,
    units: str | None = None,
    name: str | None = None,
) -> Raster:
    """Construct a ``Raster`` from a NumPy masked array.

    The mask becomes the validity array (inverted: masked means invalid).
    """
    values = np.ma.asarray(values)
    if values.ndim != 2:
        raise RasterValidationError(
            "Masked array values must be two-dimensional.",
            code="raster_invalid_shape",
            details={"ndim": int(values.ndim)},
        )
    data = np.ma.getdata(values)
    valid = ~np.ma.getmaskarray(values)
    return Raster(
        values=data,
        georef=georef,
        valid=valid,
        units=units,
        name=name,
        validity_provenance="masked-array",
    )


def from_existing(
    values: NDArray[Any],
    georef: GeoReference,
    *,
    units: str | None = None,
    name: str | None = None,
) -> Raster:
    """Wrap existing bare ``(values, georef)`` results in a ``Raster``.

    Validity is derived from ``georef.nodata``.
    """
    values = np.asarray(values)
    nodata = _validate_nodata_representable(georef.nodata, values.dtype)
    valid = _valid_from_nodata(values, nodata)
    provenance = "nodata" if nodata is not None else "all_valid"
    return Raster(
        values=values,
        georef=georef,
        valid=valid,
        units=units,
        name=name,
        validity_provenance=provenance,
    )


def to_existing(
    raster_obj: Raster,
    *,
    nodata: int | float | None = None,
) -> tuple[NDArray[Any], GeoReference]:
    """Convert a ``Raster`` back to a bare ``(values, georef)`` tuple.

    Invalid cells are filled with an exactly representable ``nodata`` and the
    returned ``GeoReference`` carries its normalized encoding value.
    """
    validated_nodata = _validate_nodata_representable(nodata, raster_obj.values.dtype)
    if validated_nodata is not None:
        values, _validated_fill = _fill_invalid_exact(
            raster_obj.values, raster_obj.valid, validated_nodata,
        )
    else:
        values = raster_obj.values.copy()
    georef = raster_obj.georef.with_nodata(validated_nodata)
    return values, georef


def _read_rasterio_validity_provenance(
    mask_flags: list[Any] | None,
    band_idx: int,
) -> str:
    if mask_flags is None:
        return "all_valid"
    band_flags = mask_flags[band_idx]
    flags = (
        tuple(band_flags)
        if isinstance(band_flags, (list, tuple, set, frozenset))
        else (band_flags,)
    )
    names = {
        str(getattr(flag, "name", flag)).lower().split(".")[-1]
        for flag in flags
    }
    if "all_valid" in names:
        return "all_valid"
    parts = []
    if "per_dataset" in names:
        parts.append("per_dataset")
    if "per_band" in names:
        parts.append("per_band")
    if "alpha" in names:
        parts.append("alpha")
    if "nodata" in names:
        parts.append("nodata")
    return "+".join(parts) if parts else "all_valid"


def read(
    path: str | Path,
    *,
    band: int = 1,
    units: str | None = None,
    name: str | None = None,
) -> Raster:
    """Read a single-band GeoTIFF as a ``Raster``.

    Combines the GDAL band mask, dataset mask, alpha, and declared nodata
    into a canonical validity mask and preserves the native band values.
    """
    import rasterio as _rasterio

    path = Path(path).expanduser().resolve()
    values, georef = _read_geotiff(path, band=band)
    if georef is None:
        raise GeoTiffOpenError(
            "GeoTIFF is not georeferenced; cannot construct a Raster.",
            code="geotiff_unreferenced",
            details={"path": str(path)},
        )

    dataset = _rasterio.open(path)
    with dataset:
        mask_flags = dataset.mask_flag_enums if hasattr(dataset, "mask_flag_enums") else None
        read_mask: NDArray[np.bool_] | None = None
        try:
            mask_arrays = dataset.read_masks(band)
            if mask_arrays.ndim == 3 and mask_arrays.shape[0] == 1:
                mask_arrays = mask_arrays[0]
            if mask_arrays.ndim == 2:
                read_mask = np.asarray(mask_arrays, dtype=np.bool_)
        except Exception:
            read_mask = None

    flag_provenance = _read_rasterio_validity_provenance(mask_flags, band - 1)

    if read_mask is not None and flag_provenance not in ("all_valid", "nodata"):
        valid = read_mask
        provenance = flag_provenance
    elif "nodata" in flag_provenance or flag_provenance == "nodata":
        valid = _valid_from_nodata(values, georef.nodata)
        if read_mask is not None and flag_provenance != "nodata":
            valid = valid & read_mask
        provenance = flag_provenance
    else:
        valid = _valid_from_nodata(values, georef.nodata)
        provenance = flag_provenance

    georef = georef.with_nodata(None)
    raster_name = name if name is not None else path.stem
    return Raster(
        values=values,
        georef=georef,
        valid=valid,
        units=units,
        name=raster_name,
        validity_provenance="geotiff:" + provenance,
    )


# ---------------------------------------------------------------------------
# Public terrain wrappers
# ---------------------------------------------------------------------------


def slope(
    raster: Raster | RasterExpression,
    *,
    output_nodata: int | float = np.nan,
    units: str = "degrees",
    compute_edges: bool = False,
    scale: float = 1.0,
) -> Raster | RasterExpression:
    """Calculate terrain slope from an elevation raster.

    Accepts a ``Raster`` to compute the slope eagerly, or a
    ``RasterExpression`` to return a ``terrain.slope`` expression node
    that can be materialised later with ``ma.compute()`` or ``ma.write()``.

    Parameters
    ----------
    raster:
        Elevation raster or expression.
    output_nodata:
        Sentinel value stored at invalid cells. Defaults to ``NaN``.
    units:
        ``"degrees"`` (default) or ``"percent"``.
    compute_edges:
        When ``False`` (default), border pixels are marked invalid.
    scale:
        Positive horizontal-to-vertical unit ratio. Elevation is divided by
        this value before the gradient is calculated.
    """
    return _terrain_dispatch(
        "terrain.slope", raster, output_nodata=output_nodata,
        units=units, compute_edges=compute_edges, scale=scale,
    )


def aspect(
    raster: Raster | RasterExpression,
    *,
    output_nodata: int | float = np.nan,
    compute_edges: bool = False,
) -> Raster | RasterExpression:
    """Calculate terrain aspect (azimuth) from an elevation raster.

    Accepts a ``Raster`` to compute the aspect eagerly, or a
    ``RasterExpression`` to return a ``terrain.aspect`` expression node.

    Parameters
    ----------
    raster:
        Elevation raster or expression.
    output_nodata:
        Sentinel value stored at invalid cells. Defaults to ``NaN``.
    compute_edges:
        When ``False`` (default), border pixels are marked invalid.
    """
    return _terrain_dispatch(
        "terrain.aspect", raster, output_nodata=output_nodata,
        compute_edges=compute_edges,
    )


def hillshade(
    raster: Raster | RasterExpression,
    *,
    output_nodata: int | float = 0,
    azimuth: float = 315.0,
    altitude: float = 45.0,
    compute_edges: bool = False,
    scale: float = 1.0,
    z_factor: float = 1.0,
) -> Raster | RasterExpression:
    """Calculate a shaded relief raster from an elevation raster.

    Accepts a ``Raster`` to compute hillshade eagerly, or a
    ``RasterExpression`` to return a ``terrain.hillshade`` expression node.

    Parameters
    ----------
    raster:
        Elevation raster or expression.
    output_nodata:
        Sentinel value stored at invalid cells. Defaults to ``0``.
    azimuth:
        Illumination azimuth in degrees clockwise from north.  Default 315.
    altitude:
        Illumination altitude in degrees above the horizon.  Default 45.
    compute_edges:
        When ``False`` (default), border pixels are marked invalid.
    scale:
        Horizontal-to-vertical ratio.  Default 1.0.
    z_factor:
        Vertical exaggeration factor.  Default 1.0.
    """
    return _terrain_dispatch(
        "terrain.hillshade", raster, output_nodata=output_nodata,
        azimuth=azimuth, altitude=altitude,
        compute_edges=compute_edges, scale=scale,
        z_factor=z_factor,
    )


def _terrain_dispatch(
    operation_id: str,
    raster: Raster | RasterExpression,
    **parameters: Any,
) -> Raster | RasterExpression:
    if isinstance(raster, RasterExpression):
        return make_terrain_expression(operation_id, raster, **parameters)
    if isinstance(raster, Raster):
        expr = make_terrain_expression(operation_id, raster.expression(), **parameters)
        return compute(expr)
    raise TypeError(
        f"slope/aspect/hillshade expect a Raster or RasterExpression, "
        f"not {type(raster).__name__}. Use ls.slope() / ls.aspect() / "
        f"ls.hillshade() for bare NumPy arrays."
    )


# ---------------------------------------------------------------------------
# Public resampling and alignment wrappers
# ---------------------------------------------------------------------------


_CATEGORICAL_RESAMPLING = frozenset({"nearest", "mode"})


def _is_categorical_dtype(dtype: np.dtype[Any]) -> bool:
    return dtype.kind in "bi"


def resample_to(
    raster: Raster | RasterExpression,
    grid: GeoReference,
    *,
    resampling: str = "nearest",
    output_dtype: Any = None,
    validity_coverage_threshold: float | None = None,
    categorical: bool | None = None,
    allow_unsafe: bool = False,
) -> Raster | RasterExpression:
    """Resample a raster or expression onto an explicit destination grid.

    ``resample_to`` is an explicit cross-grid operation.  It is never
    inserted implicitly by other map-algebra operations.

    Parameters
    ----------
    raster:
        Source raster or expression.
    grid:
        Destination ``GeoReference`` grid.
    resampling:
        Algorithm name.  ``"nearest"`` is the default.
    output_dtype:
        Explicit output dtype.  When ``None`` the source dtype is preserved.
    validity_coverage_threshold:
        When a float between 0 and 1, each output pixel must have at least
        this fraction of valid source coverage.  When ``None``, validity
        follows the default nearest-neighbour categorical rule.
    categorical:
        When ``True``, apply categorical safety rules (only ``nearest`` and
        ``mode`` are allowed).  When ``None``, inferred from the source
        dtype: integer and Boolean rasters are treated as categorical.
    allow_unsafe:
        Suppress the categorical/interpolation safety rejection.  Use only
        when the caller understands why a resampling combination is safe
        for their specific data.
    """
    if not isinstance(grid, GeoReference):
        from ..errors import AlignmentError

        raise AlignmentError(
            "resample_to() requires a GeoReference destination grid.",
            code="alignment_invalid_destination_grid",
            details={"type": type(grid).__name__},
        )
    normalized_resampling = str(resampling).strip().lower()
    from ..alignment import _resampling_algorithm

    _resampling_algorithm(normalized_resampling)

    if isinstance(raster, RasterExpression):
        source_dtype = _raster_expression_source_dtype(raster)
        target_dtype = _resampling_output_dtype(source_dtype, output_dtype)
        _validate_resampling_safety(
            normalized_resampling,
            source_dtype=source_dtype,
            output_dtype=target_dtype,
            categorical=categorical,
            allow_unsafe=allow_unsafe,
        )
        return make_resample_expression(
            raster, grid,
            resampling=normalized_resampling,
            output_dtype=output_dtype,
            validity_coverage_threshold=validity_coverage_threshold,
        )
    if isinstance(raster, Raster):
        source_dtype = raster.dtype
        target_dtype = _resampling_output_dtype(source_dtype, output_dtype)
        _validate_resampling_safety(
            normalized_resampling,
            source_dtype=source_dtype,
            output_dtype=target_dtype,
            categorical=categorical,
            allow_unsafe=allow_unsafe,
        )
        expr = make_resample_expression(
            raster.expression(), grid,
            resampling=normalized_resampling,
            output_dtype=output_dtype,
            validity_coverage_threshold=validity_coverage_threshold,
        )
        return compute(expr)
    raise TypeError(
        f"resample_to() expects a Raster or RasterExpression, "
        f"not {type(raster).__name__}."
    )


def _raster_expression_source_dtype(
    expr: RasterExpression,
) -> np.dtype[Any]:
    if expr.dtype is not None:
        return expr.dtype
    return np.dtype(np.float64)


def _resampling_output_dtype(
    source_dtype: np.dtype[Any],
    output_dtype: Any,
) -> np.dtype[Any]:
    if output_dtype is None:
        return source_dtype
    try:
        return np.dtype(output_dtype)
    except (TypeError, ValueError) as exc:
        from ..errors import AlignmentError

        raise AlignmentError(
            "Invalid resampling output dtype.",
            code="alignment_invalid_output_dtype",
            details={"output_dtype": str(output_dtype)},
        ) from exc


def _validate_resampling_safety(
    resampling: str,
    *,
    source_dtype: np.dtype[Any],
    output_dtype: np.dtype[Any],
    categorical: bool | None,
    allow_unsafe: bool,
) -> None:
    from ..errors import AlignmentError

    supported_dtypes = {
        np.dtype(np.bool_),
        np.dtype(np.int8), np.dtype(np.uint8),
        np.dtype(np.int16), np.dtype(np.uint16),
        np.dtype(np.int32), np.dtype(np.uint32),
        np.dtype(np.int64), np.dtype(np.uint64),
        np.dtype(np.float32), np.dtype(np.float64),
    }
    if source_dtype not in supported_dtypes or output_dtype not in supported_dtypes:
        raise AlignmentError(
            "Resampling requires a supported Boolean, integer, or floating dtype.",
            code="alignment_unsupported_dtype",
            details={
                "source_dtype": str(source_dtype),
                "output_dtype": str(output_dtype),
            },
        )
    if categorical is not None and not isinstance(categorical, (bool, np.bool_)):
        raise AlignmentError(
            "categorical must be True, False, or None.",
            code="alignment_invalid_categorical_flag",
            details={"categorical": categorical},
        )
    if categorical is None:
        categorical = _is_categorical_dtype(source_dtype)

    if allow_unsafe:
        return

    if not np.can_cast(source_dtype, output_dtype, casting="safe"):
        raise AlignmentError(
            f"Output dtype {output_dtype} is not a safe conversion from {source_dtype}. "
            "Use allow_unsafe=True to override.",
            code="alignment_unsafe_output_dtype",
            details={
                "source_dtype": str(source_dtype),
                "output_dtype": str(output_dtype),
            },
        )

    if categorical and resampling not in _CATEGORICAL_RESAMPLING:
        raise AlignmentError(
            f"Resampling '{resampling}' is not safe for categorical data "
            f"(source dtype {source_dtype}). Use allow_unsafe=True to override.",
            code="alignment_unsafe_categorical_resampling",
            details={
                "resampling": resampling,
                "source_dtype": str(source_dtype),
                "allowed": sorted(_CATEGORICAL_RESAMPLING),
            },
        )

    if resampling == "mode" and not categorical:
        raise AlignmentError(
            "Resampling 'mode' is intended for categorical data. "
            "Use allow_unsafe=True to apply mode to non-categorical data.",
            code="alignment_unsafe_categorical_resampling",
            details={"resampling": "mode", "source_dtype": str(source_dtype)},
        )

    if source_dtype.kind == "b" and resampling not in _CATEGORICAL_RESAMPLING:
        raise AlignmentError(
            f"Boolean raster cannot be resampled with '{resampling}'. "
            f"Use nearest or allow_unsafe=True.",
            code="alignment_unsafe_categorical_resampling",
            details={"resampling": resampling, "source_dtype": str(source_dtype)},
        )

    if (
        not categorical
        and output_dtype.kind in "iu"
        and resampling not in _CATEGORICAL_RESAMPLING
    ):
        raise AlignmentError(
            f"Continuous resampling '{resampling}' into integer dtype "
            f"{output_dtype} can round or truncate values. Select a floating "
            "output_dtype or use allow_unsafe=True.",
            code="alignment_unsafe_integer_interpolation",
            details={
                "resampling": resampling,
                "source_dtype": str(source_dtype),
                "output_dtype": str(output_dtype),
            },
        )


def align(
    raster: Raster,
    *,
    to: GeoReference,
    resampling: str = "nearest",
    output_nodata: int | float | None | Literal["auto"] = "auto",
    output_dtype: Any = None,
    validity_coverage_threshold: float | None = None,
    categorical: bool | None = None,
    allow_unsafe: bool = False,
) -> Raster:
    """Eagerly resample a ``Raster`` onto a destination grid.

    This is the map-algebra adapter corresponding to the root-level
    ``ls.align()``.  It accepts a ``Raster`` (not a
    ``RasterExpression``) and returns a materialised ``Raster``.

    Parameters
    ----------
    raster:
        Source ``Raster`` value.
    to:
        Destination ``GeoReference`` grid.
    resampling:
        Algorithm name.  ``"nearest"`` is the default.
    output_nodata:
        ``"auto"`` preserves the source grid's nodata metadata; a numeric
        value sets destination nodata and ``None`` disables it. Canonical
        validity remains independent of this payload metadata.
    output_dtype:
        Explicit output dtype.
    validity_coverage_threshold:
        Optional minimum valid coverage fraction per output pixel.
    categorical:
        Explicit categorical flag.
    allow_unsafe:
        Suppress categorical safety rejection.
    """
    if not isinstance(raster, Raster):
        if isinstance(raster, RasterExpression):
            from ..errors import AlignmentError

            raise AlignmentError(
                "ma.align() requires a Raster. Use ma.resample_to() for "
                "RasterExpression operands.",
                code="alignment_expression_requires_resample_to",
            )
        raise TypeError(
            f"ma.align() requires a Raster, not {type(raster).__name__}."
        )
    if not isinstance(to, GeoReference):
        from ..errors import AlignmentError

        raise AlignmentError(
            "align() requires to=GeoReference.",
            code="alignment_invalid_destination_grid",
            details={"type": type(to).__name__},
        )
    if output_nodata == "auto":
        destination_nodata = raster.georef.nodata
    elif output_nodata is None or isinstance(
        output_nodata, (int, float, np.integer, np.floating),
    ):
        destination_nodata = output_nodata
    else:
        from ..errors import AlignmentError

        raise AlignmentError(
            "output_nodata must be 'auto', None, or a numeric value.",
            code="alignment_invalid_output_nodata",
            details={"output_nodata": output_nodata},
        )
    if isinstance(destination_nodata, (np.integer, np.floating)):
        destination_nodata = destination_nodata.item()
    target_dtype = _resampling_output_dtype(raster.dtype, output_dtype)
    from ..geotiff import _validate_nodata

    _validate_nodata(target_dtype, destination_nodata)
    destination_grid = to.with_nodata(destination_nodata)
    result = resample_to(
        raster,
        destination_grid,
        resampling=resampling,
        output_dtype=output_dtype,
        validity_coverage_threshold=validity_coverage_threshold,
        categorical=categorical,
        allow_unsafe=allow_unsafe,
    )
    assert isinstance(result, Raster)
    return result
