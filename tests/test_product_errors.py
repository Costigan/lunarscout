from __future__ import annotations

import pytest

import lunarscout as ls


@pytest.mark.parametrize(
    ("exception_type", "parent_type", "code"),
    [
        (ls.InputError, ls.LunarscoutError, "input_error"),
        (ls.GridError, ls.InputError, "grid_error"),
        (ls.VectorError, ls.InputError, "vector_error"),
        (ls.ProductTimeError, ls.InputError, "product_time_error"),
        (ls.HorizonError, ls.LunarscoutError, "horizon_error"),
        (ls.HorizonFormatError, ls.HorizonError, "horizon_format_error"),
        (
            ls.HorizonGenerationError,
            ls.HorizonError,
            "horizon_generation_error",
        ),
        (ls.ComputeBackendError, ls.LunarscoutError, "compute_backend_error"),
        (ls.CudaError, ls.ComputeBackendError, "cuda_error"),
        (ls.ProductError, ls.LunarscoutError, "product_error"),
        (
            ls.ProductCalculationError,
            ls.ProductError,
            "product_calculation_error",
        ),
        (ls.ProductStorageError, ls.ProductError, "product_storage_error"),
        (
            ls.OperationCancelledError,
            ls.LunarscoutError,
            "operation_cancelled",
        ),
    ],
)
def test_public_product_error_taxonomy(
    exception_type: type[ls.LunarscoutError],
    parent_type: type[ls.LunarscoutError],
    code: str,
) -> None:
    error = exception_type("example", details={"field": "value"})

    assert isinstance(error, parent_type)
    assert error.code == code
    assert error.details == {"field": "value"}


def test_public_product_error_allows_stable_specific_code() -> None:
    error = ls.ProductStorageError(
        "Unable to synchronize the staged product.",
        code="product_stage_sync_failed",
        details={"path": "/tmp/product.tif"},
    )

    assert error.code == "product_stage_sync_failed"
    assert error.details == {"path": "/tmp/product.tif"}
