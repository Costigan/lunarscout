from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from .errors import NativeInputError, NativeProductError

TerrainProductKind = Literal["hillshade", "slope", "aspect", "roughness"]

_METHOD_NAMES: dict[TerrainProductKind, str] = {
    "hillshade": "GenerateHillshade",
    "slope": "GenerateSlope",
    "aspect": "GenerateAspect",
    "roughness": "GenerateRoughness",
}


def _terrain_products_type() -> Any:
    from .native import _bootstrap_module

    try:
        bootstrap = _bootstrap_module()
        moonlib = bootstrap.import_moonlib(
            force_bootstrap=True,
            verify_bridge_smoke=False,
        )
        terrain_products = getattr(moonlib, "TerrainProducts", None)
        if terrain_products is None:
            raise AttributeError("moonlib.TerrainProducts is unavailable")
        return terrain_products
    except Exception as exc:
        raise NativeProductError(
            "Unable to create native terrain product components.",
            code="native_terrain_creation_failed",
            details={"error": str(exc)},
        ) from exc


def generate_terrain_product(
    kind: TerrainProductKind,
    *,
    dem_path: str | Path,
    output_path: str | Path,
    overwrite: bool = False,
    _terrain_products: Any | None = None,
) -> Path:
    dem = Path(dem_path).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    if kind not in _METHOD_NAMES:
        raise NativeInputError(
            "Unknown native terrain product kind.",
            code="native_terrain_unknown_product",
            details={"kind": kind},
        )
    if not dem.is_file():
        raise NativeInputError(
            "DEM path does not exist.",
            code="native_terrain_dem_missing",
            details={"dem_path": str(dem)},
        )
    if output.exists() and not overwrite:
        raise NativeInputError(
            "Native terrain output already exists.",
            code="native_terrain_output_exists",
            details={"output_path": str(output)},
        )

    terrain_products = (
        _terrain_products if _terrain_products is not None else _terrain_products_type()
    )
    try:
        getattr(terrain_products, _METHOD_NAMES[kind])(
            str(dem),
            str(output),
            bool(overwrite),
        )
    except NativeProductError:
        raise
    except Exception as exc:
        raise NativeProductError(
            "Native terrain product generation failed.",
            code="native_terrain_generation_failed",
            details={"kind": kind, "error": str(exc)},
        ) from exc
    return output
