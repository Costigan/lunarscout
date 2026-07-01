from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
import json
from pathlib import Path
import re
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from .errors import OutputExistsError, ProductCatalogError, ScenarioError
from .scenario import open_scenario


ProgressCallback = Callable[[int, int | None], None]


@dataclass(frozen=True, slots=True)
class MapRegion:
    name: str
    overview_geotiff: Path | None


@dataclass(frozen=True, slots=True)
class MapProduct:
    id: int
    name: str
    url: str
    description: str
    bounds: tuple[float, float, float, float]
    region: str

    @property
    def filename(self) -> str:
        name = Path(urlparse(self.url).path).name
        if not name:
            raise ProductCatalogError(
                "Product URL does not contain a filename.",
                code="product_url_missing_filename",
                details={"id": self.id, "url": self.url},
            )
        return name


@dataclass(frozen=True, slots=True)
class MapProductCatalog:
    regions: dict[str, MapRegion]
    products: tuple[MapProduct, ...]

    def product(self, product_id: int) -> MapProduct:
        if product_id < 1 or product_id > len(self.products):
            raise ProductCatalogError(
                "Unknown product id.",
                code="product_id_not_found",
                details={"id": product_id},
            )
        return self.products[product_id - 1]


def load_map_product_catalog(path: str | Path) -> MapProductCatalog:
    catalog_path = Path(path)
    try:
        raw = json.loads(catalog_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ProductCatalogError(
            "Product catalog is not valid JSON.",
            code="product_catalog_invalid_json",
            details={"path": str(catalog_path), "error": str(exc)},
        ) from exc
    except OSError as exc:
        raise ProductCatalogError(
            "Product catalog could not be read.",
            code="product_catalog_read_error",
            details={"path": str(catalog_path), "error": str(exc)},
        ) from exc

    try:
        region_items = raw["regions"]
        product_items = raw["products"]
    except KeyError as exc:
        raise ProductCatalogError(
            "Product catalog is missing a required top-level key.",
            code="product_catalog_missing_key",
            details={"path": str(catalog_path), "key": str(exc)},
        ) from exc

    regions: dict[str, MapRegion] = {}
    for item in region_items:
        name = str(item["name"])
        overview_text = str(item.get("overview_geotiff", ""))
        regions[name] = MapRegion(
            name=name,
            overview_geotiff=Path(overview_text) if overview_text else None,
        )

    products: list[MapProduct] = []
    for index, item in enumerate(product_items, start=1):
        bounds = item["bounds"]
        if not isinstance(bounds, Sequence) or len(bounds) != 4:
            raise ProductCatalogError(
                "Product bounds must contain four numbers.",
                code="product_bounds_invalid",
                details={"index": index},
            )
        region = str(item["region"])
        if region not in regions:
            raise ProductCatalogError(
                "Product references an unknown region.",
                code="product_region_unknown",
                details={"index": index, "region": region},
            )
        products.append(
            MapProduct(
                id=index,
                name=str(item["name"]),
                url=str(item["url"]),
                description=str(item.get("description", "")),
                bounds=tuple(float(value) for value in bounds),  # type: ignore[arg-type]
                region=region,
            )
        )

    return MapProductCatalog(regions=regions, products=tuple(products))


def search_map_products(
    products: Iterable[MapProduct],
    text_pattern: str,
) -> list[MapProduct]:
    words = text_pattern.casefold().split()
    if not words:
        return list(products)

    matches: list[MapProduct] = []
    for product in products:
        haystack = f"{product.name} {product.description}".casefold()
        position = 0
        for word in words:
            next_position = haystack.find(word, position)
            if next_position < 0:
                break
            position = next_position + len(word)
        else:
            matches.append(product)
    return matches


def map_product_scenario_name(product: MapProduct) -> str:
    return filesystem_safe_scenario_name(product.name)


def filesystem_safe_scenario_name(name: str) -> str:
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", name.strip()).strip("._-")
    safe_name = re.sub(r"_+", "_", safe_name)
    if not safe_name:
        raise ProductCatalogError(
            "Product name cannot be converted to a scenario directory name.",
            code="product_scenario_name_empty",
            details={"name": name},
        )
    return safe_name.lower()


def map_product_download_directory(
    *,
    scenario_root: str | Path | None,
    scenario: str | None,
    environ_scenario_root: str | None,
    current_directory: str | Path = ".",
    create: bool = False,
) -> Path:
    if scenario is None and scenario_root is None:
        return Path(current_directory)
    if scenario is None:
        raise ScenarioError(
            "--scenario is required when --scenario-root is specified.",
            code="scenario_name_required",
        )
    root_text = str(scenario_root or environ_scenario_root or "")
    if not root_text:
        raise ScenarioError(
            "--scenario-root is required when --scenario is specified and "
            "LUNARSCOUT_SCENARIO_ROOT is unset.",
            code="scenario_root_required",
        )
    root = Path(root_text).expanduser()
    if create:
        try:
            root = root.resolve(strict=True)
        except (OSError, RuntimeError, ValueError) as exc:
            raise ScenarioError(
                "Scenario root does not exist or cannot be resolved.",
                code="scenario_root_not_found",
                details={"path": str(root), "error": str(exc)},
            ) from exc
        if not root.is_dir():
            raise ScenarioError(
                "Scenario root must be a directory.",
                code="scenario_root_not_directory",
                details={"path": str(root)},
            )
    scenario_path = root / scenario
    if create:
        scenario_path.mkdir(parents=True, exist_ok=True)
    opened = open_scenario(scenario_path)
    return opened.root


def download_map_product(
    product: MapProduct,
    directory: str | Path,
    *,
    overwrite: bool = False,
    progress_callback: ProgressCallback | None = None,
) -> Path:
    output_path = Path(directory) / product.filename
    if output_path.exists() and not overwrite:
        raise OutputExistsError(
            "Output file already exists.",
            code="download_output_exists",
            details={"path": str(output_path)},
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(output_path.name + ".part")

    request = Request(product.url, headers={"User-Agent": "lunarscout/0.1"})
    try:
        with urlopen(request) as response, temporary_path.open("wb") as output:
            length_header = response.headers.get("Content-Length")
            total = int(length_header) if length_header else None
            received = 0
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                output.write(chunk)
                received += len(chunk)
                if progress_callback is not None:
                    progress_callback(received, total)
        temporary_path.replace(output_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
    if progress_callback is not None:
        size = output_path.stat().st_size
        progress_callback(size, size)
    return output_path
