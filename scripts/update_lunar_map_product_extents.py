#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any


CATALOG_PATH = Path(__file__).with_name("lunar_map_products.json")


def _status(message: str) -> None:
    print(message, flush=True)


def _load_catalog(path: Path) -> dict[str, Any]:
    _status(f"Reading catalog: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _write_catalog(path: Path, catalog: dict[str, Any]) -> None:
    output = json.dumps(catalog, indent=2, ensure_ascii=True)
    path.write_text(output + "\n", encoding="utf-8")
    _status(f"Wrote updated catalog: {path}")


def _product_ids(catalog: dict[str, Any], only: list[int] | None) -> list[int]:
    products = catalog["products"]
    if only is None:
        return list(range(1, len(products) + 1))
    for product_id in only:
        if product_id < 1 or product_id > len(products):
            raise ValueError(f"Unknown product id: {product_id}")
    return only


def _vsicurl_url(url: str) -> str:
    return url if url.startswith("/vsi") else f"/vsicurl/{url}"


def _read_bounds(url: str, *, timeout_seconds: int) -> tuple[list[float], str, int, int]:
    import rasterio

    with rasterio.Env(
        GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
        CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif,.TIF",
        GDAL_HTTP_TIMEOUT=str(timeout_seconds),
        GDAL_HTTP_MAX_RETRY="2",
        GDAL_HTTP_RETRY_DELAY="1",
    ):
        with rasterio.open(_vsicurl_url(url)) as dataset:
            bounds = dataset.bounds
            return (
                [
                    float(bounds.left),
                    float(bounds.bottom),
                    float(bounds.right),
                    float(bounds.top),
                ],
                str(dataset.crs) if dataset.crs else "",
                int(dataset.width),
                int(dataset.height),
            )


def _format_bounds(bounds: list[float]) -> str:
    return (
        f"[{bounds[0]:.3f}, {bounds[1]:.3f}, "
        f"{bounds[2]:.3f}, {bounds[3]:.3f}]"
    )


def update_extents(args: argparse.Namespace) -> int:
    catalog = _load_catalog(args.catalog)
    products = catalog["products"]
    product_ids = _product_ids(catalog, args.only)

    _status(f"Products to inspect: {len(product_ids)}")
    if args.dry_run:
        _status("Dry run: catalog will not be rewritten.")
    else:
        _status("In-place update enabled: catalog will be rewritten after reads finish.")

    failures: list[tuple[int, str, str]] = []
    updates = 0
    for ordinal, product_id in enumerate(product_ids, start=1):
        product = products[product_id - 1]
        name = product["name"]
        url = product["url"]
        old_bounds = product.get("bounds")
        _status("")
        _status(f"[{ordinal}/{len(product_ids)}] Product {product_id}: {name}")
        _status(f"  URL: {url}")
        _status("  Opening GeoTIFF metadata through /vsicurl/ ...")
        try:
            bounds, crs, width, height = _read_bounds(
                url,
                timeout_seconds=args.timeout,
            )
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            failures.append((product_id, name, message))
            _status(f"  FAILED: {message}")
            if args.stop_on_error:
                break
            continue

        _status(f"  Size: {width} x {height}")
        _status(f"  CRS: {crs or '(none)'}")
        _status(f"  Old bounds: {old_bounds}")
        _status(f"  New bounds: {_format_bounds(bounds)}")
        if old_bounds != bounds:
            updates += 1
            product["bounds"] = bounds
            _status("  Status: changed")
        else:
            _status("  Status: unchanged")

        if ordinal < len(product_ids) and args.delay > 0:
            _status(f"  Waiting {args.delay:g} seconds before next product ...")
            time.sleep(args.delay)

    _status("")
    _status(f"Finished. Changed products: {updates}. Failures: {len(failures)}.")
    if failures:
        _status("Failures:")
        for product_id, name, message in failures:
            _status(f"  {product_id}: {name}: {message}")

    if not args.dry_run and updates:
        _write_catalog(args.catalog, catalog)
    elif not args.dry_run:
        _status("No catalog changes to write.")

    return 1 if failures and args.fail_on_error else 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Read GeoTIFF metadata over HTTP range requests and update lunar "
            "map product bounds."
        )
    )
    parser.add_argument(
        "--catalog",
        type=Path,
        default=CATALOG_PATH,
        help=f"catalog JSON path (default: {CATALOG_PATH})",
    )
    parser.add_argument(
        "--only",
        type=int,
        nargs="+",
        help="only inspect these product ids",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="HTTP timeout per metadata open in seconds (default: 60)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="seconds to wait between product metadata reads (default: 1)",
    )
    parser.add_argument(
        "--in-place",
        action="store_false",
        dest="dry_run",
        help="rewrite the catalog with updated bounds",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="stop after the first product metadata read failure",
    )
    parser.add_argument(
        "--fail-on-error",
        action="store_true",
        help="return a nonzero exit code if any product fails",
    )
    parser.set_defaults(dry_run=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        return update_extents(args)
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
