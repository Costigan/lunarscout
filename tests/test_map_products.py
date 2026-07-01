from __future__ import annotations

import json
from pathlib import Path

import pytest

import lunarscout as ls


def _write_catalog(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "regions": [
                    {
                        "name": "south_pole",
                        "overview_geotiff": "/maps/south.tif",
                    }
                ],
                "products": [
                    {
                        "name": "Site04: Shackleton rim",
                        "url": "https://example.test/Site04.tif",
                        "description": "",
                        "bounds": [1, 2, 3, 4],
                        "region": "south_pole",
                    },
                    {
                        "name": "LM1: Shackleton Rim B",
                        "url": "https://example.test/LM1.tif",
                        "description": "ridge product",
                        "bounds": [5, 6, 7, 8],
                        "region": "south_pole",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )


def test_load_map_product_catalog_assigns_ids_from_order(tmp_path: Path):
    catalog_path = tmp_path / "products.json"
    _write_catalog(catalog_path)

    catalog = ls.load_map_product_catalog(catalog_path)

    assert catalog.regions["south_pole"].overview_geotiff == Path("/maps/south.tif")
    assert [product.id for product in catalog.products] == [1, 2]
    assert catalog.product(2).filename == "LM1.tif"


def test_search_map_products_matches_case_insensitive_words_in_order(tmp_path: Path):
    catalog_path = tmp_path / "products.json"
    _write_catalog(catalog_path)
    catalog = ls.load_map_product_catalog(catalog_path)

    matches = ls.search_map_products(catalog.products, "shackleton rim")
    reversed_matches = ls.search_map_products(catalog.products, "rim shackleton")

    assert [product.id for product in matches] == [1, 2]
    assert reversed_matches == []


def test_map_product_scenario_name_is_filesystem_safe(tmp_path: Path):
    catalog_path = tmp_path / "products.json"
    _write_catalog(catalog_path)
    catalog = ls.load_map_product_catalog(catalog_path)

    assert ls.map_product_scenario_name(catalog.product(1)) == "site04_shackleton_rim"
    assert (
        ls.filesystem_safe_scenario_name("NPC: Idel'son L crater 1")
        == "npc_idel_son_l_crater_1"
    )


def test_map_product_download_directory_uses_current_directory_without_options(
    tmp_path: Path,
):
    directory = ls.map_product_download_directory(
        scenario_root=None,
        scenario=None,
        environ_scenario_root=str(tmp_path / "ignored"),
        current_directory=tmp_path,
    )

    assert directory == tmp_path


def test_map_product_download_directory_requires_scenario_with_root(tmp_path: Path):
    with pytest.raises(ls.ScenarioError) as raised:
        ls.map_product_download_directory(
            scenario_root=tmp_path,
            scenario=None,
            environ_scenario_root=None,
        )

    assert raised.value.code == "scenario_name_required"


def test_map_product_download_directory_uses_environment_root(tmp_path: Path):
    scenario = tmp_path / "example"
    scenario.mkdir()

    directory = ls.map_product_download_directory(
        scenario_root=None,
        scenario="example",
        environ_scenario_root=str(tmp_path),
    )

    assert directory == scenario.resolve()


def test_map_product_download_directory_can_create_scenario(tmp_path: Path):
    directory = ls.map_product_download_directory(
        scenario_root=tmp_path,
        scenario="site04_shackleton_rim",
        environ_scenario_root=None,
        create=True,
    )

    assert directory == (tmp_path / "site04_shackleton_rim").resolve()
    assert directory.is_dir()


def test_map_product_download_directory_create_requires_existing_root(tmp_path: Path):
    with pytest.raises(ls.ScenarioError) as raised:
        ls.map_product_download_directory(
            scenario_root=tmp_path / "missing",
            scenario="site04_shackleton_rim",
            environ_scenario_root=None,
            create=True,
        )

    assert raised.value.code == "scenario_root_not_found"
