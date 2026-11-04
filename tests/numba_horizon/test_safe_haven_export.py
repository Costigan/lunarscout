"""The GeoTIFF exports of the viewer's background and monthly chain pictures."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import rasterio

from lunarscout._numba_horizon import safe_haven_export as se
from lunarscout._numba_horizon import safe_haven_viewer_data as vd
from lunarscout.errors import InputError, ProductStorageError

from test_safe_haven_viewer_data import SIZE, products, write_monthly_chains, write_psr  # noqa: F401
from test_safe_haven_overview import GT

CHAIN_TABLE = dict(chain_table=([2, 1, 0], [60.0, 600.0, 900.0], [0, 0, 1], {"2026-03": [0, 1], "2026-04": [2]}))
ENTRIES = {"2026-03": (2, [(5, 6, 1, 0), (5, 7, 2, 1), (9, 9, 3, 0), (9, 9, 3, 1), (14, 600, 2, 1), (511, 511, 1, 1)], 2),
           "2026-04": (1, [(20, 20, 1, 2), (700, 700, 2, 2)], 1)}


@pytest.fixture
def prepared(products):
    tmp_path, dem, chains = products
    write_monthly_chains(tmp_path, ENTRIES, **CHAIN_TABLE)
    psr = write_psr(tmp_path / "psr.tif", [(5, 6), (600, 700)])
    vd.prepare_viewer(tmp_path / "viewer", tmp_path / "cache", chains_path=chains, dem_path=dem, psr_path=psr)
    return tmp_path, psr


def stitched(data, month, psr_class, hop_m):
    image = np.zeros((SIZE, SIZE, 4), dtype=np.uint8)
    for ty in range(data.tiles_across(0)):
        for tx in range(data.tiles_across(0)):
            tile = data.monthly_chain_tile(month, 0, tx, ty, psr_class, hop_m)
            piece = tile[:min(512, SIZE - ty * 512), :min(512, SIZE - tx * 512)]
            image[ty * 512:ty * 512 + piece.shape[0], tx * 512:tx * 512 + piece.shape[1]] = piece
    return image


def test_the_background_export_is_the_viewers_full_resolution_background_as_a_compressed_rgb_geotiff(prepared) -> None:
    tmp_path, psr = prepared
    path = se.export_background(tmp_path / "viewer", tmp_path / "cache", psr, tmp_path / "out" / se.BACKGROUND_NAME)
    data = vd.ViewerData(tmp_path / "viewer", tmp_path / "cache", psr)
    with rasterio.open(path) as image:
        assert (image.count, image.width, image.height, str(image.dtypes[0])) == (3, SIZE, SIZE, "uint8")
        assert image.block_shapes[0] == (512, 512) and image.compression.name == "deflate"
        assert image.transform.to_gdal() == pytest.approx(GT) and image.crs is not None
        assert [c.name for c in image.colorinterp] == ["red", "green", "blue"]
        pixels = np.moveaxis(image.read(), 0, -1)
    expected = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
    for ty in range(3):
        for tx in range(3):
            tile = data.background_tile(0, tx, ty)
            piece = tile[:min(512, SIZE - ty * 512), :min(512, SIZE - tx * 512)]
            expected[ty * 512:ty * 512 + piece.shape[0], tx * 512:tx * 512 + piece.shape[1]] = piece
    assert (pixels == expected).all()
    assert not list((tmp_path / "out").glob("*.partial"))


def test_the_chain_exports_draw_only_the_chains_that_pass_both_filters_one_rgba_file_per_month(prepared) -> None:
    tmp_path, _psr = prepared
    report = se.export_monthly_chains(tmp_path / "viewer", tmp_path / "cache", tmp_path / "out", psr_class=1, hop_m=500.0)
    data = vd.ViewerData(tmp_path / "viewer", tmp_path / "cache")
    # chain 1 (class 1, hop 600) passes in 2026-03; chain 2 (class 0) fails the PSR filter in 2026-04; chain 0 fails the hop
    assert list(report.chain_files) == ["2026-03"] and report.empty_months == ["2026-04", "2026-05"]
    assert report.chains == {"2026-03": 1} and report.pixels == {"2026-03": 4}
    path = Path(report.chain_files["2026-03"])
    assert path.name == "chains_monthly_small_hop_500m_2026-03.tif"
    with rasterio.open(path) as image:
        assert (image.count, image.width, image.height) == (4, SIZE, SIZE)
        assert image.compression.name == "deflate" and image.block_shapes[0] == (512, 512)
        assert image.transform.to_gdal() == pytest.approx(GT)
        assert [c.name for c in image.colorinterp] == ["red", "green", "blue", "alpha"]
        assert image.tags()["MONTH"] == "2026-03" and image.tags()["CHAINS"] == "1"
        pixels = np.moveaxis(image.read(), 0, -1)
    assert (pixels == stitched(data, 0, 1, 500.0)).all()                       # exactly what the viewer draws at full zoom
    assert {(int(r), int(c)) for r, c in np.argwhere(pixels[..., 3] > 0)} == {(5, 7), (9, 9), (14, 600), (511, 511)}
    assert pixels[5, 7].tolist() == [0, 255, 0, 255] and pixels[0, 0].tolist() == [0, 0, 0, 0]   # no background behind it
    assert not list((tmp_path / "out").glob("*.partial"))


def test_every_filter_setting_matches_the_viewer_and_empty_months_can_be_written(prepared) -> None:
    tmp_path, _psr = prepared
    data = vd.ViewerData(tmp_path / "viewer", tmp_path / "cache")
    for psr_class, hop_m in ((0, 0.0), (2, 50.0), (0, 800.0)):
        out = tmp_path / f"out_{psr_class}_{int(hop_m)}"
        report = se.export_monthly_chains(tmp_path / "viewer", tmp_path / "cache", out, psr_class=psr_class, hop_m=hop_m,
                                          write_empty=True)
        assert sorted(report.chain_files) == ["2026-03", "2026-04", "2026-05"]                   # empty months get a file too
        for month, path in report.chain_files.items():
            with rasterio.open(path) as image:
                assert (np.moveaxis(image.read(), 0, -1) == stitched(data, data.months.index(month), psr_class, hop_m)).all()
    with rasterio.open(out / se.chain_image_name("2026-03", 0, 800.0)) as image:             # chain 2 (900 m) is not drawn in 2026-03
        assert not image.read(4).any()
    with rasterio.open(out / se.chain_image_name("2026-04", 0, 800.0)) as image:
        assert image.read(4)[20, 20] == 255


def test_the_export_names_and_inputs_are_checked(prepared, products) -> None:
    tmp_path, psr = prepared
    assert se.chain_image_name("2026-03", 2, 500.0) == "chains_monthly_medium_hop_500m_2026-03.tif"
    assert se.chain_image_name("2026-03", 0, 0.0) == "chains_monthly_any_hop_0m_2026-03.tif"
    assert se.chain_image_name("2026-03", 3, 62.5) == "chains_monthly_large_hop_62.5m_2026-03.tif"
    for bad in (dict(psr_class=4), dict(psr_class=-1), dict(hop_m=-1.0), dict(hop_m=float("nan"))):
        with pytest.raises(InputError) as error:
            se.export_monthly_chains(tmp_path / "viewer", tmp_path / "cache", tmp_path / "out", **bad)
        assert error.value.code in ("export_psr_class_invalid", "export_hop_invalid")
    with pytest.raises(InputError) as error:
        se.export_background(tmp_path / "viewer", tmp_path / "cache", tmp_path / "nothing.tif", tmp_path / "o.tif")
    assert error.value.code == "export_psr_missing"
    assert not (tmp_path / "out").exists()


def test_files_that_cannot_support_the_filters_are_refused_and_a_failed_run_keeps_the_old_file(products) -> None:
    tmp_path, dem, chains = products
    write_monthly_chains(tmp_path, {"2026-03": (2, [(5, 6, 1), (5, 7, 2)], 1)})              # an older file: no per-chain table
    vd.prepare_viewer(tmp_path / "viewer", tmp_path / "cache", chains_path=chains, dem_path=dem)
    with pytest.raises(InputError) as error:
        se.export_monthly_chains(tmp_path / "viewer", tmp_path / "cache", tmp_path / "out", psr_class=2, hop_m=0.0)
    assert error.value.code == "export_psr_unknown"
    with pytest.raises(InputError) as error:
        se.export_monthly_chains(tmp_path / "viewer", tmp_path / "cache", tmp_path / "out", psr_class=0, hop_m=500.0)
    assert error.value.code == "export_filters_unavailable"
    with pytest.raises(ProductStorageError) as error:                                         # no background pyramid in the cache
        se.export_background(tmp_path / "viewer", tmp_path / "cache", write_psr_file(tmp_path), tmp_path / "o.tif")
    assert error.value.code in ("export_background_missing", "viewer_cache_missing")


def write_psr_file(folder: Path) -> Path:
    return write_psr(folder / "other_psr.tif", [(1, 1)])


def test_the_script_runs_through_its_public_function(prepared) -> None:
    tmp_path, psr = prepared
    spec = importlib.util.spec_from_file_location("export_safe_haven_images",
                                                  Path(__file__).resolve().parents[2] / "scripts" / "export_safe_haven_images.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    result = module.export_safe_haven_images(psr_path=psr, viewer_directory=tmp_path / "viewer",
                                             cache_directory=tmp_path / "cache", output_directory=tmp_path / "exports",
                                             psr_class=1, hop_m=500.0, progress=lambda _m: None)
    assert Path(result["background"]).name == "hillshade_psr_rgb.tif" and Path(result["background"]).is_file()
    assert result["empty_months"] == ["2026-04", "2026-05"] and list(result["chain_files"]) == ["2026-03"]   # chain 1: class 1, 600 m
    args = module.build_parser().parse_args(["--psr-class", "3", "--hop-m", "1000", "--no-background", "--write-empty"])
    assert (args.psr_class, args.hop_m, args.no_background, args.write_empty) == (3, 1000.0, True, True)
    assert (args.background_compression, args.background_quality) == ("deflate", 95)
    args = module.build_parser().parse_args(["--background-compression", "jpeg", "--background-quality", "90"])
    assert (args.background_compression, args.background_quality) == ("jpeg", 90)
    jpeg = module.export_safe_haven_images(psr_path=psr, viewer_directory=tmp_path / "viewer", cache_directory=tmp_path / "cache",
                                           output_directory=tmp_path / "jpeg_exports", chains=False, background_compression="jpeg",
                                           progress=lambda _m: None)
    with rasterio.open(jpeg["background"]) as image:
        assert image.compression.name == "jpeg"


def test_the_background_can_be_lossless_webp_or_lossy_jpeg_and_the_option_is_checked(prepared) -> None:
    tmp_path, psr = prepared
    args = (tmp_path / "viewer", tmp_path / "cache", psr)
    plain = se.export_background(*args, tmp_path / "deflate.tif")
    webp = se.export_background(*args, tmp_path / "webp.tif", compression="webp")
    jpeg = se.export_background(*args, tmp_path / "jpeg.tif", compression="jpeg", jpeg_quality=95)
    with rasterio.open(plain) as image:
        expected = image.read()
    with rasterio.open(webp) as image:
        assert image.compression.name == "webp" and image.count == 3 and image.block_shapes[0] == (512, 512)
        assert (image.read() == expected).all()                                  # lossless
        assert image.transform.to_gdal() == pytest.approx(GT)
    with rasterio.open(jpeg) as image:
        assert image.compression.name == "jpeg" and image.count == 3 and image.block_shapes[0] == (512, 512)
        assert image.transform.to_gdal() == pytest.approx(GT)
        lossy = image.read().astype(int)
    assert np.abs(lossy - expected.astype(int)).mean() < 3.0                      # close, not identical
    for bad in (dict(compression="png"), dict(compression="jpeg", jpeg_quality=0), dict(compression="jpeg", jpeg_quality=101)):
        with pytest.raises(InputError) as error:
            se.export_background(*args, tmp_path / "bad.tif", **bad)
        assert error.value.code in ("export_compression_invalid", "export_jpeg_quality_invalid")
    assert not (tmp_path / "bad.tif").exists()
