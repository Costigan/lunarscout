"""The seed haven areas: the public ``lunarscout.generate_seed_haven_areas``, its products and the script that wraps it."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import rasterio

import lunarscout as ls
from lunarscout.errors import InputError

from test_safe_haven_viewer_data import AREA_IMAGES, products, seed, write_chains, write_haven_months  # noqa: F401  (products is a fixture)

ROOT = Path(__file__).resolve().parents[2]


def pixels(result: dict, month: str):
    k = result["months"].index(month)
    low, high = int(result["offsets"][k]), int(result["offsets"][k + 1])
    return list(zip(result["row"][low:high].tolist(), result["column"][low:high].tolist(), result["hours"][low:high].tolist()))


def make(products, radius=200.0, images=AREA_IMAGES, **options):
    tmp_path, dem, chains = products
    if images is not None:
        write_haven_months(chains, images)
    report = ls.generate_seed_haven_areas(chains, dem, tmp_path, radius_meters=radius, **options)
    return report, ls.read_seed_areas(tmp_path / "safe_haven_stage_4_seed_haven_areas.npz")


def test_the_functions_are_in_the_public_api() -> None:
    for name in ("generate_seed_haven_areas", "read_seed_areas", "SeedHavenAreasReport"):
        assert name in ls.__all__ and hasattr(ls, name)


def test_a_months_area_is_its_chain_starts_and_the_havens_within_the_radius_of_them(products) -> None:
    report, result = make(products)
    assert pixels(result, "2026-03") == [(10, 10, 0.0), (10, 18, 6.0), (10, 20, 48.0), (18, 16, 0.0)]    # not (10, 21) or (19, 16): 11 pixels
    assert pixels(result, "2026-04") == []                                # seed 4 is a haven but starts no chain
    assert pixels(result, "2026-05") == [(700, 700, 0.0), (700, 709, 24.0), (702, 703, 12.0)]
    assert result["starts"].tolist() == [1, 0, 1] and result["with_image"].tolist() == [True, True, True]
    assert (report.months, report.chain_starts, report.pixels, report.months_with_images, report.radius_meters) == (3, 2, 7, 3, 200.0)
    assert result["radius_meters"] == 200.0


def test_the_start_keeps_its_table_hours_and_pixels_over_the_limit_or_far_away_are_not_in_the_area(products) -> None:
    _report, result = make(products)
    month = {p[:2]: p[2] for p in pixels(result, "2026-03")}
    assert month[(10, 10)] == 0.0                                          # the image says 5 counts (10 h), the chain table 0 h
    for excluded in ((10, 12), (500, 500), (20, 12), (14, 600)):           # 25 counts is over 24; far; seed 2 at 10.2 pixels; seed 3
        assert excluded not in month


def test_the_radius_is_in_metres_and_a_larger_one_reaches_further_and_clips_at_the_dem_edge(products) -> None:
    assert [p[:2] for p in pixels(make(products, 0.0)[1], "2026-03")] == [(10, 10)]       # no radius: the start alone
    wide = [p[:2] for p in pixels(make(products, 220.0, overwrite=True)[1], "2026-03")]
    assert (20, 12) in wide and (10, 21) in wide and (10, 12) not in wide                  # 11 pixels reach seed 2 at 10.2
    huge = [p[:2] for p in pixels(make(products, 2000.0, overwrite=True)[1], "2026-03")]   # 100 pixels: the window passes the DEM edge
    assert (10, 10) in huge and (19, 16) in huge and (500, 500) not in huge


def test_a_month_without_a_haven_image_has_its_chain_starts_alone(products) -> None:
    report, result = make(products, images={"2026-03": {(10, 18): 3}})
    assert pixels(result, "2026-03") == [(10, 10, 0.0), (10, 18, 6.0)]
    assert pixels(result, "2026-05") == [(700, 700, 0.0)] and result["with_image"].tolist() == [True, False, False]
    assert report.months_with_images == 1
    _report, none = make(products, images=None, overwrite=True)            # the images of the run before stay beside the table
    assert none["with_image"].any()


def test_the_geotiff_of_a_month_holds_the_counts_with_a_palette_and_overviews_that_keep_a_haven(products) -> None:
    tmp_path, _dem, _chains = products
    make(products)
    path = tmp_path / "safe_haven_stage_4_seed_haven_areas" / "safe_haven_stage_4_seed_haven_areas_2026-03.tif"
    with rasterio.open(path) as image:
        data = image.read(1)
        assert image.dtypes == ("uint8",) and image.nodata == 255 and image.crs is not None
        assert np.argwhere(data != 255).tolist() == [[10, 10], [10, 18], [10, 20], [18, 16]]
        assert data[10, 10] == 0 and data[10, 18] == 3 and data[10, 20] == 24            # counts of 2 h; the start is 0 h
        palette = image.colormap(1)
        assert palette[0] == (255, 255, 255, 255) and palette[3] == (242, 201, 76, 255) and palette[6] == (242, 201, 76, 255)
        assert palette[7] == (56, 182, 200, 255) and palette[12] == (56, 182, 200, 255)                    # 12 h, 24 h
        assert palette[13] == (122, 92, 214, 255) and palette[24] == (122, 92, 214, 255) and palette[255][3] == 0
        tags = image.tags()
        assert tags["SAFE_HAVEN_AREA_RADIUS_METERS"] == "200" and tags["SAFE_HAVEN_AREA_STARTS"] == "1"
        assert tags["SAFE_HAVEN_PIXELS"] == "4" and tags["SAFE_HAVEN_MONTH"] == "2026-03"
        assert image.overviews(1) == [2, 4]
    with rasterio.open(path, overview_level=0) as overview:
        level1 = overview.read(1)
    assert level1[5, 5] == 0 and level1[5, 9] == 3 and level1[5, 10] == 24 and level1[9, 8] == 0 and (level1 != 255).sum() == 4
    with rasterio.open(path, overview_level=1) as overview:
        assert (overview.read(1) != 255).sum() == 4                         # (2, 2), (2, 4), (2, 5), (4, 4): none lost


def test_an_existing_product_is_refused_without_overwrite_and_a_failure_leaves_it_untouched(products) -> None:
    tmp_path, dem, chains = products
    make(products)
    before = (tmp_path / "safe_haven_stage_4_seed_haven_areas.npz").read_bytes()
    with pytest.raises(InputError) as error:
        ls.generate_seed_haven_areas(chains, dem, tmp_path)
    assert error.value.code == "seed_areas_output_exists"
    with pytest.raises(InputError) as error:
        ls.generate_seed_haven_areas(chains, dem, tmp_path, radius_meters=-1.0, overwrite=True)
    assert error.value.code == "seed_areas_setting_invalid"
    with pytest.raises(InputError) as error:
        ls.generate_seed_haven_areas(chains, dem, tmp_path, hours_per_count=1.0, overwrite=True)    # the images count 2 h
    assert error.value.code == "seed_areas_hours_per_count_mismatch"
    assert (tmp_path / "safe_haven_stage_4_seed_haven_areas.npz").read_bytes() == before
    assert not list(tmp_path.glob("*.partial"))


def test_a_file_that_is_not_a_monthly_haven_image_is_refused(products) -> None:
    tmp_path, dem, chains = products
    write_haven_months(chains, AREA_IMAGES)
    path = tmp_path / "safe_haven_stage_1_haven_months" / "safe_haven_stage_1_haven_months_2026-03.tif"
    with rasterio.open(path) as image:                                      # the same pixels without the tags
        data, profile = image.read(1), image.profile
    with rasterio.open(path, "w", **profile) as image:
        image.write(data, 1)
    with pytest.raises(InputError) as error:
        ls.generate_seed_haven_areas(chains, dem, tmp_path)
    assert error.value.code == "seed_areas_haven_image_invalid"


def test_a_fresh_process_makes_the_areas_through_the_public_import(products) -> None:
    tmp_path, dem, chains = products
    write_haven_months(chains, AREA_IMAGES)
    code = ("import sys; sys.path.insert(0, %r); import json, lunarscout as ls; "
            "r = ls.generate_seed_haven_areas(%r, %r, %r, radius_meters=200.0); print(json.dumps(r.as_dict()))"
            % (str(ROOT / "src"), str(chains), str(dem), str(tmp_path)))
    run = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, cwd=tmp_path, check=False)
    assert run.returncode == 0, run.stderr
    report = json.loads(run.stdout.strip().splitlines()[-1])
    assert (report["months"], report["chain_starts"], report["pixels"]) == (3, 2, 7)


def test_the_script_has_the_constants_as_options_and_runs_the_function(products) -> None:
    spec = importlib.util.spec_from_file_location("make_seed_haven_areas", ROOT / "scripts" / "make_seed_haven_areas.py")
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    assert script.default_options()["area_radius_meters"] == 200.0 and script.AREA_RADIUS_METERS == 200.0
    assert script.build_parser().parse_args(["--area-radius-meters", "300"]).area_radius_meters == 300.0
    tmp_path, dem, chains = products
    write_haven_months(chains, AREA_IMAGES)
    assert script.main(["--dem-path", str(dem), "--chains-path", str(chains), "--output-directory", str(tmp_path),
                        "--haven-months-directory", str(tmp_path / "safe_haven_stage_1_haven_months"),
                        "--area-radius-meters", "220", "--color", "never"]) == 0
    assert (20, 12) in [p[:2] for p in pixels(ls.read_seed_areas(tmp_path / "safe_haven_stage_4_seed_haven_areas.npz"), "2026-03")]


def test_a_continuation_starts_an_area_in_each_later_month_and_a_lone_last_seed_is_its_pixel_alone(products) -> None:
    # one chain of three seeds A, B, C of 2026-03, 04 and 05: B starts the chain in 2026-04 and C is alone in 2026-05
    tmp_path, dem, chains = products
    start = "2026-03-05T00:00:00.000000Z"
    write_chains(chains, [seed(1, "2026-03", 10, 10, 1, 0, start, 0.0), seed(2, "2026-04", 20, 12, 1, 1, start, 6.0, 2),
                          seed(3, "2026-05", 14, 600, 1, 2, start, 20.0, 3)])
    write_haven_months(chains, {"2026-03": {(10, 12): 2}, "2026-04": {(22, 14): 3, (20, 40): 3}, "2026-05": {(14, 603): 1}})
    report = ls.generate_seed_haven_areas(chains, dem, tmp_path)
    result = ls.read_seed_areas(tmp_path / "safe_haven_stage_4_seed_haven_areas.npz")
    assert pixels(result, "2026-03") == [(10, 10, 0.0), (10, 12, 4.0)]                  # A, and a haven near it
    assert pixels(result, "2026-04") == [(20, 12, 6.0), (22, 14, 6.0)]                  # B, the continuation: (20, 40) is too far
    assert pixels(result, "2026-05") == [(14, 600, 20.0)]                               # C alone: the haven at (14, 603) is not added
    assert result["starts"].tolist() == [1, 1, 0] and result["alone"].tolist() == [0, 0, 1]
    assert (report.chain_starts, report.lone_seeds, report.pixels) == (2, 1, 5)


def test_without_haven_images_the_areas_are_the_current_starts_and_lone_seeds_alone(products) -> None:
    tmp_path, dem, chains = products
    start = "2026-03-05T00:00:00.000000Z"
    write_chains(chains, [seed(1, "2026-03", 10, 10, 1, 0, start, 0.0), seed(2, "2026-04", 20, 12, 1, 1, start, 6.0, 2),
                          seed(3, "2026-05", 14, 600, 1, 2, start, 20.0, 3)])
    ls.generate_seed_haven_areas(chains, dem, tmp_path)
    result = ls.read_seed_areas(tmp_path / "safe_haven_stage_4_seed_haven_areas.npz")
    assert [pixels(result, m) for m in ("2026-03", "2026-04", "2026-05")] == [
        [(10, 10, 0.0)], [(20, 12, 6.0)], [(14, 600, 20.0)]] and result["with_image"].tolist() == [False] * 3
