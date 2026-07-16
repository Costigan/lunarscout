from __future__ import annotations

from pathlib import Path
import struct

import numpy as np
import pytest

import lunarscout as ls
import lunarscout.scenario as scenario_module


def test_open_scenario_resolves_existing_directory(tmp_path: Path):
    root = tmp_path / "scenario"
    root.mkdir()

    scenario = ls.open_scenario(root / ".")

    assert scenario.root == root.resolve()


def test_open_scenario_rejects_missing_root(tmp_path: Path):
    missing = tmp_path / "missing"

    with pytest.raises(ls.ScenarioError) as raised:
        ls.open_scenario(missing)

    assert raised.value.code == "scenario_root_not_found"


def test_open_scenario_rejects_file_root(tmp_path: Path):
    path = tmp_path / "not-a-directory"
    path.write_text("data", encoding="utf-8")

    with pytest.raises(ls.ScenarioError) as raised:
        ls.open_scenario(path)

    assert raised.value.code == "scenario_root_not_directory"


def test_standard_scenario_paths_do_not_need_to_exist(tmp_path: Path):
    root = tmp_path / "scenario"
    root.mkdir()
    scenario = ls.open_scenario(root)

    assert scenario.root_path() == root
    assert scenario.dem_path() == root / "dem.tif"
    assert scenario.horizons_path() == root / "horizons"
    assert scenario.hillshade_path() == root / "hillshade.tif"
    assert scenario.slope_path() == root / "slope.tif"
    assert scenario.aspect_path() == root / "aspect.tif"
    assert scenario.roughness_path() == root / "roughness.tif"
    assert scenario.output_path("analysis/result.tif") == root / "analysis" / "result.tif"
    assert not (root / "horizons").exists()
    assert not (root / "analysis").exists()


def test_path_normalizes_relative_components(tmp_path: Path):
    root = tmp_path / "scenario"
    root.mkdir()
    scenario = ls.open_scenario(root)

    assert scenario.path("analysis/../dem.tif") == root / "dem.tif"
    assert scenario.path(".") == root


@pytest.mark.parametrize("method_name", ["path", "output_path"])
def test_relative_methods_reject_absolute_paths(tmp_path: Path, method_name: str):
    root = tmp_path / "scenario"
    root.mkdir()
    scenario = ls.open_scenario(root)

    with pytest.raises(ls.ScenarioPathError) as raised:
        getattr(scenario, method_name)(tmp_path / "outside.tif")

    assert raised.value.code == "scenario_absolute_path_rejected"


@pytest.mark.parametrize("relative_path", ["../outside.tif", "a/../../outside.tif"])
def test_relative_methods_reject_parent_escape(tmp_path: Path, relative_path: str):
    root = tmp_path / "scenario"
    root.mkdir()
    scenario = ls.open_scenario(root)

    with pytest.raises(ls.ScenarioPathError) as raised:
        scenario.path(relative_path)

    assert raised.value.code == "scenario_path_escape"


def test_relative_methods_reject_symlink_escape(tmp_path: Path):
    root = tmp_path / "scenario"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (root / "linked").symlink_to(outside, target_is_directory=True)
    scenario = ls.open_scenario(root)

    with pytest.raises(ls.ScenarioPathError) as raised:
        scenario.output_path("linked/result.tif")

    assert raised.value.code == "scenario_path_escape"


@pytest.mark.parametrize("relative_path", ["", "."])
def test_output_path_rejects_scenario_root(tmp_path: Path, relative_path: str):
    root = tmp_path / "scenario"
    root.mkdir()
    scenario = ls.open_scenario(root)

    with pytest.raises(ls.ScenarioPathError) as raised:
        scenario.output_path(relative_path)

    assert raised.value.code == "scenario_output_path_empty"


def test_open_scenario_rejects_attached_state_until_implemented(tmp_path: Path):
    root = tmp_path / "scenario"
    root.mkdir()

    with pytest.raises(ls.ScenarioStateError) as raised:
        ls.open_scenario(root, state=object())

    assert raised.value.code == "scenario_state_unavailable"


class _TerrainProducts:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, str, bool]] = []

    def GenerateHillshade(self, dem_path: str, output_path: str, overwrite: bool) -> None:
        self.calls.append(("hillshade", dem_path, output_path, overwrite))
        Path(output_path).write_bytes(b"hillshade")

    def GenerateSlope(self, dem_path: str, output_path: str, overwrite: bool) -> None:
        self.calls.append(("slope", dem_path, output_path, overwrite))
        Path(output_path).write_bytes(b"slope")

    def GenerateAspect(self, dem_path: str, output_path: str, overwrite: bool) -> None:
        self.calls.append(("aspect", dem_path, output_path, overwrite))
        Path(output_path).write_bytes(b"aspect")

    def GenerateRoughness(self, dem_path: str, output_path: str, overwrite: bool) -> None:
        self.calls.append(("roughness", dem_path, output_path, overwrite))
        Path(output_path).write_bytes(b"roughness")


def test_scenario_native_terrain_methods_use_canonical_paths(tmp_path: Path) -> None:
    root = tmp_path / "scenario"
    root.mkdir()
    (root / "dem.tif").write_bytes(b"dem")
    scenario = ls.open_scenario(root)
    terrain_products = _TerrainProducts()

    assert scenario.create_hillshade(_terrain_products=terrain_products) == root / "hillshade.tif"
    assert scenario.create_slope(_terrain_products=terrain_products) == root / "slope.tif"
    assert scenario.create_aspect(_terrain_products=terrain_products) == root / "aspect.tif"
    assert scenario.create_roughness(overwrite=True, _terrain_products=terrain_products) == root / "roughness.tif"

    assert terrain_products.calls == [
        ("hillshade", str(root / "dem.tif"), str(root / "hillshade.tif"), False),
        ("slope", str(root / "dem.tif"), str(root / "slope.tif"), False),
        ("aspect", str(root / "dem.tif"), str(root / "aspect.tif"), False),
        ("roughness", str(root / "dem.tif"), str(root / "roughness.tif"), True),
    ]


def test_scenario_generate_horizons_uses_primary_dem_by_default(tmp_path: Path) -> None:
    root = tmp_path / "scenario"
    root.mkdir()
    scenario = ls.open_scenario(root)
    calls: list[tuple[object, ...]] = []

    def fake_generate(output_dir, dem_paths, **kwargs):
        calls.append((output_dir, list(dem_paths), kwargs))
        return Path(output_dir)

    result = scenario.generate_horizons(_generator=fake_generate)

    assert result == root / "horizons"
    assert calls == [
        (
            root / "horizons",
            [root / "dem.tif"],
            {
                "observer_elevation": 0.0,
                "skip_existing": True,
                "compress_horizons": True,
                "disable_hierarchy": False,
                "progress_callback": None,
                "cancellation_requested": None,
            },
        )
    ]


def test_scenario_generate_horizons_prepends_primary_to_surrounding_dems(
    tmp_path: Path,
) -> None:
    root = tmp_path / "scenario"
    outside = tmp_path / "outside.tif"
    root.mkdir()
    scenario = ls.open_scenario(root)
    calls: list[tuple[object, ...]] = []

    def fake_generate(output_dir, dem_paths, **kwargs):
        calls.append((output_dir, list(dem_paths), kwargs))
        return Path(output_dir)

    progress_callback = calls.append
    scenario.generate_horizons(
        surrounding_dems=["surrounding/one.tif", outside],
        observer_elevation=2.5,
        skip_existing=False,
        compress_horizons=False,
        disable_hierarchy=True,
        progress_callback=progress_callback,
        cancellation_requested=lambda: False,
        _generator=fake_generate,
    )

    output_dir, dem_paths, kwargs = calls[0]
    assert output_dir == root / "horizons"
    assert dem_paths == [
        root / "dem.tif",
        root / "surrounding" / "one.tif",
        outside.resolve(),
    ]
    assert kwargs["observer_elevation"] == 2.5
    assert kwargs["skip_existing"] is False
    assert kwargs["compress_horizons"] is False
    assert kwargs["disable_hierarchy"] is True
    assert kwargs["progress_callback"] is progress_callback
    assert kwargs["cancellation_requested"]() is False


def test_scenario_generate_horizons_uses_explicit_dem_paths(tmp_path: Path) -> None:
    root = tmp_path / "scenario"
    root.mkdir()
    scenario = ls.open_scenario(root)
    calls: list[list[Path]] = []

    def fake_generate(_output_dir, dem_paths, **_kwargs):
        calls.append(list(dem_paths))
        return Path(_output_dir)

    scenario.generate_horizons(
        dem_paths=["custom_primary.tif", "custom_outer.tif"],
        _generator=fake_generate,
    )

    assert calls == [[root / "custom_primary.tif", root / "custom_outer.tif"]]


def test_scenario_generate_horizons_rejects_dem_paths_and_surrounding_dems(
    tmp_path: Path,
) -> None:
    root = tmp_path / "scenario"
    root.mkdir()
    scenario = ls.open_scenario(root)

    with pytest.raises(ls.NativeInputError) as raised:
        scenario.generate_horizons(
            dem_paths=["primary.tif"],
            surrounding_dems=["outer.tif"],
            _generator=lambda *_args, **_kwargs: None,
        )

    assert raised.value.code == "native_horizons_ambiguous_dem_paths"


def test_horizon_patch_coordinate_helpers_and_file_lookup(tmp_path: Path) -> None:
    root = tmp_path / "scenario"
    root.mkdir()
    scenario = ls.open_scenario(root)
    horizon_dir = root / "horizons" / "00128"
    horizon_dir.mkdir(parents=True)
    cbin = horizon_dir / "horizon_00128_00256_005.cbin"
    bin_path = horizon_dir / "horizon_00128_00256_005.bin"
    cbin.write_bytes(b"compressed")
    bin_path.write_bytes(b"raw")

    assert scenario.horizon_patch_pixel(257, 130) == (1, 2)
    assert scenario.horizon_patch_row_col(257, 130) == (1, 2)
    assert scenario.horizon_file_path(257, 130, 5) == cbin

    cbin.unlink()
    assert scenario.horizon_file_path(257, 130, 5) == bin_path
    assert scenario.horizon_file_path(257, 130, 6) is None


def test_horizon_file_lookup_supports_legacy_flat_files(tmp_path: Path) -> None:
    root = tmp_path / "scenario"
    root.mkdir()
    horizons = root / "horizons"
    horizons.mkdir()
    flat = horizons / "horizon_00000_00000_000.cbin"
    flat.write_bytes(b"compressed")
    scenario = ls.open_scenario(root)

    assert scenario.horizon_file_path(1, 2, 0) == flat


def _write_raw_horizon_file(path: Path) -> np.ndarray:
    data = np.arange(128 * 128 * 1440, dtype=np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data.astype("<f4", copy=False).tobytes())
    return data


def _encode_constant_compressed_horizon(value: float) -> bytes:
    quantized = int(round(max(-50.0, min(50.0, value)) * 32767.0 / 50.0))
    quantized = max(-32767, min(32767, quantized))
    return struct.pack(">h", quantized) + bytes(1439)


def _write_constant_compressed_horizon_file(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        for pixel_index in range(128 * 128):
            encoded = _encode_constant_compressed_horizon(float(pixel_index % 100) / 10.0)
            handle.write(len(encoded).to_bytes(2, "little"))
            handle.write(encoded)


def test_horizon_from_open_file_reads_raw_horizon(tmp_path: Path) -> None:
    path = tmp_path / "horizon_00000_00000_000.bin"
    data = _write_raw_horizon_file(path)

    with path.open("rb") as handle:
        horizon = ls.Scenario.horizon_from_open_file(handle, 1, 2)

    pixel_index = 2 * 128 + 1
    assert horizon.dtype == np.float32
    assert horizon.shape == (1440,)
    np.testing.assert_array_equal(
        horizon,
        data[pixel_index * 1440 : (pixel_index + 1) * 1440],
    )


def test_horizon_from_open_file_reads_compressed_horizon(tmp_path: Path) -> None:
    path = tmp_path / "horizon_00000_00000_000.cbin"
    _write_constant_compressed_horizon_file(path)

    with path.open("rb") as handle:
        horizon = ls.Scenario.horizon_from_open_file(handle, 3, 2)

    assert horizon.dtype == np.float32
    assert horizon.shape == (1440,)
    np.testing.assert_allclose(horizon, np.full(1440, 5.9, dtype=np.float32), atol=0.002)


def test_horizon_for_pixel_uses_single_cached_file_handle(tmp_path: Path) -> None:
    root = tmp_path / "scenario"
    root.mkdir()
    path = root / "horizons" / "00000" / "horizon_00000_00000_000.bin"
    data = _write_raw_horizon_file(path)
    scenario = ls.open_scenario(root)

    first = scenario.horizon_for_pixel(1, 2, 0)
    first_handle = scenario._horizon_file_handle
    second = scenario.horizon_for_pixel(2, 2, 0)

    assert first_handle is not None
    assert scenario._horizon_file_handle is first_handle
    np.testing.assert_array_equal(first, data[(2 * 128 + 1) * 1440 : (2 * 128 + 2) * 1440])
    np.testing.assert_array_equal(second, data[(2 * 128 + 2) * 1440 : (2 * 128 + 3) * 1440])

    scenario.close_horizon_file()
    assert first_handle.closed
    assert scenario._horizon_file_handle is None


def test_horizon_for_pixel_returns_none_for_missing_file(tmp_path: Path) -> None:
    root = tmp_path / "scenario"
    root.mkdir()
    scenario = ls.open_scenario(root)

    assert scenario.horizon_for_pixel(0, 0, 0) is None


def _scenario_georef(lunar_projection) -> ls.GeoReference:
    return ls.GeoReference(
        projection_wkt=lunar_projection[0],
        projection_proj4=lunar_projection[1],
        affine_transform=(1000.0, 20.0, 0.0, 2000.0, 0.0, -20.0),
        width=4,
        height=4,
        pixel_size_x=20.0,
        pixel_size_y=-20.0,
        nodata=None,
    )


def test_lonlat_to_dem_pixel_uses_scenario_dem_georeference(
    tmp_path: Path,
    lunar_projection,
) -> None:
    root = tmp_path / "scenario"
    root.mkdir()
    georef = _scenario_georef(lunar_projection)
    ls.write_geotiff(root / "dem.tif", np.zeros((4, 4), dtype=np.float32), georef)
    longitude, latitude = georef.pixel_to_lonlat(2.0, 1.0)
    scenario = ls.open_scenario(root)

    x, y = scenario.lonlat_to_dem_pixel(ls.LonLat(longitude, latitude))

    assert x == pytest.approx(2.0, abs=1e-8)
    assert y == pytest.approx(1.0, abs=1e-8)


def test_plot_horizon_uses_north_zero_and_east_quarter_turn(
    tmp_path: Path,
    lunar_projection,
) -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    root = tmp_path / "scenario"
    root.mkdir()
    georef = _scenario_georef(lunar_projection)
    ls.write_geotiff(root / "dem.tif", np.zeros((4, 4), dtype=np.float32), georef)
    data = _write_raw_horizon_file(
        root / "horizons" / "00000" / "horizon_00000_00000_000.bin"
    )
    longitude, latitude = georef.pixel_to_lonlat(3.0, 2.0)
    scenario = ls.open_scenario(root)

    fig, ax = scenario.plot_horizon(ls.LonLat(longitude, latitude), grid=False)

    line = ax.lines[0]
    xdata = line.get_xdata()
    ydata = line.get_ydata()
    pixel_index = 2 * 128 + 3
    horizon_base = pixel_index * 1440
    assert xdata[0] == pytest.approx(-180.0)
    assert xdata[1440 // 2] == pytest.approx(0.0)
    assert ydata[0] == pytest.approx(data[horizon_base + 720])
    assert ydata[1440 // 2] == pytest.approx(data[horizon_base])
    assert ax.get_xticklabels()[0].get_text() == "S"
    assert ax.get_xticklabels()[2].get_text() == "N"
    assert ax.get_ylabel() == "Horizon elevation (deg)"
    assert not any(gridline.get_visible() for gridline in ax.get_xgridlines())
    assert ydata.shape == (1440,)
    plt.close(fig)

    fig, ax = scenario.plot_horizon(
        ls.LonLat(longitude, latitude),
        center_azimuth=90.0,
    )
    line = ax.lines[0]
    xdata = line.get_xdata()
    ydata = line.get_ydata()
    assert xdata[0] == pytest.approx(-90.0)
    assert xdata[1440 // 2] == pytest.approx(90.0)
    assert ydata[0] == pytest.approx(data[horizon_base + 1080])
    assert ydata[1440 // 2] == pytest.approx(data[horizon_base + 360])
    assert ax.get_xticklabels()[0].get_text() == "W"
    assert ax.get_xticklabels()[2].get_text() == "E"
    plt.close(fig)


def test_body_azimuth_elevation_over_horizon_uses_scenario_horizon(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "scenario"
    root.mkdir()
    scenario = ls.open_scenario(root)
    horizon = np.arange(1440, dtype=np.float32)
    calls = []

    monkeypatch.setattr(
        scenario_module.Scenario,
        "lonlat_to_dem_pixel",
        lambda _self, _point: (1.2, 2.4),
    )

    def fake_horizon_for_pixel(_self, x, y, observer_height_decimeters):
        calls.append((x, y, observer_height_decimeters))
        return horizon

    def fake_over_horizon(point, body, times, provided_horizon, *, ensure_kernels):
        assert point == ls.LonLat(0.0, 0.0)
        assert body == "sun"
        assert list(times) == ["t0"]
        assert provided_horizon is horizon
        assert ensure_kernels is False
        return np.asarray([[90.0, 12.0]], dtype=np.float64)

    monkeypatch.setattr(
        scenario_module.Scenario,
        "horizon_for_pixel",
        fake_horizon_for_pixel,
    )
    monkeypatch.setattr(
        scenario_module,
        "body_azimuth_elevation_over_horizon",
        fake_over_horizon,
    )

    result = scenario.body_azimuth_elevation_over_horizon(
        ls.LonLat(0.0, 0.0),
        "sun",
        ["t0"],
        observer_height_decimeters=3,
        ensure_kernels=False,
    )

    np.testing.assert_array_equal(result, [[90.0, 12.0]])
    assert calls == [(1, 2, 3)]


def test_plot_body_elevations_can_fetch_scenario_horizon(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "scenario"
    root.mkdir()
    scenario = ls.open_scenario(root)
    horizon = np.arange(1440, dtype=np.float32)
    calls = []

    monkeypatch.setattr(
        scenario_module.Scenario,
        "lonlat_to_dem_pixel",
        lambda _self, _point: (1.2, 2.4),
    )

    def fake_horizon_for_pixel(_self, x, y, observer_height_decimeters):
        calls.append((x, y, observer_height_decimeters))
        return horizon

    def fake_plot_body_elevations(
        point,
        bodies,
        times,
        *,
        horizon: np.ndarray | None,
        grid: bool,
        ensure_kernels: bool,
    ):
        assert point == ls.LonLat(0.0, 0.0)
        assert list(bodies) == ["sun", "earth"]
        assert list(times) == ["t0"]
        assert horizon is horizon_array
        assert grid is False
        assert ensure_kernels is False
        return "fig", "ax"

    horizon_array = horizon
    monkeypatch.setattr(
        scenario_module.Scenario,
        "horizon_for_pixel",
        fake_horizon_for_pixel,
    )
    monkeypatch.setattr(
        scenario_module,
        "plot_body_elevations",
        fake_plot_body_elevations,
    )

    fig, ax = scenario.plot_body_elevations(
        ls.LonLat(0.0, 0.0),
        ["sun", "earth"],
        ["t0"],
        over_horizon=True,
        observer_height_decimeters=3,
        grid=False,
        ensure_kernels=False,
    )

    assert (fig, ax) == ("fig", "ax")
    assert calls == [(1, 2, 3)]


def test_plot_body_elevations_explicit_horizon_overrides_fetch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "scenario"
    root.mkdir()
    scenario = ls.open_scenario(root)
    explicit_horizon = np.arange(1440, dtype=np.float32)

    def fail_horizon_for_pixel(*_args, **_kwargs):
        raise AssertionError("scenario horizon should not be fetched")

    def fake_plot_body_elevations(
        _point,
        _bodies,
        _times,
        *,
        horizon: np.ndarray | None,
        grid: bool,
        ensure_kernels: bool,
    ):
        assert horizon is explicit_horizon
        assert grid is True
        assert ensure_kernels is True
        return "fig", "ax"

    monkeypatch.setattr(
        scenario_module.Scenario,
        "horizon_for_pixel",
        fail_horizon_for_pixel,
    )
    monkeypatch.setattr(
        scenario_module,
        "plot_body_elevations",
        fake_plot_body_elevations,
    )

    fig, ax = scenario.plot_body_elevations(
        ls.LonLat(0.0, 0.0),
        ["sun"],
        ["t0"],
        horizon=explicit_horizon,
        over_horizon=True,
    )

    assert (fig, ax) == ("fig", "ax")


def test_plot_azimuth_elevation_axes_creates_body_overlay_background(
    tmp_path: Path,
) -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    root = tmp_path / "scenario"
    root.mkdir()
    scenario = ls.open_scenario(root)

    fig, ax = scenario.plot_azimuth_elevation_axes(
        center_azimuth=90.0,
        elevation_limits=(-10.0, 20.0),
        grid=False,
    )

    assert ax.get_xlim() == pytest.approx((-90.0, 270.0))
    assert ax.get_ylim() == pytest.approx((-10.0, 20.0))
    assert ax.get_xticklabels()[0].get_text() == "W"
    assert ax.get_xticklabels()[2].get_text() == "E"
    assert not any(gridline.get_visible() for gridline in ax.get_xgridlines())
    plt.close(fig)


def test_plot_body_position_overlays_center_and_limb(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgba
    from matplotlib.patches import Ellipse

    root = tmp_path / "scenario"
    root.mkdir()
    scenario = ls.open_scenario(root)
    fig, ax = scenario.plot_azimuth_elevation_axes(center_azimuth=90.0)
    calls: list[tuple[object, ...]] = []

    def fake_angles(point, body, times, *, ensure_kernels):
        calls.append((point, body, list(times), ensure_kernels))
        return np.asarray([[90.0, 5.0]], dtype=np.float64)

    monkeypatch.setattr(scenario_module, "body_azimuth_elevation", fake_angles)

    center_artist = scenario.plot_body_position(
        ax,
        ls.LonLat(0.0, 0.0),
        "sun",
        "2027-01-01T00:00:00Z",
        ensure_kernels=False,
    )
    limb_artist = scenario.plot_body_position(
        ax,
        ls.LonLat(0.0, 0.0),
        "earth",
        "2027-01-01T00:00:00Z",
        style="limb",
        ensure_kernels=False,
    )
    override_artist = scenario.plot_body_position(
        ax,
        ls.LonLat(0.0, 0.0),
        "sun",
        "2027-01-01T00:00:00Z",
        ensure_kernels=False,
        color="black",
    )

    assert center_artist.get_xdata()[0] == pytest.approx(90.0)
    assert center_artist.get_ydata()[0] == pytest.approx(5.0)
    assert to_rgba(center_artist.get_color()) == pytest.approx(to_rgba("gold"))
    assert isinstance(limb_artist, Ellipse)
    assert limb_artist.center == pytest.approx((90.0, 5.0))
    assert limb_artist.width == pytest.approx(2.0)
    assert limb_artist.height == pytest.approx(2.0)
    assert limb_artist.get_edgecolor() == pytest.approx(to_rgba("blue"))
    assert to_rgba(override_artist.get_color()) == pytest.approx(to_rgba("black"))
    assert calls[0][1] == "sun"
    assert calls[0][3] is False
    plt.close(fig)


def test_plot_body_path_overlays_center_and_filled_limb_band(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.collections import FillBetweenPolyCollection
    from matplotlib.colors import to_rgba

    root = tmp_path / "scenario"
    root.mkdir()
    scenario = ls.open_scenario(root)
    fig, ax = scenario.plot_azimuth_elevation_axes(center_azimuth=0.0)

    def fake_angles(_point, _body, times, *, ensure_kernels):
        assert list(times) == ["t0", "t1", "t2"]
        assert ensure_kernels is False
        return np.asarray(
            [
                [350.0, 1.0],
                [10.0, 2.0],
                [90.0, 3.0],
            ],
            dtype=np.float64,
        )

    monkeypatch.setattr(scenario_module, "body_azimuth_elevation", fake_angles)

    artists = scenario.plot_body_path(
        ax,
        ls.LonLat(0.0, 0.0),
        "sun",
        ["t0", "t1", "t2"],
        style="center_and_limbs",
        label="sun",
        ensure_kernels=False,
    )

    assert len(artists) == 2
    center_line, limb_band = artists
    np.testing.assert_allclose(center_line.get_xdata(), [-10.0, 10.0, 90.0])
    np.testing.assert_allclose(center_line.get_ydata(), [1.0, 2.0, 3.0])
    assert center_line.get_color() == "gold"
    assert isinstance(limb_band, FillBetweenPolyCollection)
    assert limb_band.get_alpha() == pytest.approx(0.5)
    assert limb_band.get_linewidths()[0] == pytest.approx(0.0)
    assert limb_band.get_facecolors()[0] == pytest.approx(to_rgba("gold", 0.5))
    assert limb_band.get_label().startswith("_")
    plt.close(fig)


def test_plot_zoomed_body_path_frames_equal_scale_horizon_and_limbs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg", force=True)
    from matplotlib.collections import FillBetweenPolyCollection
    from matplotlib.colors import to_rgba
    from matplotlib.patches import Ellipse
    import matplotlib.pyplot as plt

    root = tmp_path / "scenario"
    root.mkdir()
    scenario = ls.open_scenario(root)
    horizon = np.linspace(0.0, 1.0, 1440, dtype=np.float32)
    horizon[320:401] = -2.0
    calls: list[tuple[object, ...]] = []

    monkeypatch.setattr(
        scenario_module.Scenario,
        "lonlat_to_dem_pixel",
        lambda _self, _point: (1.2, 2.4),
    )

    def fake_horizon_for_pixel(_self, x, y, observer_height_decimeters):
        calls.append(("horizon", x, y, observer_height_decimeters))
        return horizon

    def fake_angles(_point, body, times, *, ensure_kernels):
        calls.append((body, times, ensure_kernels))
        if body == "sun":
            return np.asarray(
                [
                    [80.0, 3.0],
                    [90.0, 4.0],
                    [100.0, 5.0],
                ],
                dtype=np.float64,
            )
        return np.asarray(
            [
                [85.0, 6.0],
                [95.0, 7.0],
                [105.0, 8.0],
            ],
            dtype=np.float64,
        )

    monkeypatch.setattr(
        scenario_module.Scenario,
        "horizon_for_pixel",
        fake_horizon_for_pixel,
    )
    monkeypatch.setattr(scenario_module, "body_azimuth_elevation", fake_angles)

    fig, ax = scenario.plot_zoomed_body_path(
        ls.LonLat(0.0, 0.0),
        ["sun", "earth"],
        ["t0", "t1", "t2"],
        observer_height_decimeters=3,
        grid=False,
        ensure_kernels=False,
        margin_degrees=1.0,
    )

    assert calls[0] == ("horizon", 1, 2, 3)
    assert calls[1] == ("sun", ["t0", "t1", "t2"], False)
    assert calls[2] == ("earth", ["t0", "t1", "t2"], False)
    assert ax.get_aspect() == pytest.approx(1.0)
    assert ax.get_xlim() == pytest.approx((79.0, 106.0))
    assert ax.get_ylim()[0] <= -3.0
    assert ax.get_ylim()[1] >= 10.0
    assert not any(gridline.get_visible() for gridline in ax.get_xgridlines())

    limb_bands = [
        collection
        for collection in ax.collections
        if isinstance(collection, FillBetweenPolyCollection)
    ]
    assert len(limb_bands) == 2
    assert limb_bands[0].get_alpha() == pytest.approx(0.5)
    assert limb_bands[0].get_facecolors()[0] == pytest.approx(to_rgba("gold", 0.5))
    assert limb_bands[1].get_facecolors()[0] == pytest.approx(to_rgba("blue", 0.5))

    limb_patches = [patch for patch in ax.patches if isinstance(patch, Ellipse)]
    assert len(limb_patches) == 2
    assert limb_patches[0].center == pytest.approx((80.0, 3.0))
    assert limb_patches[0].width == pytest.approx(0.536)
    assert limb_patches[0].get_alpha() == pytest.approx(1.0)
    assert limb_patches[1].center == pytest.approx((85.0, 6.0))
    assert limb_patches[1].width == pytest.approx(2.0)
    plt.close(fig)


def test_plot_zoomed_body_path_rejects_empty_body_list(tmp_path: Path) -> None:
    root = tmp_path / "scenario"
    root.mkdir()
    scenario = ls.open_scenario(root)

    with pytest.raises(ls.ScenarioPathError) as raised:
        scenario.plot_zoomed_body_path(
            ls.LonLat(0.0, 0.0),
            [],
            ["t0"],
        )

    assert raised.value.code == "scenario_body_list_empty"
