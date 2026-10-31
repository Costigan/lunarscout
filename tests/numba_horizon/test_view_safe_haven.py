"""The pole viewer script: the window's view logic, and a whole launch from the command line (offscreen)."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QPoint, QPointF, Qt  # noqa: E402
from PySide6.QtGui import QWheelEvent  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from lunarscout._numba_horizon import safe_haven_viewer_data as vd  # noqa: E402

from test_safe_haven_overview import write_dem  # noqa: E402
from test_safe_haven_viewer_data import (  # noqa: E402
    SEEDS, SIZE, write_chains, write_haven_months, write_monthly_chains, write_psr)

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "view_safe_haven.py"


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def prepared(tmp_path):
    dem = tmp_path / "dem.tif"
    write_dem(dem, SIZE, SIZE)
    chains = write_chains(tmp_path / "safe_haven_stage_4_chains.csv", SEEDS)
    vd.prepare_viewer(tmp_path / "viewer", tmp_path / "cache", chains_path=chains, dem_path=dem)
    return tmp_path, dem, chains


@pytest.fixture
def window(app, prepared):
    sys.path.insert(0, str(SCRIPT.parent))
    try:
        import view_safe_haven as viewer
    finally:
        sys.path.pop(0)
    tmp_path, _dem, _chains = prepared
    data = vd.ViewerData(tmp_path / "viewer", tmp_path / "cache")
    win = viewer.Window(data, "boxes_havens", None)
    win.resize(900, 700)
    win.show()
    app.processEvents()
    return win


def viewer_module():
    sys.path.insert(0, str(SCRIPT.parent))
    try:
        import view_safe_haven as viewer
    finally:
        sys.path.pop(0)
    return viewer


def test_the_monthly_chain_layer_is_offered_only_with_its_file(app, prepared, tmp_path_factory) -> None:
    viewer = viewer_module()
    folder, dem, chains = prepared
    data = vd.ViewerData(folder / "viewer", folder / "cache")
    with pytest.raises(ValueError, match="run_safe_haven_monthly_chain_maps"):
        viewer.Window(data, "chains_month", None)
    write_monthly_chains(folder, {"2026-03": (2, [(5, 6, 1), (5, 7, 2)], 1), "2026-05": (1, [(700, 701, 3)])})
    vd.prepare_viewer(folder / "viewer", folder / "cache", chains_path=chains, dem_path=dem)
    window = viewer.Window(vd.ViewerData(folder / "viewer", folder / "cache"), "chains_month", None)
    window.resize(900, 700)
    window.show()
    app.processEvents()
    assert [window.layers.itemText(i) for i in range(4)] == [
        "Havens and cluster boxes (monthly)", "Seed haven areas (monthly)", "Chains (monthly)", "Chains (yearly)"]
    assert window.view.layer == "chains_month" and window.caption.text() == "2026-03    2 chains (1 starting)"
    window.slider.setValue(1)
    assert window.caption.text() == "2026-04    0 chains"
    window.slider.setValue(2)
    window.view.centre = QPointF(701, 700)
    window.view.scale = 12.0
    window.view.repaint()
    assert window.view.grab().toImage().pixelColor(window.view.width() // 2 + 6, window.view.height() // 2 + 6).green() > 200
    window.layers.setCurrentIndex(3)
    assert window.view.layer == "chains" and window.caption.text() == "2026-05    no chain image for 2026"


def test_the_all_havens_layer_is_offered_only_with_its_images_and_shows_every_haven_pixel(app, prepared) -> None:
    viewer = viewer_module()
    folder, dem, chains = prepared
    data = vd.ViewerData(folder / "viewer", folder / "cache")
    with pytest.raises(ValueError, match="threshold_safe_havens"):
        viewer.Window(data, "haven_months", None)
    write_haven_months(chains, {"2026-03": {(5, 6): 0, (5, 7): 24}, "2026-05": {(700, 701): 12}})        # 0 h, 48 h; 24 h
    vd.prepare_viewer(folder / "viewer", folder / "cache", chains_path=chains, dem_path=dem)
    window = viewer.Window(vd.ViewerData(folder / "viewer", folder / "cache"), "haven_months", None)
    window.resize(900, 700)
    window.show()
    app.processEvents()
    assert [window.layers.itemText(i) for i in range(3)] == [
        "Havens and cluster boxes (monthly)", "Seed haven areas (monthly)", "All havens (monthly)"]
    assert window.view.layer == "haven_months"
    assert window.caption.text() == "2026-03    2 haven pixels (shadow period at most 48 h)"
    window.slider.setValue(1)
    assert window.caption.text() == "2026-04    no haven image for 2026-04"
    window.slider.setValue(2)
    assert window.caption.text().startswith("2026-05    1 haven pixels")
    window.view.centre = QPointF(701, 700)
    window.view.scale = 12.0
    window.view.repaint()
    image = window.view.grab().toImage()
    colour = image.pixelColor(window.view.width() // 2 + 6, window.view.height() // 2 + 6)
    assert (colour.red(), colour.green(), colour.blue()) == (56, 182, 200)        # 24 h: cyan, not the orange of a seed
    # the legend: lower left, one swatch per class in its colour, each with its label
    panel, _title, rows = window.view.legend_layout()
    assert panel.left() == 12 and abs(panel.bottom() - (window.view.height() - 12)) < 1 and len(rows) == 4
    for (swatch, _label), expected in zip(rows, [(255, 255, 255), (242, 201, 76), (56, 182, 200), (122, 92, 214)]):
        centre = image.pixelColor(int(swatch.center().x()), int(swatch.center().y()))
        assert (centre.red(), centre.green(), centre.blue()) == expected
    assert window.view.legend()[0] == "Havens: shadow period" and [t for _c, t in window.view.legend()[1]] == [
        "0 h", "up to 12 h", "12 to 24 h", "24 to 48 h"]
    window.layers.setCurrentIndex(1)
    assert window.view.layer == "havens" and window.caption.text() == "2026-05    1 havens around 1 chain starts"
    assert window.view.legend()[0] == "Havens: shadow period" and window.view.legend_layout() is not None     # and so does this one


def test_every_layer_that_draws_havens_has_the_legend_and_the_chain_layer_adds_the_start_mark(app, prepared) -> None:
    import numpy as np

    viewer = viewer_module()
    folder, dem, chains = prepared
    path = write_monthly_chains(folder, {"2026-03": (1, [(5, 6, 1)])})
    with np.load(path) as data:
        arrays = {k: data[k] for k in data.files}
    arrays["table"] = np.concatenate([arrays["table"], [[255, 0, 255, 255]]]).astype(np.uint8)
    arrays["labels"] = np.array(["a", "b", "haven", "start of a chain"])
    np.savez_compressed(path, **arrays)
    write_haven_months(chains, {"2026-03": {(5, 6): 0}})
    vd.prepare_viewer(folder / "viewer", folder / "cache", chains_path=chains, dem_path=dem)
    window = viewer.Window(vd.ViewerData(folder / "viewer", folder / "cache"), "boxes_havens", None)
    window.resize(900, 700)
    window.show()
    app.processEvents()
    shadow = ["0 h", "up to 12 h", "12 to 24 h", "24 to 48 h"]
    for index, key in enumerate(window.layer_keys):
        window.layers.setCurrentIndex(index)
        title, items = window.view.legend()
        assert title == "Havens: shadow period" and [label for _c, label in items][:4] == shadow, key
        assert [label for _c, label in items][4:] == (["start of a chain"] if key == "chains_month" else []), key
        panel, _t, rows = window.view.legend_layout()
        assert panel.left() == 12 and len(rows) == len(items) and panel.bottom() <= window.view.height()
    window.layers.setCurrentIndex(window.layer_keys.index("chains_month"))
    assert window.view.legend()[1][-1] == ((255, 0, 255), "start of a chain")


def wheel(view, x, y, delta):
    event = QWheelEvent(QPointF(x, y), QPointF(x, y), QPoint(0, 0), QPoint(0, delta), Qt.NoButton, Qt.NoModifier,
                        Qt.NoScrollPhase, False)
    view.wheelEvent(event)


def test_the_view_starts_fitted_and_reset_returns_to_the_fit(window) -> None:
    view = window.view
    fit = view.fit_scale()
    assert view.scale == pytest.approx(fit) and (view.centre.x(), view.centre.y()) == (SIZE / 2, SIZE / 2)
    wheel(view, 300, 200, 360)
    view.centre = QPointF(100, 100)
    assert view.scale > fit
    view.reset_view()
    assert view.scale == pytest.approx(fit) and (view.centre.x(), view.centre.y()) == (SIZE / 2, SIZE / 2)


def test_the_wheel_zooms_about_the_pointer_within_limits(window) -> None:
    view = window.view
    before = view._to_image(300, 200)
    wheel(view, 300, 200, 240)
    after = view._to_image(300, 200)
    assert (after.x(), after.y()) == pytest.approx((before.x(), before.y()))      # the point under the pointer stays put
    for _ in range(60):
        wheel(view, 300, 200, 120)
    assert view.scale == pytest.approx(24.0)
    for _ in range(80):
        wheel(view, 300, 200, -120)
    assert view.scale == pytest.approx(view.fit_scale() * 0.5)


def test_dragging_pans_and_the_centre_stays_on_the_pole(window, app) -> None:
    from PySide6.QtCore import QEvent
    from PySide6.QtGui import QMouseEvent

    view = window.view
    wheel(view, 450, 350, 480)                                            # zoom in so there is room to pan
    start = QPointF(view.centre)
    scale = view.scale

    def mouse(kind, x, y, button=Qt.LeftButton):
        return QMouseEvent(kind, QPointF(x, y), QPointF(x, y), button, button, Qt.NoModifier)

    view.mousePressEvent(mouse(QEvent.MouseButtonPress, 400, 300))
    view.mouseMoveEvent(mouse(QEvent.MouseMove, 300, 250))                # drag 100 left, 50 up
    view.mouseReleaseEvent(mouse(QEvent.MouseButtonRelease, 300, 250))
    assert (view.centre.x() - start.x(), view.centre.y() - start.y()) == pytest.approx((100 / scale, 50 / scale))
    view.mouseMoveEvent(mouse(QEvent.MouseMove, 0, 0))                    # no button held: no more panning
    assert (view.centre.x() - start.x()) == pytest.approx(100 / scale)
    view.centre = QPointF(-50, 99999)
    view._clamp()
    assert (view.centre.x(), view.centre.y()) == (0.0, float(SIZE))


def test_the_level_is_the_one_drawn_at_between_half_and_one_screen_pixel(window) -> None:
    view = window.view
    for scale, level in ((2.0, 0), (1.0, 0), (0.99, 0), (0.5, 1), (0.49, 1), (0.25, 2), (0.001, 2)):      # 3 levels: 0 to 2
        view.scale = scale
        assert view.level() == level
        if scale < 1.0 and level < 2:
            assert 0.5 < scale * 2 ** level <= 1.0                         # drawn at half to one screen pixel per pixel


def test_the_slider_runs_over_the_months_and_the_caption_follows_the_layer(window) -> None:
    assert window.slider.minimum() == 0 and window.slider.maximum() == 2
    assert window.caption.text() == "2026-03    1 clusters, 1 havens"
    window.slider.setValue(1)
    assert window.caption.text() == "2026-04    0 clusters, 0 havens"
    window.layers.setCurrentIndex(1)
    assert window.view.layer == "havens" and window.caption.text() == "2026-04    0 havens around 0 chain starts"
    window.layers.setCurrentIndex(2)
    assert window.view.layer == "chains" and window.caption.text() == "2026-04    no chain image for 2026"
    assert [window.layers.itemText(i) for i in range(3)] == ["Havens and cluster boxes (monthly)", "Seed haven areas (monthly)", "Chains (yearly)"]


def test_every_layer_paints_at_every_level_without_error(window, app) -> None:
    view = window.view
    for layer in range(3):
        window.layers.setCurrentIndex(layer)
        for scale in (0.02, 0.2, 0.8, 3.0, 20.0):
            view.scale = scale
            view.centre = QPointF(700, 700)
            view.repaint()
            app.processEvents()
    assert view.grab().width() > 0


def test_the_tile_cache_stays_within_its_budget(window, app, monkeypatch) -> None:
    import view_safe_haven as viewer

    monkeypatch.setattr(viewer, "TILE_BUDGET_BYTES", 3 * 2 ** 20)
    view = window.view
    for scale in (0.05, 0.3, 1.5, 6.0):
        view.scale = scale
        view.repaint()
    assert view._bytes <= 3 * 2 ** 20 + 2 ** 20 and view._bytes == sum(i.sizeInBytes() for i in view._tiles.values() if i is not False)


def run(arguments, tmp_path):
    return subprocess.run([sys.executable, str(SCRIPT), *arguments], capture_output=True, text=True, timeout=300,
                          env={**os.environ, "QT_QPA_PLATFORM": "offscreen"}, cwd=tmp_path)


def test_a_whole_launch_prepares_shows_and_saves_a_picture_of_each_layer(tmp_path) -> None:
    dem = tmp_path / "dem.tif"
    write_dem(dem, SIZE, SIZE)
    chains = write_chains(tmp_path / "safe_haven_stage_4_chains.csv", SEEDS)
    common = ["--dem-path", str(dem), "--chains-path", str(chains), "--viewer-directory", str(tmp_path / "viewer"),
              "--cache-directory", str(tmp_path / "cache"), "--psr-path", str(tmp_path / "no_psr.tif")]    # no background
    sizes = {}
    for layer in ("boxes_havens", "havens", "chains"):
        picture = tmp_path / f"{layer}.png"
        done = run([*common, "--layer", layer, "--month", "2026-05", "--screenshot", str(picture)], tmp_path)
        assert done.returncode == 0, done.stderr
        sizes[layer] = picture.stat().st_size
    assert min(sizes.values()) > 5000                                      # a real picture of the window
    assert (tmp_path / "viewer" / "viewer.json").is_file() and (tmp_path / "viewer" / "havens.npz").is_file()


def test_a_launch_that_cannot_prepare_says_why_and_exits_with_an_error(tmp_path) -> None:
    done = run(["--chains-path", str(tmp_path / "missing.csv"), "--dem-path", str(tmp_path / "missing.tif"),
                "--viewer-directory", str(tmp_path / "viewer"), "--cache-directory", str(tmp_path / "cache"),
                "--screenshot", str(tmp_path / "x.png")], tmp_path)
    assert done.returncode == 1 and "missing.csv" in done.stderr and not (tmp_path / "x.png").exists()


def test_the_psr_layer_has_a_threshold_drop_down_that_defaults_to_medium_and_applies_to_that_layer_only(app, prepared) -> None:
    viewer = viewer_module()
    folder, dem, chains = prepared
    data = vd.ViewerData(folder / "viewer", folder / "cache")
    with pytest.raises(ValueError, match="PSR_SIZES"):
        viewer.Window(data, "chains_psr", None)
    write_monthly_chains(folder, {"2026-03": (2, [(5, 6, 1), (5, 7, 2)], 2)})
    for size, pixels in (("small", [(5, 6, 1), (5, 7, 2)]), ("medium", [(5, 6, 1)])):
        write_monthly_chains(folder, {"2026-03": (len(pixels), pixels, 1)}, psr_size=size)
    vd.prepare_viewer(folder / "viewer", folder / "cache", chains_path=chains, dem_path=dem)
    window = viewer.Window(vd.ViewerData(folder / "viewer", folder / "cache"), "chains_psr", None)
    window.resize(900, 700)
    window.show()
    app.processEvents()
    assert [window.layers.itemText(i) for i in range(5)] == [
        "Havens and cluster boxes (monthly)", "Seed haven areas (monthly)", "Chains (monthly)",
        "Chains (monthly, touch PSR)", "Chains (yearly)"]
    assert [window.psr_choice.itemText(i) for i in range(window.psr_choice.count())] == ["Any PSR", "Medium or large PSR"]
    assert window.view.layer == "chains_psr" and window.view.psr_size == "medium" and window.psr_choice.isVisible()
    assert window.caption.text() == "2026-03    1 chains touching a PSR (medium or large psr), 1 starting"
    window.psr_choice.setCurrentIndex(0)
    assert window.view.psr_size == "small" and window.caption.text().startswith("2026-03    2 chains touching a PSR (any psr)")
    window.layers.setCurrentIndex(0)
    assert not window.psr_choice.isVisible()                                     # the threshold is for the one layer
    window.layers.setCurrentIndex(2)
    assert not window.psr_choice.isVisible()                                     # not on Chains (monthly) either
    window.layers.setCurrentIndex(3)
    assert window.psr_choice.isVisible()
    assert window.view.legend()[0] == "Havens: shadow period"                      # the layer names the havens' colours; no background here
    assert [label for _c, label in window.view.legend()[1]] == ["0 h", "up to 12 h", "12 to 24 h", "24 to 48 h"]


def test_the_permanent_shadow_is_drawn_light_blue_behind_every_layer_and_named_in_the_legend(app, tmp_path) -> None:
    viewer = viewer_module()
    dem = tmp_path / "dem.tif"
    write_dem(dem, SIZE, SIZE)
    chains = write_chains(tmp_path / "safe_haven_stage_4_chains.csv", SEEDS)
    psr = write_psr(tmp_path / "psr.tif", [(600, 700), (601, 700), (600, 701), (601, 701)])
    vd.prepare_viewer(tmp_path / "viewer", tmp_path / "cache", chains_path=chains, dem_path=dem, psr_path=psr)
    data = vd.ViewerData(tmp_path / "viewer", tmp_path / "cache", psr)
    window = viewer.Window(data, "havens", "2026-03")
    window.resize(900, 700)
    window.show()
    app.processEvents()
    window.view.centre = QPointF(700.5, 600.5)
    window.view.scale = 12.0
    window.view.repaint()
    centre = (window.view.width() // 2, window.view.height() // 2)
    for index, key in enumerate(window.layer_keys):
        window.layers.setCurrentIndex(index)
        window.view.repaint()
        image = window.view.grab().toImage()
        colour = image.pixelColor(*centre)
        assert colour.blue() > colour.red() + 20, key                              # light blue (30 %) in the dark hillshade
    items = window.view.legend()[1]
    assert items[-1] == ((142, 203, 255), "permanent shadow")
    far = QPointF(300.0, 300.0)
    window.view.centre = far
    window.view.repaint()
    away = window.view.grab().toImage().pixelColor(*centre)
    assert away.blue() <= away.red() + 5                                            # elsewhere only the grey hillshade


def test_a_launch_with_a_permanent_shadow_file_builds_its_pyramid_and_a_missing_file_only_warns(tmp_path) -> None:
    dem = tmp_path / "dem.tif"
    write_dem(dem, SIZE, SIZE)
    chains = write_chains(tmp_path / "safe_haven_stage_4_chains.csv", SEEDS)
    psr = write_psr(tmp_path / "psr.tif", [(600, 700), (601, 701)])
    common = ["--dem-path", str(dem), "--chains-path", str(chains), "--viewer-directory", str(tmp_path / "viewer"),
              "--cache-directory", str(tmp_path / "cache"), "--layer", "chains", "--month", "2026-05"]
    done = run([*common, "--psr-path", str(psr), "--screenshot", str(tmp_path / "with.png")], tmp_path)
    assert done.returncode == 0, done.stderr
    assert list((tmp_path / "cache").glob("background_*/done.json")) and (tmp_path / "with.png").stat().st_size > 5000
    done = run([*common, "--psr-path", str(tmp_path / "gone.tif"), "--screenshot", str(tmp_path / "without.png")], tmp_path)
    assert done.returncode == 0 and "no permanent shadow is drawn" in done.stderr
