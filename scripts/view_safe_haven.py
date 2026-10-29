#!/usr/bin/env python3
"""A window onto the pole: the dark hillshade with the safe-haven products of one month on it.

    .venv/bin/python scripts/view_safe_haven.py [--layer havens --month 2027-03 --rebuild ...]

Needs PySide6 (`pip install "lunarscout[viewer]"`), the stage-4 chain table and the yearly chain images
(scripts/run_safe_haven_chain_maps.py and run_safe_haven_yearly_chain_maps.py), and the DEM. Edit the constants below or
give them as options (--constant-name). CPU only; no NVIDIA device is used.

Controls: drag with the left mouse button to pan, turn the wheel to zoom about the pointer (there are no scroll bars),
use the slider (or its keyboard arrows) to pick the month, the drop-down to pick what is drawn, and Reset view to fit the
whole pole.

The drop-down chooses the layer:

* Cluster boxes (monthly): the bounding box of each group of connected havens whose chain starts in the month, coloured
  by the havens in the group (the aqua ramp of safe_haven_stage_4_cluster_boxes_<year>.png);
* Havens (monthly): one pixel for each haven (seed) of the month;
* Chains (monthly): the longest chain of each root that starts in the month, as the yearly image draws it (the same
  routes and colours; the havens white), from safe_haven_stage_4_monthly_chain_pixels.npz
  (scripts/run_safe_haven_monthly_chain_maps.py). The entry is in the drop-down only when that file is beside the
  chain table;
* Chains (yearly): the yearly chain image (safe_haven_stage_4_longest_chain_per_root_<year>.tif) of the slider's year,
  so the picture changes only when the slider crosses into another year.

The first run prepares what the window draws: viewer.json and havens.npz in VIEWER_DIRECTORY (the boxes of every month,
as pregenerated lists) and image pyramids of the hillshade and of each yearly chain image in CACHE_DIRECTORY (about
10 GB, a few minutes; later runs find them and start at once). A pyramid is rebuilt only when its source file changes;
--rebuild makes the lists again. Boxes and haven dots are drawn from the lists when a month is shown.
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import OrderedDict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

try:
    from PySide6.QtCore import QPointF, QRectF, Qt, QThread, Signal
    from PySide6.QtGui import QColor, QImage, QPainter
    from PySide6.QtWidgets import (
        QApplication, QComboBox, QHBoxLayout, QLabel, QMainWindow, QProgressBar, QPushButton, QSlider, QVBoxLayout,
        QWidget,
    )
except ImportError as error:                                               # pragma: no cover - needs the extra
    raise SystemExit(f'view_safe_haven.py needs PySide6: pip install "lunarscout[viewer]" ({error})')

from lunarscout._numba_horizon.safe_haven_viewer_data import TILE, ViewerData, prepare_viewer  # noqa: E402

# Edit these paths before running.
DEM_PATH = Path("/d/viper/maps/lola/ldem_80s_20m.clipped.tif")
STAGE_DIRECTORY = Path("/e/lunar_analyst_scenarios/70south/safe_havens")
CHAINS_PATH = STAGE_DIRECTORY / "safe_haven_stage_4_chains.csv"
EDGES_PATH = STAGE_DIRECTORY / "safe_haven_stage_3_edges.csv"       # needed only for CONNECT = "edges"
VIEWER_DIRECTORY = STAGE_DIRECTORY / "viewer"
CACHE_DIRECTORY = Path("/tmp/lunarscout_viewer")
CONNECT = "chains"            # what joins two havens in a cluster: "chains" (stage 4's chains) or "edges" (stage-3 edges)

LAYERS = (("Cluster boxes (monthly)", "boxes"), ("Havens (monthly)", "havens"), ("Chains (monthly)", "chains_month"),
          ("Chains (yearly)", "chains"))
BACKGROUND = "#0d0d0d"
TILE_BUDGET_BYTES = 512 * 2 ** 20
MAX_SCALE = 24.0                # screen pixels per DEM pixel when fully zoomed in


class MapView(QWidget):
    """The pole as one bitmap: tiles of the pyramid are drawn at the zoom, panned by dragging, zoomed by the wheel."""

    def __init__(self, data: ViewerData):
        super().__init__()
        self.data = data
        self.layer = "boxes"
        self.month = 0
        self.centre = QPointF(data.size / 2, data.size / 2)       # in DEM pixels
        self.scale = 0.05                                          # screen pixels per DEM pixel
        self._moved = False                                        # the user has panned or zoomed: keep the view on resizes
        self._drag = None
        self._tiles: OrderedDict = OrderedDict()
        self._bytes = 0
        self.setMinimumSize(480, 360)
        self.setCursor(Qt.OpenHandCursor)

    # -- the view ----------------------------------------------------------------------------------------------

    def fit_scale(self) -> float:
        return max(min(self.width(), self.height()), 1) / self.data.size

    def reset_view(self) -> None:
        self.centre = QPointF(self.data.size / 2, self.data.size / 2)
        self.scale = self.fit_scale()
        self._moved = False
        self.update()

    def resizeEvent(self, event) -> None:
        if not self._moved:
            self.reset_view()

    def _to_image(self, x: float, y: float) -> QPointF:
        return QPointF(self.centre.x() + (x - self.width() / 2) / self.scale, self.centre.y() + (y - self.height() / 2) / self.scale)

    def wheelEvent(self, event) -> None:
        position = event.position()
        before = self._to_image(position.x(), position.y())
        factor = 1.2 ** (event.angleDelta().y() / 120.0)
        self.scale = min(max(self.scale * factor, self.fit_scale() * 0.5), MAX_SCALE)
        self.centre = QPointF(before.x() - (position.x() - self.width() / 2) / self.scale,
                              before.y() - (position.y() - self.height() / 2) / self.scale)
        self._clamp()
        self._moved = True
        self.update()

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            self._drag = (event.position(), QPointF(self.centre))
            self.setCursor(Qt.ClosedHandCursor)

    def mouseMoveEvent(self, event) -> None:
        if self._drag is not None:
            start, centre = self._drag
            delta = event.position() - start
            self.centre = QPointF(centre.x() - delta.x() / self.scale, centre.y() - delta.y() / self.scale)
            self._clamp()
            self._moved = True
            self.update()

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            self._drag = None
            self.setCursor(Qt.OpenHandCursor)

    def _clamp(self) -> None:
        size = self.data.size
        self.centre = QPointF(min(max(self.centre.x(), 0.0), size), min(max(self.centre.y(), 0.0), size))

    # -- what is drawn -----------------------------------------------------------------------------------------

    def set_layer(self, layer: str) -> None:
        self.layer = layer
        self.update()

    def set_month(self, month: int) -> None:
        self.month = month
        self.update()

    def level(self) -> int:
        """The pyramid level whose pixels are drawn at between half and one screen pixel (full size when zoomed in)."""
        if self.scale >= 1.0:
            return 0
        return min(self.data.levels - 1, int(math.floor(math.log2(1.0 / self.scale) + 1e-9)))

    def _remember(self, key, image):
        old = self._tiles.pop(key, None)
        if old is not None:
            self._bytes -= old.sizeInBytes() if old is not False else 0
        self._tiles[key] = image
        self._bytes += image.sizeInBytes() if image is not False else 0
        while self._bytes > TILE_BUDGET_BYTES and len(self._tiles) > 1:
            _key, dropped = self._tiles.popitem(last=False)
            self._bytes -= dropped.sizeInBytes() if dropped is not False else 0
        return image

    def _cached(self, key, make):
        image = self._tiles.get(key)
        if image is None:
            return self._remember(key, make())
        self._tiles.move_to_end(key)
        return image

    def _hillshade(self, level: int, tx: int, ty: int):
        def make():
            array = self.data.hillshade_tile(level, tx, ty)
            return QImage(array.data, TILE, TILE, TILE, QImage.Format_Grayscale8).copy()
        return self._cached(("hillshade", level, tx, ty), make)

    def _overlay(self, level: int, tx: int, ty: int):
        """The QImage of the layer's tile, or False when it has nothing to draw."""
        if self.layer == "chains":
            year = int(self.data.months[self.month][:4])
            source = ("chains", year)
            tile = lambda: self.data.chain_tile(year, level, tx, ty)             # noqa: E731
        elif self.layer == "chains_month":
            source = ("chains_month", self.month)
            tile = lambda: self.data.monthly_chain_tile(self.month, level, tx, ty)  # noqa: E731
        elif self.layer == "havens":
            source = ("havens", self.month)
            tile = lambda: self.data.havens_tile(self.month, level, tx, ty)      # noqa: E731
        else:
            source = ("boxes", self.month)
            tile = lambda: self.data.boxes_tile(self.month, level, tx, ty)       # noqa: E731

        def make():
            array = tile()
            if array is None or not array[..., 3].any():
                return False
            return QImage(array.data, TILE, TILE, TILE * 4, QImage.Format_RGBA8888).copy()
        return self._cached((*source, level, tx, ty), make)

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(BACKGROUND))
        level = self.level()
        step = 2 ** level
        shown = self.scale * step                                  # screen pixels per pixel of this level
        painter.setRenderHint(QPainter.SmoothPixmapTransform, shown < 1.0)
        size = self.data.sizes[level]
        across = self.data.tiles_across(level)
        top_left, bottom_right = self._to_image(0, 0), self._to_image(self.width(), self.height())
        first_x = max(int(math.floor(top_left.x() / step / TILE)), 0)
        last_x = min(int(math.floor(bottom_right.x() / step / TILE)), across - 1)
        first_y = max(int(math.floor(top_left.y() / step / TILE)), 0)
        last_y = min(int(math.floor(bottom_right.y() / step / TILE)), across - 1)

        def screen(value: float, centre: float, half: float) -> int:         # a level coordinate to a screen pixel
            return round((value - centre / step) * shown + half)

        for ty in range(first_y, last_y + 1):
            for tx in range(first_x, last_x + 1):
                width, height = min(TILE, size - tx * TILE), min(TILE, size - ty * TILE)
                left, right = screen(tx * TILE, self.centre.x(), self.width() / 2), screen(tx * TILE + width, self.centre.x(), self.width() / 2)
                top, bottom = screen(ty * TILE, self.centre.y(), self.height() / 2), screen(ty * TILE + height, self.centre.y(), self.height() / 2)
                target, source = QRectF(left, top, right - left, bottom - top), QRectF(0, 0, width, height)
                painter.drawImage(target, self._hillshade(level, tx, ty), source)
                overlay = self._overlay(level, tx, ty)
                if overlay is not False:
                    painter.drawImage(target, overlay, source)
        painter.end()


class Window(QMainWindow):
    def __init__(self, data: ViewerData, layer: str, month: str | None):
        super().__init__()
        self.data = data
        self.setWindowTitle("Safe havens: the pole")
        self.view = MapView(data)
        self.layer_keys = [key for _title, key in LAYERS if key != "chains_month" or data.monthly_chains is not None]
        if layer not in self.layer_keys:
            raise ValueError(f"the layer {layer!r} is not available: no monthly chain pixel file beside the chain table "
                             "(run scripts/run_safe_haven_monthly_chain_maps.py)")
        self.layers = QComboBox()
        for title, key in LAYERS:
            if key in self.layer_keys:
                self.layers.addItem(title)
        reset = QPushButton("Reset view")
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, len(data.months) - 1)
        self.slider.setPageStep(12)
        self.caption = QLabel()
        self.caption.setMinimumWidth(260)
        top = QHBoxLayout()
        top.addWidget(QLabel("Show"))
        top.addWidget(self.layers)
        top.addStretch(1)
        top.addWidget(reset)
        bottom = QHBoxLayout()
        bottom.addWidget(self.slider, 1)
        bottom.addWidget(self.caption)
        layout = QVBoxLayout()
        layout.addLayout(top)
        layout.addWidget(self.view, 1)
        layout.addLayout(bottom)
        central = QWidget()
        central.setLayout(layout)
        self.setCentralWidget(central)
        self.layers.currentIndexChanged.connect(self._layer_changed)
        self.slider.valueChanged.connect(self._month_changed)
        reset.clicked.connect(self.view.reset_view)
        self.layers.setCurrentIndex(self.layer_keys.index(layer))
        if month is not None:
            self.slider.setValue(data.months.index(month))
        self._layer_changed()
        self._month_changed()

    def _layer_changed(self) -> None:
        self.view.set_layer(self.layer_keys[self.layers.currentIndex()])
        self._caption()

    def _month_changed(self) -> None:
        self.view.set_month(self.slider.value())
        self._caption()

    def _caption(self) -> None:
        month = self.slider.value()
        name = self.data.months[month]
        layer = self.view.layer
        if layer == "boxes":
            detail = f"{self.data.boxes_count(month):,} clusters"
        elif layer == "havens":
            detail = f"{self.data.havens_count(month):,} havens"
        elif layer == "chains_month":
            detail = f"{self.data.monthly_chain_count(month):,} chains"
        else:
            year = int(name[:4])
            detail = f"chains of {year}" if year in self.data.chain_years else f"no chain image for {year}"
        self.caption.setText(f"{name}    {detail}")


class Preparer(QThread):
    """Prepares the viewer's lists and pyramids off the user interface thread."""

    message = Signal(str)
    failed = Signal(str)

    def __init__(self, options: dict):
        super().__init__()
        self.options = options

    def run(self) -> None:
        try:
            prepare_viewer(self.options["viewer_directory"], self.options["cache_directory"],
                           chains_path=self.options["chains_path"], dem_path=self.options["dem_path"],
                           connect=self.options["connect"], edges_path=self.options["edges_path"],
                           rebuild=self.options["rebuild"], progress=self.message.emit)
        except Exception as error:                                           # shown in the window, then the program ends
            self.failed.emit(f"{type(error).__name__}: {error}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="View the safe-haven products of each month on the dark hillshade of the pole.")
    for flag, default, text in (
        ("--dem-path", DEM_PATH, "the DEM GeoTIFF"), ("--chains-path", CHAINS_PATH, "the final stage-4 table"),
        ("--edges-path", EDGES_PATH, "the stage-3 edge table (used when --connect is edges)"),
        ("--viewer-directory", VIEWER_DIRECTORY, "where viewer.json and havens.npz are kept"),
        ("--cache-directory", CACHE_DIRECTORY, "where the image pyramids are cached"),
    ):
        parser.add_argument(flag, type=lambda value: Path(value).expanduser(), default=default, help=f"{text} (default: {default})")
    parser.add_argument("--connect", choices=("chains", "edges"), default=CONNECT, help="what joins two havens in a cluster")
    parser.add_argument("--layer", choices=[key for _title, key in LAYERS], default="boxes", help="the layer shown first")
    parser.add_argument("--month", default=None, help="the month shown first, YYYY-MM (default: the first)")
    parser.add_argument("--rebuild", action="store_true", help="make viewer.json and havens.npz again")
    parser.add_argument("--screenshot", type=Path, default=None, help="save a picture of the window to this PNG and quit")
    return parser


def main(argv=()) -> int:
    args = build_parser().parse_args(list(argv))
    app = QApplication.instance() or QApplication([sys.argv[0]])
    app.setStyleSheet("QWidget { background: #1a1a19; color: #f0efec; } QPushButton, QComboBox { background: #2b2b29; "
                      "border: 1px solid #4a4a46; padding: 4px 10px; } QPushButton:hover { background: #3a3a37; }")
    status = QWidget()
    status.setWindowTitle("Preparing the viewer")
    label, bar = QLabel("Preparing the viewer..."), QProgressBar()
    bar.setRange(0, 0)
    column = QVBoxLayout(status)
    column.addWidget(label)
    column.addWidget(bar)
    status.setMinimumWidth(480)
    status.show()
    options = vars(args)
    preparer = Preparer(options)
    outcome = {"code": 0, "window": None}

    def open_window() -> None:
        try:
            data = ViewerData(args.viewer_directory, args.cache_directory)
            outcome["window"] = Window(data, args.layer, args.month)
        except Exception as error:
            fail(f"{type(error).__name__}: {error}")
            return
        outcome["window"].resize(1400, 950)
        outcome["window"].show()
        status.close()
        if args.screenshot is not None:
            app.processEvents()
            outcome["window"].grab().save(str(args.screenshot))
            app.quit()

    def fail(text: str) -> None:
        print(text, file=sys.stderr)
        outcome["code"] = 1
        app.quit()

    preparer.message.connect(label.setText)
    preparer.failed.connect(fail)
    preparer.finished.connect(lambda: open_window() if outcome["code"] == 0 else None)
    preparer.start()
    app.exec()
    preparer.wait()
    return outcome["code"]


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
