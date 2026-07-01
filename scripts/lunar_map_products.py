#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
from typing import NoReturn

import lunarscout as ls


CATALOG_PATH = Path(__file__).with_name("lunar_map_products.json")
REGION_NAMES = ("south_pole", "north_pole", "nearside", "farside")


def _load_catalog(path: Path) -> ls.MapProductCatalog:
    return ls.load_map_product_catalog(path)


def _print_products(products: list[ls.MapProduct]) -> None:
    rows = [
        (str(product.id), product.name, product.description)
        for product in products
    ]
    headers = ("id", "name", "description")
    widths = [len(header) for header in headers]
    for row in rows:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(value))

    print(
        f"{headers[0]:>{widths[0]}}  "
        f"{headers[1]:<{widths[1]}}  "
        f"{headers[2]:<{widths[2]}}"
    )
    print(
        f"{'-' * widths[0]:>{widths[0]}}  "
        f"{'-' * widths[1]:<{widths[1]}}  "
        f"{'-' * widths[2]:<{widths[2]}}"
    )
    for row in rows:
        print(
            f"{row[0]:>{widths[0]}}  "
            f"{row[1]:<{widths[1]}}  "
            f"{row[2]:<{widths[2]}}"
        )


def _progress_line(received: int, total: int | None) -> None:
    if total:
        percent = min(100.0, received * 100.0 / total)
        sys.stderr.write(f"\rDownloading {percent:5.1f}%")
    else:
        sys.stderr.write(f"\rDownloading {received:,} bytes")
    sys.stderr.flush()


def _cmd_list(args: argparse.Namespace) -> int:
    catalog = _load_catalog(args.catalog)
    pattern = " ".join(args.text_pattern)
    _print_products(ls.search_map_products(catalog.products, pattern))
    return 0


def _cmd_download(args: argparse.Namespace) -> int:
    catalog = _load_catalog(args.catalog)
    product = catalog.product(args.id)
    directory = ls.map_product_download_directory(
        scenario_root=args.scenario_root,
        scenario=args.scenario,
        environ_scenario_root=os.environ.get("LUNARSCOUT_SCENARIO_ROOT"),
    )
    path = ls.download_map_product(
        product,
        directory,
        overwrite=args.overwrite,
        progress_callback=_progress_line,
    )
    sys.stderr.write("\n")
    print(path)
    return 0


def _cmd_gui(args: argparse.Namespace) -> int:  # pragma: no cover - GUI
    try:
        import numpy as np
        import rasterio
        from PySide6.QtCore import QObject, Qt, QThread, QTimer, Signal, Slot
        from PySide6.QtGui import (
            QBrush,
            QColor,
            QImage,
            QPainter,
            QPen,
            QPixmap,
        )
        from PySide6.QtWidgets import (
            QApplication,
            QAbstractItemView,
            QCheckBox,
            QComboBox,
            QFormLayout,
            QGraphicsItem,
            QGraphicsRectItem,
            QGraphicsScene,
            QGraphicsTextItem,
            QGraphicsView,
            QHBoxLayout,
            QLabel,
            QLineEdit,
            QMainWindow,
            QMessageBox,
            QPushButton,
            QProgressBar,
            QSplitter,
            QTableWidget,
            QTableWidgetItem,
            QVBoxLayout,
            QWidget,
        )
    except ImportError as exc:
        raise SystemExit(
            "The gui verb requires PySide6, rasterio, and numpy in the active "
            f"Python environment. Missing import: {exc}"
        ) from exc

    class MapView(QGraphicsView):
        def __init__(self) -> None:
            super().__init__()
            self.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
            self.setTransformationAnchor(
                QGraphicsView.ViewportAnchor.AnchorUnderMouse
            )

        def wheelEvent(self, event) -> None:
            factor = 1.25 if event.angleDelta().y() > 0 else 0.8
            self.scale(factor, factor)

    class DownloadWorker(QObject):
        progress = Signal(int)
        message = Signal(str)
        finished = Signal(list, str)
        failed = Signal(str)

        def __init__(self, downloads) -> None:
            super().__init__()
            self._downloads = downloads

        @Slot()
        def run(self) -> None:
            paths: list[str] = []
            last_scenario = ""
            total_products = len(self._downloads)
            try:
                for index, (product, directory, scenario_name) in enumerate(
                    self._downloads
                ):
                    base = index * 100.0 / total_products
                    span = 100.0 / total_products
                    self.message.emit(f"Downloading {product.name}")

                    def on_progress(received: int, total: int | None) -> None:
                        if total:
                            fraction = min(1.0, received / total)
                            self.progress.emit(int(base + span * fraction))
                        else:
                            self.progress.emit(int(base))

                    path = ls.download_map_product(
                        product,
                        directory,
                        progress_callback=on_progress,
                    )
                    paths.append(str(path))
                    last_scenario = scenario_name
                self.progress.emit(100)
                self.finished.emit(paths, last_scenario)
            except Exception as exc:
                self.failed.emit(str(exc))

    class MainWindow(QMainWindow):
        def __init__(self, catalog: ls.MapProductCatalog) -> None:
            super().__init__()
            self.catalog = catalog
            self.scene = QGraphicsScene(self)
            self.view = MapView()
            self.view.setScene(self.scene)
            self.bounds: tuple[float, float, float, float] | None = None
            self.pixmap_width = 1
            self.pixmap_height = 1
            self.outline_items: dict[int, list] = {}
            self.worker_thread: QThread | None = None
            self.worker: DownloadWorker | None = None

            self.region_combo = QComboBox()
            self.region_combo.addItems(REGION_NAMES)
            self.region_combo.currentTextChanged.connect(self._load_region)

            self.scenario_root = QLineEdit(
                os.environ.get("LUNARSCOUT_SCENARIO_ROOT", "")
            )
            self.scenario = QLineEdit()
            self.scenario_from_dem = QCheckBox("Use DEM name as scenario")
            self.scenario_from_dem.toggled.connect(self._scenario_from_dem_toggled)
            self._remembered_scenario_name: str | None = None

            self.reset_button = QPushButton("Reset Map")
            self.reset_button.clicked.connect(self._reset_map)

            self.download_button = QPushButton("Download Selected")
            self.download_button.clicked.connect(self._download_selected)
            self.download_button.setEnabled(False)

            self.progress = QProgressBar()
            self.progress.setRange(0, 100)
            self.progress_label = QLabel("0%")

            self.table = QTableWidget(0, 3)
            self.table.setHorizontalHeaderLabels(["id", "name", "description"])
            self.table.setSelectionBehavior(
                QAbstractItemView.SelectionBehavior.SelectRows
            )
            self.table.setSelectionMode(
                QAbstractItemView.SelectionMode.ExtendedSelection
            )
            self.table.itemSelectionChanged.connect(self._selection_changed)
            self.table.horizontalHeader().setStretchLastSection(True)

            controls = QWidget()
            form = QFormLayout(controls)
            form.addRow("Region", self.region_combo)
            form.addRow("Scenario root", self.scenario_root)
            form.addRow("Scenario", self.scenario)
            form.addRow("", self.scenario_from_dem)

            top_buttons = QHBoxLayout()
            top_buttons.addWidget(self.reset_button)
            top_buttons.addWidget(self.download_button)

            progress_row = QHBoxLayout()
            progress_row.addWidget(self.progress)
            progress_row.addWidget(self.progress_label)

            side = QWidget()
            side_layout = QVBoxLayout(side)
            side_layout.addWidget(controls)
            side_layout.addLayout(top_buttons)
            side_layout.addWidget(self.table)
            side_layout.addLayout(progress_row)

            splitter = QSplitter()
            splitter.addWidget(side)
            splitter.addWidget(self.view)
            splitter.setStretchFactor(0, 2)
            splitter.setStretchFactor(1, 3)
            self.setCentralWidget(splitter)
            self.resize(1300, 800)
            self.setWindowTitle("Lunarscout Map Products")

            self._load_region(self.region_combo.currentText())

        def _products_for_region(self, region: str) -> list[ls.MapProduct]:
            return [
                product
                for product in self.catalog.products
                if product.region == region
            ]

        def _load_region(self, region_name: str) -> None:
            self.scene.clear()
            self.outline_items.clear()
            region = self.catalog.regions[region_name]
            if region.overview_geotiff is None:
                self.bounds = None
                self.scene.addText(f"No overview map configured for {region_name}")
            else:
                try:
                    with rasterio.open(region.overview_geotiff) as dataset:
                        scale = max(dataset.width / 2200, dataset.height / 2200, 1)
                        width = max(1, int(dataset.width / scale))
                        height = max(1, int(dataset.height / scale))
                        if dataset.count >= 3:
                            data = dataset.read(
                                [1, 2, 3],
                                out_shape=(3, height, width),
                            )
                        else:
                            data = dataset.read(1, out_shape=(height, width))
                        self.bounds = (
                            dataset.bounds.left,
                            dataset.bounds.bottom,
                            dataset.bounds.right,
                            dataset.bounds.top,
                        )
                except Exception as exc:
                    self.bounds = None
                    self.scene.addText(f"Could not load overview map: {exc}")
                else:
                    if data.ndim == 3:
                        image_data = np.moveaxis(data[:3], 0, 2)
                        if image_data.dtype != np.uint8:
                            image_data = image_data.astype(np.float32)
                            for band_index in range(3):
                                band = image_data[:, :, band_index]
                                valid = band[np.isfinite(band)]
                                if valid.size:
                                    low, high = np.percentile(valid, [1, 99])
                                else:
                                    low, high = 0, 1
                                if high <= low:
                                    high = low + 1
                                image_data[:, :, band_index] = np.clip(
                                    (band - low) * 255.0 / (high - low),
                                    0,
                                    255,
                                )
                            image_data = image_data.astype(np.uint8)
                        image_data = np.ascontiguousarray(image_data)
                        image = QImage(
                            image_data.tobytes(),
                            width,
                            height,
                            width * 3,
                            QImage.Format.Format_RGB888,
                        ).copy()
                    else:
                        valid = data[np.isfinite(data)]
                        if valid.size:
                            low, high = np.percentile(valid, [1, 99])
                        else:
                            low, high = 0, 1
                        if high <= low:
                            high = low + 1
                        image_data = np.clip(
                            (data - low) * 255.0 / (high - low),
                            0,
                            255,
                        )
                        image_data = np.ascontiguousarray(image_data.astype(np.uint8))
                        image = QImage(
                            image_data.tobytes(),
                            width,
                            height,
                            width,
                            QImage.Format.Format_Grayscale8,
                        ).copy()
                    pixmap = QPixmap.fromImage(image)
                    self.pixmap_width = width
                    self.pixmap_height = height
                    self.scene.addPixmap(pixmap)
                    self.scene.setSceneRect(0, 0, width, height)

            self._populate_table(region_name)
            self._draw_outlines()
            self._reset_map()

        def _populate_table(self, region_name: str) -> None:
            products = self._products_for_region(region_name)
            self.table.setRowCount(len(products))
            for row, product in enumerate(products):
                for column, value in enumerate(
                    (str(product.id), product.name, product.description)
                ):
                    item = QTableWidgetItem(value)
                    item.setData(Qt.ItemDataRole.UserRole, product.id)
                    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                    self.table.setItem(row, column, item)
            self.table.resizeColumnsToContents()

        def _map_to_scene(self, x: float, y: float) -> tuple[float, float]:
            if self.bounds is None:
                return x, y
            left, bottom, right, top = self.bounds
            scene_x = (x - left) * self.pixmap_width / (right - left)
            scene_y = (top - y) * self.pixmap_height / (top - bottom)
            return scene_x, scene_y

        def _selected_product_ids(self) -> set[int]:
            ids: set[int] = set()
            for item in self.table.selectedItems():
                product_id = item.data(Qt.ItemDataRole.UserRole)
                if product_id is not None:
                    ids.add(int(product_id))
            return ids

        def _selected_products(self) -> list[ls.MapProduct]:
            return [
                self.catalog.product(product_id)
                for product_id in sorted(self._selected_product_ids())
            ]

        def _scenario_name_for_selection(self) -> str:
            products = self._selected_products()
            if not products:
                return ""
            return ls.map_product_scenario_name(products[0])

        def _scenario_from_dem_toggled(self, checked: bool) -> None:
            if checked:
                self._remembered_scenario_name = self.scenario.text()
                scenario_name = self._scenario_name_for_selection()
                if scenario_name:
                    self.scenario.setText(scenario_name)
            else:
                self.scenario.setText(self._remembered_scenario_name or "")
                self._remembered_scenario_name = None

        def _draw_outlines(self) -> None:
            for items in self.outline_items.values():
                for item in items:
                    self.scene.removeItem(item)
            self.outline_items.clear()
            selected = self._selected_product_ids()
            products = self._products_for_region(self.region_combo.currentText())
            for product in products:
                if selected and product.id not in selected:
                    continue
                left, bottom, right, top = product.bounds
                x1, y1 = self._map_to_scene(left, top)
                x2, y2 = self._map_to_scene(right, bottom)
                rect = QGraphicsRectItem(min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1))
                pen = QPen(QColor("#ffcc00"), 2)
                pen.setCosmetic(True)
                rect.setPen(pen)
                rect.setBrush(QBrush(Qt.BrushStyle.NoBrush))
                label = QGraphicsTextItem(str(product.id))
                label.setDefaultTextColor(QColor("#ffffff"))
                label.setPos(min(x1, x2) + 4, min(y1, y2) + 2)
                label.setFlag(
                    QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations
                )
                self.scene.addItem(rect)
                self.scene.addItem(label)
                self.outline_items[product.id] = [rect, label]

        def _selection_changed(self) -> None:
            self.download_button.setEnabled(bool(self._selected_product_ids()))
            if self.scenario_from_dem.isChecked():
                scenario_name = self._scenario_name_for_selection()
                if scenario_name:
                    self.scenario.setText(scenario_name)
            self._draw_outlines()

        def _reset_map(self) -> None:
            self.view.fitInView(
                self.scene.sceneRect(),
                Qt.AspectRatioMode.KeepAspectRatio,
            )

        def _download_selected(self) -> None:
            products = self._selected_products()
            if not products:
                return
            try:
                downloads = []
                if self.scenario_from_dem.isChecked():
                    for product in products:
                        scenario_name = ls.map_product_scenario_name(product)
                        directory = ls.map_product_download_directory(
                            scenario_root=self.scenario_root.text().strip() or None,
                            scenario=scenario_name,
                            environ_scenario_root=os.environ.get(
                                "LUNARSCOUT_SCENARIO_ROOT"
                            ),
                            create=True,
                        )
                        downloads.append((product, directory, scenario_name))
                else:
                    scenario_name = self.scenario.text().strip()
                    directory = ls.map_product_download_directory(
                        scenario_root=self.scenario_root.text().strip() or None,
                        scenario=scenario_name or None,
                        environ_scenario_root=os.environ.get(
                            "LUNARSCOUT_SCENARIO_ROOT"
                        ),
                    )
                    downloads = [
                        (product, directory, scenario_name)
                        for product in products
                    ]
            except Exception as exc:
                QMessageBox.critical(self, "Download Error", str(exc))
                return
            self.progress.setValue(0)
            self.progress_label.setText("0%")
            self.download_button.setEnabled(False)

            self.worker_thread = QThread(self)
            self.worker = DownloadWorker(downloads)
            self.worker.moveToThread(self.worker_thread)
            self.worker_thread.started.connect(self.worker.run)
            self.worker.progress.connect(self._set_progress)
            self.worker.message.connect(self.progress_label.setText)
            self.worker.finished.connect(self._download_finished)
            self.worker.failed.connect(self._download_failed)
            self.worker.finished.connect(self.worker_thread.quit)
            self.worker.failed.connect(self.worker_thread.quit)
            self.worker_thread.finished.connect(self.worker.deleteLater)
            self.worker_thread.finished.connect(self.worker_thread.deleteLater)
            self.worker_thread.start()

        @Slot(int)
        def _set_progress(self, percent: int) -> None:
            self.progress.setValue(percent)
            self.progress_label.setText(f"{percent}%")

        @Slot(list, str)
        def _download_finished(self, paths: list[str], last_scenario: str) -> None:
            self.download_button.setEnabled(bool(self._selected_product_ids()))
            self.progress.setValue(100)
            self.progress_label.setText("100%")
            if last_scenario:
                self.scenario.setText(last_scenario)
                if self.scenario_from_dem.isChecked():
                    self._remembered_scenario_name = last_scenario
            QMessageBox.information(
                self,
                "Download Complete",
                "Downloaded:\n" + "\n".join(paths),
            )

        @Slot(str)
        def _download_failed(self, message: str) -> None:
            self.download_button.setEnabled(bool(self._selected_product_ids()))
            QMessageBox.critical(self, "Download Error", message)

    app = QApplication(sys.argv)
    window = MainWindow(_load_catalog(args.catalog))
    window.show()
    QTimer.singleShot(0, window._reset_map)
    return app.exec()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="List and download well-known lunar map products.",
    )
    parser.add_argument(
        "--catalog",
        type=Path,
        default=CATALOG_PATH,
        help=f"product catalog JSON path (default: {CATALOG_PATH})",
    )
    subparsers = parser.add_subparsers(dest="verb")

    list_parser = subparsers.add_parser("list", help="list matching products")
    list_parser.add_argument("text_pattern", nargs="*")
    list_parser.set_defaults(func=_cmd_list)

    download_parser = subparsers.add_parser("download", help="download a product")
    download_parser.add_argument("id", type=int)
    download_parser.add_argument("--scenario-root", type=Path)
    download_parser.add_argument("--scenario")
    download_parser.add_argument("--overwrite", action="store_true")
    download_parser.set_defaults(func=_cmd_download)

    gui_parser = subparsers.add_parser("gui", help="open the product download GUI")
    gui_parser.set_defaults(func=_cmd_gui)

    subparsers.add_parser("help", help="show this help")
    return parser


def _fail(message: str) -> NoReturn:
    print(message, file=sys.stderr)
    raise SystemExit(1)


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    arguments = sys.argv[1:] if argv is None else argv
    if not arguments or arguments[0] == "help":
        parser.print_help()
        return 0
    args = parser.parse_args(arguments)
    if not hasattr(args, "func"):
        parser.print_help()
        return 0
    try:
        return args.func(args)
    except Exception as exc:
        _fail(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
