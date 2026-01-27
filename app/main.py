from __future__ import annotations

import sys
import traceback
import webbrowser
from pathlib import Path
from typing import Dict, List, Optional

if __package__ in (None, ""):
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

import torch

from app.inference import (
    ModelResult,
    get_device,
    load_2d_case,
    load_3d_case,
    normalize_case_id,
    run_swin_unet_3d,
    run_transunet,
    run_unet2d,
    run_unetpp,
    run_unetr,
    run_vnet,
)
from app.visualization import build_plotly_figure


try:
    from PyQt6 import QtCore, QtWidgets
    QT6 = True
except ImportError:
    from PyQt5 import QtCore, QtWidgets
    QT6 = False

try:
    if QT6:
        from PyQt6.QtWebEngineWidgets import QWebEngineView
    else:
        from PyQt5.QtWebEngineWidgets import QWebEngineView
except Exception:  # pragma: no cover - optional dependency
    QWebEngineView = None


class CaseSpinBox(QtWidgets.QSpinBox):
    def textFromValue(self, value: int) -> str:
        return f"{value:03d}"

    def valueFromText(self, text: str) -> int:
        return int(text)


class HtmlVisualizationWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.html_path: Optional[Path] = None

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.tabs = QtWidgets.QTabWidget(self)
        layout.addWidget(self.tabs)

        preview = QtWidgets.QWidget(self)
        preview_layout = QtWidgets.QVBoxLayout(preview)
        self.info = QtWidgets.QLabel(
            "Visualization is rendered to HTML. Click the link below or switch to the Open HTML tab.",
            preview,
        )
        self.info.setWordWrap(True)
        preview_layout.addWidget(self.info)

        self.link = QtWidgets.QTextBrowser(preview)
        self.link.setOpenExternalLinks(True)
        self.link.setText("No visualization yet.")
        preview_layout.addWidget(self.link)

        self.open_btn = QtWidgets.QPushButton("Open HTML in Browser", preview)
        self.open_btn.clicked.connect(self._open_in_browser)
        preview_layout.addWidget(self.open_btn)
        preview_layout.addStretch(1)

        open_tab = QtWidgets.QWidget(self)
        open_layout = QtWidgets.QVBoxLayout(open_tab)
        open_layout.addWidget(QtWidgets.QLabel("Switching here opens the HTML in your browser.", open_tab))
        open_layout.addStretch(1)

        self.tabs.addTab(preview, "Preview")
        self.tabs.addTab(open_tab, "Open HTML")
        self.tabs.currentChanged.connect(self._on_tab_changed)

        self.open_btn.setEnabled(False)

    def _on_tab_changed(self, index: int) -> None:
        if index == 1:
            self._open_in_browser()
            self.tabs.setCurrentIndex(0)

    def _open_in_browser(self) -> None:
        if self.html_path and self.html_path.exists():
            webbrowser.open(self.html_path.as_uri())

    def set_html_path(self, html_path: Optional[Path], empty_message: Optional[str] = None) -> None:
        self.html_path = html_path
        if not html_path or not html_path.exists():
            self.link.setText(empty_message or "No visualization yet.")
            self.open_btn.setEnabled(False)
            return

        self.link.setText(f'<a href="{html_path.as_uri()}">Open visualization HTML</a>')
        self.open_btn.setEnabled(True)


class InferenceWorker(QtCore.QObject):
    finished = QtCore.pyqtSignal(object, object)
    error = QtCore.pyqtSignal(str)
    log = QtCore.pyqtSignal(str)

    def __init__(
        self,
        case_id: str,
        case_dir_2d: Path,
        case_dir_3d: Path,
        selected_models: List[str],
        weights_dir: Path,
        device: torch.device,
        brain_modality: str,
    ):
        super().__init__()
        self.case_id = case_id
        self.case_dir_2d = case_dir_2d
        self.case_dir_3d = case_dir_3d
        self.selected_models = selected_models
        self.weights_dir = weights_dir
        self.device = device
        self.brain_modality = brain_modality

    def run(self) -> None:
        try:
            results_2d: List[ModelResult] = []
            results_3d: List[ModelResult] = []

            need_2d = any(name in {"UNet", "UNet++", "TransUNet"} for name in self.selected_models)
            need_3d = any(name in {"VNet", "UNETR", "SwinUNet3D"} for name in self.selected_models)

            if need_2d:
                self.log.emit(f"[INFO] Loading 2D case: {self.case_dir_2d}")
                vol_2d, gt_2d = load_2d_case(self.case_dir_2d)
            else:
                vol_2d = gt_2d = None

            if need_3d:
                self.log.emit(f"[INFO] Loading 3D case: {self.case_dir_3d}")
                vol_3d, gt_3d, spacing_3d = load_3d_case(self.case_dir_3d)
            else:
                vol_3d = gt_3d = spacing_3d = None

            modality_map = {"flair": 0, "t1": 1, "t1ce": 2, "t2": 3}
            mod_idx = modality_map.get(self.brain_modality, 0)
            brain_2d = vol_2d[:, mod_idx] > 0 if vol_2d is not None else None
            brain_3d = vol_3d[mod_idx] > 0 if vol_3d is not None else None

            for name in self.selected_models:
                self.log.emit(f"[RUN] {name} ...")
                if name == "UNet":
                    res = run_unet2d(vol_2d, gt_2d, self.weights_dir / "unet.pth", self.device)
                    res.brain_mask = brain_2d
                    results_2d.append(res)
                elif name == "UNet++":
                    res = run_unetpp(vol_2d, gt_2d, self.weights_dir / "unet++.pth", self.device)
                    res.brain_mask = brain_2d
                    results_2d.append(res)
                elif name == "TransUNet":
                    res = run_transunet(vol_2d, gt_2d, self.weights_dir / "transunet.pth", self.device)
                    res.brain_mask = brain_2d
                    results_2d.append(res)
                elif name == "VNet":
                    res = run_vnet(vol_3d, gt_3d, self.weights_dir / "vnet.pth", self.device, spacing=spacing_3d)
                    res.brain_mask = brain_3d
                    results_3d.append(res)
                elif name == "UNETR":
                    res = run_unetr(vol_3d, gt_3d, self.weights_dir / "unetr.pth", self.device, spacing=spacing_3d)
                    res.brain_mask = brain_3d
                    results_3d.append(res)
                elif name == "SwinUNet3D":
                    res = run_swin_unet_3d(
                        vol_3d, gt_3d, self.weights_dir / "swinunet3d.pth", self.device, spacing=spacing_3d
                    )
                    res.brain_mask = brain_3d
                    results_3d.append(res)
                else:
                    self.log.emit(f"[WARN] Unknown model: {name}")

                if self.device.type == "cuda":
                    torch.cuda.empty_cache()

            self.finished.emit(results_2d, results_3d)
        except Exception as exc:  # pragma: no cover - UI error reporting
            trace = traceback.format_exc()
            self.error.emit(f"{exc}\n{trace}")


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Brain Tumor Segmentation App")
        self.root_dir = Path(__file__).resolve().parents[1]
        self.data_root_2d = self.root_dir / "data" / "processed" / "2d" / "labeled"
        self.data_root_3d = self.root_dir / "data" / "processed" / "3d" / "labeled"
        self.weights_dir = self.root_dir / "weights"
        self.output_dir = self.root_dir / "app_outputs"

        self.results_2d: List[ModelResult] = []
        self.results_3d: List[ModelResult] = []

        self._thread: Optional[QtCore.QThread] = None
        self._worker: Optional[InferenceWorker] = None

        self._setup_ui()
        self._apply_style()

    def _setup_ui(self) -> None:
        central = QtWidgets.QWidget(self)
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)

        layout.addWidget(self._build_case_group())
        layout.addWidget(self._build_model_group())
        layout.addWidget(self._build_visual_group())
        layout.addWidget(self._build_action_group())
        layout.addWidget(self._build_results_group())
        layout.addWidget(self._build_visualization_group())
        layout.addWidget(self._build_log_group())
        # Give the visualization and metrics areas more room.
        layout.setStretch(4, 5)
        layout.setStretch(5, 1)
        layout.setStretch(6, 2)

    def _apply_style(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow { background: #f6f7f9; }
            QGroupBox {
                border: 1px solid #c8cdd2;
                border-radius: 6px;
                margin-top: 10px;
                background: #ffffff;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 8px;
                padding: 0 4px 0 4px;
                color: #1f2933;
                font-weight: bold;
            }
            QTableWidget {
                border: 1px solid #d3d7dc;
                gridline-color: #e1e4e8;
                selection-background-color: #cfe8ff;
            }
            QHeaderView::section {
                background: #f0f2f5;
                padding: 4px;
                border: 1px solid #d3d7dc;
                font-weight: bold;
            }
            QPushButton {
                background: #1f7ae0;
                color: white;
                border-radius: 4px;
                padding: 4px 10px;
            }
            QPushButton:disabled {
                background: #a7b3bf;
            }
            QLineEdit, QSpinBox, QComboBox {
                padding: 4px;
                border: 1px solid #c8cdd2;
                border-radius: 4px;
            }
            """
        )

    def _build_case_group(self) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("Case Selection", self)
        grid = QtWidgets.QGridLayout(box)

        grid.addWidget(QtWidgets.QLabel("Case ID (001-369):"), 0, 0)
        self.case_spin = CaseSpinBox(self)
        self.case_spin.setRange(1, 369)
        self.case_spin.setValue(1)
        self.case_spin.valueChanged.connect(self._update_case_paths)
        grid.addWidget(self.case_spin, 0, 1)

        self.btn_load_case = QtWidgets.QPushButton("Load Case", self)
        self.btn_load_case.clicked.connect(self._update_case_paths)
        grid.addWidget(self.btn_load_case, 0, 2)

        self.btn_browse_2d = QtWidgets.QPushButton("Browse 2D Case", self)
        self.btn_browse_2d.clicked.connect(self._browse_2d)
        grid.addWidget(self.btn_browse_2d, 1, 0)

        self.btn_browse_3d = QtWidgets.QPushButton("Browse 3D Case", self)
        self.btn_browse_3d.clicked.connect(self._browse_3d)
        grid.addWidget(self.btn_browse_3d, 1, 1)

        self.label_2d = QtWidgets.QLabel("-", self)
        self.label_3d = QtWidgets.QLabel("-", self)
        grid.addWidget(QtWidgets.QLabel("2D Path:"), 2, 0)
        grid.addWidget(self.label_2d, 2, 1, 1, 2)
        grid.addWidget(QtWidgets.QLabel("3D Path:"), 3, 0)
        grid.addWidget(self.label_3d, 3, 1, 1, 2)

        self._update_case_paths()
        return box

    def _build_model_group(self) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("Model Selection", self)
        grid = QtWidgets.QGridLayout(box)

        self.model_checks: Dict[str, QtWidgets.QCheckBox] = {}
        models = ["UNet", "UNet++", "TransUNet", "VNet", "UNETR", "SwinUNet3D"]
        for i, name in enumerate(models):
            cb = QtWidgets.QCheckBox(name, self)
            cb.setChecked(True)
            grid.addWidget(cb, i // 3, i % 3)
            self.model_checks[name] = cb

        return box

    def _build_visual_group(self) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("Visualization Options", self)
        layout = QtWidgets.QHBoxLayout(box)

        self.chk_show_gt = QtWidgets.QCheckBox("Show GT Row", self)
        self.chk_overlay_gt = QtWidgets.QCheckBox("Overlay GT", self)
        self.chk_sync = QtWidgets.QCheckBox("Sync Camera", self)
        self.chk_show_brain = QtWidgets.QCheckBox("Show Brain Mesh", self)
        self.chk_show_brain.setChecked(True)
        self.chk_sync.setChecked(True)

        self.brain_modality = QtWidgets.QComboBox(self)
        self.brain_modality.addItems(["flair", "t1", "t1ce", "t2"])
        self.brain_modality.setCurrentText("flair")

        for cb in [self.chk_show_gt, self.chk_overlay_gt, self.chk_sync, self.chk_show_brain]:
            cb.stateChanged.connect(self._update_visuals)
            layout.addWidget(cb)

        layout.addWidget(QtWidgets.QLabel("Brain Modality:", self))
        layout.addWidget(self.brain_modality)
        self.brain_modality.currentIndexChanged.connect(self._update_visuals)
        layout.addStretch(1)
        return box

    def _build_action_group(self) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("Actions", self)
        layout = QtWidgets.QHBoxLayout(box)

        self.btn_run = QtWidgets.QPushButton("Run Prediction", self)
        self.btn_run.clicked.connect(self._run_inference)
        layout.addWidget(self.btn_run)

        layout.addStretch(1)
        return box

    def _build_results_group(self) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("Metrics", self)
        layout = QtWidgets.QVBoxLayout(box)

        self.table = QtWidgets.QTableWidget(self)
        headers = [
            "Model",
            "Type",
            "Input",
            "Time(s)",
            "Dice WT",
            "Dice TC",
            "Dice ET",
            "IoU WT",
            "IoU TC",
            "IoU ET",
            "ASD WT",
            "ASD TC",
            "ASD ET",
            "HD95 WT",
            "HD95 TC",
            "HD95 ET",
        ]
        self.table.setColumnCount(len(headers))
        self.table.setHorizontalHeaderLabels(headers)
        self.table.setAlternatingRowColors(True)
        self.table.horizontalHeader().setStretchLastSection(True)
        self._setup_table_columns()
        layout.addWidget(self.table)
        return box

    def _build_visualization_group(self) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("Visualization", self)
        layout = QtWidgets.QVBoxLayout(box)

        self.visual_view = HtmlVisualizationWidget(self)
        layout.addWidget(self.visual_view)
        box.setMaximumHeight(140)
        return box

    def _setup_table_columns(self) -> None:
        header = self.table.horizontalHeader()
        if QT6:
            resize_contents = QtWidgets.QHeaderView.ResizeMode.ResizeToContents
            stretch = QtWidgets.QHeaderView.ResizeMode.Stretch
        else:
            resize_contents = QtWidgets.QHeaderView.ResizeToContents
            stretch = QtWidgets.QHeaderView.Stretch

        for col in range(self.table.columnCount()):
            header.setSectionResizeMode(col, resize_contents)
        header.setSectionResizeMode(self.table.columnCount() - 1, stretch)

        self.table.setColumnWidth(0, 120)
        self.table.setColumnWidth(1, 50)
        self.table.setColumnWidth(2, 110)
        self.table.setColumnWidth(3, 70)
        for col in range(4, self.table.columnCount()):
            self.table.setColumnWidth(col, 75)

    def _build_log_group(self) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("Log", self)
        layout = QtWidgets.QVBoxLayout(box)
        self.log_box = QtWidgets.QPlainTextEdit(self)
        self.log_box.setReadOnly(True)
        layout.addWidget(self.log_box)
        return box

    def _log(self, text: str) -> None:
        self.log_box.appendPlainText(text)

    def _update_case_paths(self) -> None:
        case_id = normalize_case_id(str(self.case_spin.value()))
        case_dir_2d = self.data_root_2d / case_id
        case_dir_3d = self.data_root_3d / case_id
        self.label_2d.setText(str(case_dir_2d))
        self.label_3d.setText(str(case_dir_3d))

    def _browse_2d(self) -> None:
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Select 2D Case", str(self.data_root_2d))
        if not path:
            return
        case_id = Path(path).name
        if case_id.lower().startswith("brain_"):
            self.case_spin.setValue(int(case_id.split("_")[1]))
        self._update_case_paths()

    def _browse_3d(self) -> None:
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Select 3D Case", str(self.data_root_3d))
        if not path:
            return
        case_id = Path(path).name
        if case_id.lower().startswith("brain_"):
            self.case_spin.setValue(int(case_id.split("_")[1]))
        self._update_case_paths()

    def _selected_models(self) -> List[str]:
        return [name for name, cb in self.model_checks.items() if cb.isChecked()]

    def _run_inference(self) -> None:
        selected = self._selected_models()
        if not selected:
            self._log("[WARN] No models selected.")
            return

        case_id = normalize_case_id(str(self.case_spin.value()))
        case_dir_2d = self.data_root_2d / case_id
        case_dir_3d = self.data_root_3d / case_id

        if any(name in {"UNet", "UNet++", "TransUNet"} for name in selected) and not case_dir_2d.exists():
            self._log(f"[ERR] 2D case path not found: {case_dir_2d}")
            return
        if any(name in {"VNet", "UNETR", "SwinUNet3D"} for name in selected) and not case_dir_3d.exists():
            self._log(f"[ERR] 3D case path not found: {case_dir_3d}")
            return

        self.btn_run.setEnabled(False)
        self._log(f"[INFO] Starting inference for {case_id} on {get_device()} ...")

        self._thread = QtCore.QThread(self)
        self._worker = InferenceWorker(
            case_id=case_id,
            case_dir_2d=case_dir_2d,
            case_dir_3d=case_dir_3d,
            selected_models=selected,
            weights_dir=self.weights_dir,
            device=get_device(),
            brain_modality=self.brain_modality.currentText(),
        )
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.finished.connect(self._on_inference_done)
        self._worker.error.connect(self._on_inference_error)
        self._worker.log.connect(self._log)
        self._worker.finished.connect(self._thread.quit)
        self._worker.finished.connect(self._worker.deleteLater)
        self._thread.finished.connect(self._thread.deleteLater)
        self._thread.start()

    def _on_inference_done(self, results_2d: List[ModelResult], results_3d: List[ModelResult]) -> None:
        self.results_2d = results_2d
        self.results_3d = results_3d
        self.btn_run.setEnabled(True)
        self._log("[INFO] Inference completed.")
        self._update_table()
        self._update_visuals()

    def _on_inference_error(self, message: str) -> None:
        self.btn_run.setEnabled(True)
        self._log(f"[ERR] {message}")

    def _update_table(self) -> None:
        results = self.results_2d + self.results_3d
        self.table.setRowCount(len(results))
        for row, res in enumerate(results):
            self._set_item(row, 0, res.name)
            self._set_item(row, 1, res.kind.upper())
            self._set_item(row, 2, res.input_size or "-")
            self._set_item(row, 3, f"{res.time_sec:.3f}")
            self._fill_metrics(row, res.metrics)

        self.table.resizeColumnsToContents()

    def _fill_metrics(self, row: int, metrics: Dict[str, Dict[str, float]]) -> None:
        cols = [
            ("WT", "dice"),
            ("TC", "dice"),
            ("ET", "dice"),
            ("WT", "iou"),
            ("TC", "iou"),
            ("ET", "iou"),
            ("WT", "asd"),
            ("TC", "asd"),
            ("ET", "asd"),
            ("WT", "hd95"),
            ("TC", "hd95"),
            ("ET", "hd95"),
        ]
        base_col = 4
        for idx, (region, key) in enumerate(cols):
            val = metrics.get(region, {}).get(key, float("nan"))
            self._set_item(row, base_col + idx, f"{val:.4f}")

    def _set_item(self, row: int, col: int, text: str) -> None:
        item = QtWidgets.QTableWidgetItem(text)
        if QT6:
            flag = QtCore.Qt.ItemFlag.ItemIsEditable
        else:
            flag = QtCore.Qt.ItemIsEditable
        item.setFlags(item.flags() & ~flag)
        self.table.setItem(row, col, item)

    def _update_visuals(self) -> None:
        show_gt = self.chk_show_gt.isChecked()
        overlay = self.chk_overlay_gt.isChecked()
        sync = self.chk_sync.isChecked()
        show_brain = self.chk_show_brain.isChecked()
        case_id = normalize_case_id(str(self.case_spin.value()))

        all_results = self.results_3d + self.results_2d
        html_3d = build_plotly_figure(
            all_results,
            title=f"{case_id} - All Models (3D View)",
            show_gt_row=show_gt,
            overlay_gt=overlay,
            show_brain=show_brain,
            sync_camera=sync,
        )

        self.output_dir.mkdir(parents=True, exist_ok=True)
        path_all = self.output_dir / f"{case_id}_all.html"

        if html_3d:
            path_all.write_text(html_3d, encoding="utf-8")

        empty_msg = "No visualization yet. Select a model and run prediction."
        self.visual_view.set_html_path(path_all if html_3d else None, empty_message=empty_msg)


def main() -> None:
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.resize(1200, 900)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
