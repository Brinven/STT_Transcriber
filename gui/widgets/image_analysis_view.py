"""
Medical image analysis widget for STT Transcriber.

Provides an image preview (up to 4 images), preset/custom query input,
and a results panel for MedGemma vision-language analysis.  Supports
drag-and-drop of image files.

Visible only in medical mode after the user clicks Upload Image.
"""

import html
import logging
import os
from pathlib import Path

from PySide6.QtCore import QMarginsF, Qt, QUrl, Signal
from PySide6.QtGui import (
    QDragEnterEvent,
    QDropEvent,
    QPageLayout,
    QPageSize,
    QPixmap,
    QTextDocument,
)
from PySide6.QtPrintSupport import QPrintDialog, QPrinter
from PySide6.QtWidgets import (
    QComboBox,
    QGridLayout,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from backend.file_manager import default_filename
from backend.paths import PDF_DIR

logger = logging.getLogger(__name__)

_SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}
_MAX_IMAGES = 4
_THUMB_SIZE = 140


class ImageAnalysisView(QWidget):
    """Composite widget for medical image upload, query, and analysis results.

    Signals:
        analyze_requested: ``(image_paths_joined, query)`` — paths joined
            by ``|`` so the signal stays ``str, str``.
        image_loaded: ``(file_path)`` when a new image is added.
    """

    analyze_requested = Signal(str, str)
    image_loaded = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._image_paths: list[str] = []
        self._build_ui()
        self.setAcceptDrops(True)

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 4, 0, 0)

        # -- Header --
        header = QLabel("<b>Medical Image Analysis</b>")
        layout.addWidget(header)

        # -- Query bar: preset combo + custom query + Analyze button --
        query_bar = QHBoxLayout()

        query_bar.addWidget(QLabel("Prompt:"))

        self.preset_combo = QComboBox()
        self.preset_combo.setMinimumWidth(130)
        self.preset_combo.currentTextChanged.connect(self._on_preset_changed)
        query_bar.addWidget(self.preset_combo)

        self.query_input = QLineEdit()
        self.query_input.setPlaceholderText("Enter your analysis question...")
        self.query_input.returnPressed.connect(self._on_analyze_clicked)
        query_bar.addWidget(self.query_input, stretch=1)

        self.btn_analyze = QPushButton("Analyze")
        self.btn_analyze.setEnabled(False)
        self.btn_analyze.clicked.connect(self._on_analyze_clicked)
        query_bar.addWidget(self.btn_analyze)

        layout.addLayout(query_bar)

        # -- Content: image thumbnails (left) + results (right) --
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Image panel — grid of up to 4 thumbnails + counter label
        image_panel = QWidget()
        image_panel_layout = QVBoxLayout(image_panel)
        image_panel_layout.setContentsMargins(0, 0, 0, 0)

        self.image_count_label = QLabel("No images loaded (max 4)")
        self.image_count_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_count_label.setStyleSheet("color: #888; font-size: 10pt;")
        image_panel_layout.addWidget(self.image_count_label)

        self._thumb_grid = QGridLayout()
        self._thumb_grid.setSpacing(6)
        self._thumb_labels: list[QLabel] = []
        self._thumb_name_labels: list[QLabel] = []

        for i in range(_MAX_IMAGES):
            row, col = divmod(i, 2)

            cell = QVBoxLayout()
            cell.setSpacing(2)

            thumb = QLabel()
            thumb.setAlignment(Qt.AlignmentFlag.AlignCenter)
            thumb.setFixedSize(_THUMB_SIZE, _THUMB_SIZE)
            thumb.setStyleSheet(
                "QLabel { border: 2px dashed #666; border-radius: 4px; "
                "color: #666; font-size: 9pt; }"
            )
            thumb.setText(f"Slot {i + 1}")
            cell.addWidget(thumb, alignment=Qt.AlignmentFlag.AlignCenter)
            self._thumb_labels.append(thumb)

            name_lbl = QLabel("")
            name_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            name_lbl.setStyleSheet("font-size: 8pt; color: #888;")
            name_lbl.setMaximumWidth(_THUMB_SIZE)
            cell.addWidget(name_lbl, alignment=Qt.AlignmentFlag.AlignCenter)
            self._thumb_name_labels.append(name_lbl)

            self._thumb_grid.addLayout(cell, row, col)

        image_panel_layout.addLayout(self._thumb_grid)
        image_panel_layout.addStretch()
        image_panel.setMinimumWidth(_THUMB_SIZE * 2 + 20)
        splitter.addWidget(image_panel)

        # Results panel
        self.results_edit = QTextEdit()
        self.results_edit.setReadOnly(True)
        self.results_edit.setPlaceholderText(
            "Analysis results will appear here..."
        )
        splitter.addWidget(self.results_edit)

        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 3)

        layout.addWidget(splitter, stretch=1)

        # -- Action bar: Save PDF / Print --
        action_bar = QHBoxLayout()
        action_bar.addStretch()

        self.btn_save_pdf = QPushButton("Save PDF")
        self.btn_save_pdf.setEnabled(False)
        self.btn_save_pdf.clicked.connect(self._on_save_pdf_clicked)
        action_bar.addWidget(self.btn_save_pdf)

        self.btn_print = QPushButton("Print")
        self.btn_print.setEnabled(True)
        self.btn_print.clicked.connect(self._on_print_clicked)
        action_bar.addWidget(self.btn_print)

        layout.addLayout(action_bar)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def populate_presets(self, presets: dict[str, str]) -> None:
        """Fill the preset combo box from a dict of {name: prompt}."""
        self.preset_combo.blockSignals(True)
        self.preset_combo.clear()
        self.preset_combo.addItem("Custom")
        for name in presets:
            self.preset_combo.addItem(name)
        self._presets = dict(presets)
        self.preset_combo.blockSignals(False)

    def load_image(self, file_path: str) -> bool:
        """Add an image to the panel (up to _MAX_IMAGES). Returns True on success."""
        if len(self._image_paths) >= _MAX_IMAGES:
            QMessageBox.information(
                self,
                "Image Limit",
                f"Maximum of {_MAX_IMAGES} images already loaded.\n"
                "Clear to start over.",
            )
            return False

        pixmap = QPixmap(file_path)
        if pixmap.isNull():
            logger.warning("Failed to load image: %s", file_path)
            return False

        idx = len(self._image_paths)
        self._image_paths.append(file_path)

        # Update thumbnail
        scaled = pixmap.scaled(
            _THUMB_SIZE - 4, _THUMB_SIZE - 4,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._thumb_labels[idx].setPixmap(scaled)
        self._thumb_labels[idx].setStyleSheet(
            "QLabel { border: 1px solid #444; border-radius: 4px; }"
        )
        self._thumb_name_labels[idx].setText(Path(file_path).name)

        self._update_count_label()
        self.btn_analyze.setEnabled(True)
        self.image_loaded.emit(file_path)
        logger.info("Image added (%d/%d): %s", idx + 1, _MAX_IMAGES, file_path)
        return True

    def set_results(self, text: str) -> None:
        """Display analysis results (HTML-escaped for safety)."""
        safe = html.escape(text, quote=False)
        # Preserve newlines
        formatted = safe.replace("\n", "<br>")
        self.results_edit.setHtml(formatted)
        self.btn_save_pdf.setEnabled(bool(text.strip()))

    def set_busy(self, busy: bool) -> None:
        """Disable/enable controls during analysis."""
        self.btn_analyze.setEnabled(not busy and bool(self._image_paths))
        self.preset_combo.setEnabled(not busy)
        self.query_input.setEnabled(not busy)
        if busy:
            self.btn_save_pdf.setEnabled(False)

    def clear(self) -> None:
        """Reset images, results, and controls."""
        self._image_paths.clear()
        for i in range(_MAX_IMAGES):
            self._thumb_labels[i].clear()
            self._thumb_labels[i].setText(f"Slot {i + 1}")
            self._thumb_labels[i].setStyleSheet(
                "QLabel { border: 2px dashed #666; border-radius: 4px; "
                "color: #666; font-size: 9pt; }"
            )
            self._thumb_name_labels[i].setText("")
        self._update_count_label()
        self.results_edit.clear()
        self.btn_analyze.setEnabled(False)
        self.btn_save_pdf.setEnabled(False)
        self.query_input.clear()

    def get_results_text(self) -> str:
        """Return the plain-text analysis results for export."""
        return self.results_edit.toPlainText()

    def get_image_paths(self) -> list[str]:
        """Return the list of currently loaded image paths."""
        return list(self._image_paths)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _update_count_label(self) -> None:
        n = len(self._image_paths)
        if n == 0:
            self.image_count_label.setText("No images loaded (max 4)")
        else:
            self.image_count_label.setText(
                f"{n} of {_MAX_IMAGES} images loaded"
            )

    # ------------------------------------------------------------------
    # PDF / Print
    # ------------------------------------------------------------------

    def _render_to_printer(self, printer: QPrinter) -> None:
        """Render the images and analysis results to a QPrinter."""
        query = html.escape(self.query_input.text().strip(), quote=False)
        results = html.escape(self.results_edit.toPlainText(), quote=False)
        results_html = results.replace("\n", "<br>")

        # Build image tags in a 2x2 grid, each ~2.5in (180pt)
        valid_images = [
            p for p in self._image_paths[:_MAX_IMAGES] if os.path.isfile(p)
        ]
        img_section = ""
        if valid_images:
            rows_html = ""
            for i in range(0, len(valid_images), 2):
                row_cells = ""
                for img_path in valid_images[i:i + 2]:
                    img_url = QUrl.fromLocalFile(img_path).toString()
                    img_name = html.escape(Path(img_path).name, quote=False)
                    row_cells += (
                        f'<td style="text-align:center; padding:4px;">'
                        f'<img src="{img_url}" width="180"><br>'
                        f'<span style="font-size:9pt; color:#555;">'
                        f'{img_name}</span></td>'
                    )
                rows_html += f'<tr>{row_cells}</tr>'
            img_section = (
                f'<table align="center">{rows_html}</table><br>'
            )

        doc_html = (
            f'<h2 style="text-align:center;">Medical Image Analysis</h2>'
            f'{img_section}'
            f'<p><b>Query:</b> {query}</p>'
            f'<hr>'
            f'<p>{results_html}</p>'
        )

        doc = QTextDocument()
        doc.setHtml(doc_html)
        doc.print_(printer)

    def _on_save_pdf_clicked(self) -> None:
        """Save the analysis as a PDF to the Generated PDF directory."""
        filename = default_filename("ImageAnalysis", "pdf")
        filepath = os.path.join(PDF_DIR, filename)

        printer = QPrinter(QPrinter.PrinterMode.HighResolution)
        printer.setOutputFormat(QPrinter.OutputFormat.PdfFormat)
        printer.setOutputFileName(filepath)

        page_layout = QPageLayout(
            QPageSize(QPageSize.PageSizeId.A4),
            QPageLayout.Orientation.Portrait,
            QMarginsF(15, 15, 15, 15),
            QPageLayout.Unit.Millimeter,
        )
        printer.setPageLayout(page_layout)

        try:
            self._render_to_printer(printer)
            logger.info("PDF saved: %s", filepath)
            QMessageBox.information(
                self,
                "PDF Saved",
                f"Analysis saved to:\n{filepath}",
            )
        except Exception:
            logger.exception("Failed to save PDF")
            QMessageBox.critical(
                self,
                "PDF Error",
                "Failed to save PDF. Check the log for details.",
            )

    def _on_print_clicked(self) -> None:
        """Let the user pick a saved PDF from the Generated PDF folder and print it."""
        # Gather PDF files, sorted newest first
        pdf_files = sorted(
            (f for f in Path(PDF_DIR).glob("*.pdf") if f.is_file()),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not pdf_files:
            QMessageBox.information(
                self,
                "No PDFs",
                "No saved PDFs found.\nUse 'Save PDF' first to generate one.",
            )
            return

        names = [f.name for f in pdf_files]
        chosen, ok = QInputDialog.getItem(
            self,
            "Print PDF",
            "Select a PDF to print:",
            names,
            0,       # default to newest
            False,   # not editable
        )
        if not ok or not chosen:
            return

        selected_path = Path(PDF_DIR) / chosen

        printer = QPrinter(QPrinter.PrinterMode.HighResolution)
        dialog = QPrintDialog(printer, self)
        if dialog.exec() == QPrintDialog.DialogCode.Accepted:
            try:
                # Render the PDF as an image-per-page for faithful reproduction
                from PySide6.QtPdf import QPdfDocument

                pdf_doc = QPdfDocument(self)
                pdf_doc.load(str(selected_path))

                from PySide6.QtGui import QPainter

                painter = QPainter()
                if not painter.begin(printer):
                    raise RuntimeError("Failed to start print painter")

                page_rect = printer.pageRect(QPrinter.Unit.DevicePixel)
                for i in range(pdf_doc.pageCount()):
                    if i > 0:
                        printer.newPage()
                    # Render PDF page to an image at print resolution
                    pdf_size = pdf_doc.pagePointSize(i)
                    scale = min(
                        page_rect.width() / pdf_size.width(),
                        page_rect.height() / pdf_size.height(),
                    )
                    render_size = pdf_size.toSize() * scale
                    image = pdf_doc.render(i, render_size)
                    painter.drawImage(0, 0, image)

                painter.end()
                pdf_doc.close()
                logger.info("Printed PDF: %s", selected_path.name)
            except Exception:
                logger.exception("Print failed")
                QMessageBox.critical(
                    self,
                    "Print Error",
                    "Failed to print. Check the log for details.",
                )

    # ------------------------------------------------------------------
    # Drag-and-drop
    # ------------------------------------------------------------------

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                path = url.toLocalFile()
                if Path(path).suffix.lower() in _SUPPORTED_EXTENSIONS:
                    event.acceptProposedAction()
                    return
        event.ignore()

    def dropEvent(self, event: QDropEvent) -> None:
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if Path(path).suffix.lower() in _SUPPORTED_EXTENSIONS:
                self.load_image(path)
        event.acceptProposedAction()

    # ------------------------------------------------------------------
    # Internal slots
    # ------------------------------------------------------------------

    def _on_preset_changed(self, text: str) -> None:
        """Auto-fill the query input when a preset is selected."""
        if text == "Custom":
            self.query_input.clear()
            self.query_input.setFocus()
        else:
            prompt = getattr(self, "_presets", {}).get(text, "")
            self.query_input.setText(prompt)

    def _on_analyze_clicked(self) -> None:
        """Emit analyze_requested if we have images and a query."""
        if not self._image_paths:
            return
        query = self.query_input.text().strip()
        if not query:
            # Fall back to general preset
            query = getattr(self, "_presets", {}).get(
                "General",
                "Describe this medical image and identify any abnormalities.",
            )
            self.query_input.setText(query)
        # Join paths with | separator for the signal
        self.analyze_requested.emit("|".join(self._image_paths), query)
