"""
Medical image analysis widget for STT Transcriber.

Provides an image preview, preset/custom query input, and a results
panel for MedGemma vision-language analysis.  Supports drag-and-drop
of image files.

Visible only in medical mode after the user clicks Upload Image.
"""

import html
import logging
from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QDragEnterEvent, QDropEvent, QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

logger = logging.getLogger(__name__)

_SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}


class ImageAnalysisView(QWidget):
    """Composite widget for medical image upload, query, and analysis results.

    Signals:
        analyze_requested: ``(image_path, query)`` when user clicks Analyze.
        image_loaded: ``(file_path)`` when a new image is loaded.
    """

    analyze_requested = Signal(str, str)
    image_loaded = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._image_path: str = ""
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

        # -- Content: image preview (left) + results (right) --
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Image preview
        self.image_label = QLabel("Drop image here\nor use Upload Image")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumSize(200, 200)
        self.image_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self.image_label.setStyleSheet(
            "QLabel { border: 2px dashed #888; border-radius: 8px; "
            "color: #888; font-size: 13pt; }"
        )
        splitter.addWidget(self.image_label)

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
        """Load and display an image file. Returns True on success."""
        pixmap = QPixmap(file_path)
        if pixmap.isNull():
            logger.warning("Failed to load image: %s", file_path)
            return False

        self._image_path = file_path

        # Scale to fit the label while keeping aspect ratio
        scaled = pixmap.scaled(
            self.image_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.image_label.setPixmap(scaled)
        self.image_label.setStyleSheet(
            "QLabel { border: 1px solid #444; border-radius: 4px; }"
        )

        self.btn_analyze.setEnabled(True)
        self.image_loaded.emit(file_path)
        logger.info("Image loaded: %s", file_path)
        return True

    def set_results(self, text: str) -> None:
        """Display analysis results (HTML-escaped for safety)."""
        safe = html.escape(text, quote=False)
        # Preserve newlines
        formatted = safe.replace("\n", "<br>")
        self.results_edit.setHtml(formatted)

    def set_busy(self, busy: bool) -> None:
        """Disable/enable controls during analysis."""
        self.btn_analyze.setEnabled(not busy and bool(self._image_path))
        self.preset_combo.setEnabled(not busy)
        self.query_input.setEnabled(not busy)

    def clear(self) -> None:
        """Reset image, results, and controls."""
        self._image_path = ""
        self.image_label.clear()
        self.image_label.setText("Drop image here\nor use Upload Image")
        self.image_label.setStyleSheet(
            "QLabel { border: 2px dashed #888; border-radius: 8px; "
            "color: #888; font-size: 13pt; }"
        )
        self.results_edit.clear()
        self.btn_analyze.setEnabled(False)
        self.query_input.clear()

    def get_results_text(self) -> str:
        """Return the plain-text analysis results for export."""
        return self.results_edit.toPlainText()

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
                return
        event.ignore()

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
        """Emit analyze_requested if we have an image and query."""
        if not self._image_path:
            return
        query = self.query_input.text().strip()
        if not query:
            # Fall back to general preset
            query = getattr(self, "_presets", {}).get(
                "General",
                "Describe this medical image and identify any abnormalities.",
            )
            self.query_input.setText(query)
        self.analyze_requested.emit(self._image_path, query)
