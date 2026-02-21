"""
SOAP notes display widget for STT Transcriber.

Shows the four SOAP sections (Subjective, Objective, Assessment, Plan)
in labeled, read-only text areas.  Supports two layouts:

- **grid** (default): 2x2 grid — S/O on top, A/P on bottom.
- **vertical**: All four sections stacked vertically.

Visible only in medical mode after the user clicks SOAPify.
"""

import html
import logging

from PySide6.QtWidgets import (
    QGridLayout,
    QLabel,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

logger = logging.getLogger(__name__)

_SECTIONS = ("Subjective", "Objective", "Assessment", "Plan")

# Grid positions: (row, col)
_GRID_POS = {
    "Subjective": (0, 0),
    "Objective": (0, 1),
    "Assessment": (1, 0),
    "Plan": (1, 1),
}


class SoapView(QWidget):
    """Displays structured SOAP notes in four labeled sections."""

    def __init__(
        self, layout_mode: str = "grid", parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._editors: dict[str, QTextEdit] = {}
        self._labels: dict[str, QLabel] = {}
        self._containers: dict[str, QWidget] = {}
        self._layout_mode = ""
        self._build_sections()
        self.set_layout_mode(layout_mode)

    def _build_sections(self) -> None:
        """Create label + editor pairs (not yet placed in any layout)."""
        for section in _SECTIONS:
            label = QLabel(f"<b>{html.escape(section)}</b>")
            editor = QTextEdit()
            editor.setReadOnly(True)
            editor.setPlaceholderText(f"{section} content will appear here...")

            # Wrap label + editor in a small container widget
            container = QWidget()
            vbox = QVBoxLayout(container)
            vbox.setContentsMargins(0, 0, 0, 0)
            vbox.setSpacing(2)
            vbox.addWidget(label)
            vbox.addWidget(editor)

            key = section.lower()
            self._editors[key] = editor
            self._labels[key] = label
            self._containers[section] = container

    def set_layout_mode(self, mode: str) -> None:
        """Switch between ``"grid"`` (2x2) and ``"vertical"`` (stacked).

        This re-parents all section widgets into the new layout.
        """
        mode = mode if mode in ("grid", "vertical") else "grid"
        if mode == self._layout_mode:
            return

        self._layout_mode = mode

        # Remove the old layout if any
        old = self.layout()
        if old is not None:
            # Remove all widgets from the old layout
            while old.count():
                item = old.takeAt(0)
                if item.widget():
                    item.widget().setParent(None)  # type: ignore[call-overload]
            # Qt requires deleting the old layout via a helper
            QWidget().setLayout(old)

        if mode == "grid":
            grid = QGridLayout(self)
            grid.setContentsMargins(0, 4, 0, 0)
            grid.setSpacing(6)
            for section in _SECTIONS:
                row, col = _GRID_POS[section]
                grid.addWidget(self._containers[section], row, col)
            # Let rows stretch equally
            grid.setRowStretch(0, 1)
            grid.setRowStretch(1, 1)
            grid.setColumnStretch(0, 1)
            grid.setColumnStretch(1, 1)
        else:
            vbox = QVBoxLayout(self)
            vbox.setContentsMargins(0, 4, 0, 0)
            vbox.setSpacing(4)
            for section in _SECTIONS:
                self._containers[section].findChild(QTextEdit).setMaximumHeight(120)
                vbox.addWidget(self._containers[section])

        # In grid mode remove height caps so sections fill available space
        if mode == "grid":
            for section in _SECTIONS:
                self._containers[section].findChild(QTextEdit).setMaximumHeight(16777215)

        logger.info("SOAP layout set to %s", mode)

    # ------------------------------------------------------------------
    # Data access
    # ------------------------------------------------------------------

    def set_soap(self, soap: dict) -> None:
        """Populate all four SOAP sections.

        Args:
            soap: Dict with keys ``"subjective"``, ``"objective"``,
                ``"assessment"``, ``"plan"``.
        """
        for key, editor in self._editors.items():
            text = soap.get(key, "")
            editor.setPlainText(text)
        logger.info("SOAP view populated")

    def clear(self) -> None:
        """Clear all SOAP sections."""
        for editor in self._editors.values():
            editor.clear()

    def get_full_text(self) -> str:
        """Return all sections as formatted plain text.

        Returns:
            Multi-line string with section headers and content.
        """
        parts: list[str] = []
        for section in _SECTIONS:
            key = section.lower()
            content = self._editors[key].toPlainText()
            parts.append(f"{section}:\n{content}")
        return "\n\n".join(parts)

    def get_soap_dict(self) -> dict:
        """Return the current SOAP content as a dict.

        Returns:
            Dict with keys ``"subjective"``, ``"objective"``,
            ``"assessment"``, ``"plan"``.
        """
        return {
            key: editor.toPlainText()
            for key, editor in self._editors.items()
        }
