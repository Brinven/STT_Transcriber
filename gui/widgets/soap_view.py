"""
SOAP notes display widget for STT Transcriber.

Shows the four SOAP sections (Subjective, Objective, Assessment, Plan)
in labeled, read-only text areas. Visible only in medical mode after
the user clicks SOAPify.
"""

import html
import logging

from PySide6.QtWidgets import (
    QLabel,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

logger = logging.getLogger(__name__)

_SECTIONS = ("Subjective", "Objective", "Assessment", "Plan")


class SoapView(QWidget):
    """Displays structured SOAP notes in four labeled sections."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 4, 0, 0)

        self._editors: dict[str, QTextEdit] = {}

        for section in _SECTIONS:
            label = QLabel(f"<b>{html.escape(section)}</b>")
            layout.addWidget(label)

            editor = QTextEdit()
            editor.setReadOnly(True)
            editor.setMaximumHeight(120)
            editor.setPlaceholderText(f"{section} content will appear here...")
            layout.addWidget(editor)

            self._editors[section.lower()] = editor

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
