"""
Export module for STT Transcriber.

Provides functions to export transcript segments and SOAP notes
to TXT, CSV, and Markdown formats.
"""

import csv
import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


def default_filename(prefix: str, ext: str) -> str:
    """Generate a timestamped default filename.

    Args:
        prefix: File prefix, e.g. ``"Transcript"`` or ``"SOAP"``.
        ext: File extension without dot, e.g. ``"txt"``.

    Returns:
        Filename like ``"Transcript_20260219_143022.txt"``.
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{ts}.{ext}"


def export_txt(segments: list[dict], file_path: str) -> None:
    """Export transcript segments as timestamped plain text.

    Args:
        segments: List of ``{"timestamp": str, "text": str}`` dicts.
        file_path: Destination file path.

    Raises:
        OSError: If the file cannot be written.
    """
    path = Path(file_path)
    with path.open("w", encoding="utf-8") as f:
        for seg in segments:
            f.write(f"[{seg['timestamp']}] {seg['text']}\n")
    logger.info("Exported TXT to %s", file_path)


def export_csv(segments: list[dict], file_path: str) -> None:
    """Export transcript segments as CSV with Timestamp and Text columns.

    Args:
        segments: List of ``{"timestamp": str, "text": str}`` dicts.
        file_path: Destination file path.

    Raises:
        OSError: If the file cannot be written.
    """
    path = Path(file_path)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Timestamp", "Text"])
        for seg in segments:
            writer.writerow([seg["timestamp"], seg["text"]])
    logger.info("Exported CSV to %s", file_path)


def export_md(segments: list[dict], file_path: str) -> None:
    """Export transcript segments as Markdown.

    Args:
        segments: List of ``{"timestamp": str, "text": str}`` dicts.
        file_path: Destination file path.

    Raises:
        OSError: If the file cannot be written.
    """
    path = Path(file_path)
    with path.open("w", encoding="utf-8") as f:
        f.write("## Transcript\n\n")
        for seg in segments:
            f.write(f"- **[{seg['timestamp']}]** {seg['text']}\n")
    logger.info("Exported Markdown to %s", file_path)


def export_soap_txt(soap: dict, file_path: str) -> None:
    """Export SOAP notes as labeled plain text blocks.

    Args:
        soap: Dict with keys ``"subjective"``, ``"objective"``,
            ``"assessment"``, ``"plan"``.
        file_path: Destination file path.

    Raises:
        OSError: If the file cannot be written.
    """
    path = Path(file_path)
    sections = [
        ("SUBJECTIVE", soap.get("subjective", "")),
        ("OBJECTIVE", soap.get("objective", "")),
        ("ASSESSMENT", soap.get("assessment", "")),
        ("PLAN", soap.get("plan", "")),
    ]
    with path.open("w", encoding="utf-8") as f:
        for label, content in sections:
            f.write(f"{'=' * 40}\n")
            f.write(f"{label}\n")
            f.write(f"{'=' * 40}\n")
            f.write(f"{content}\n\n")
    logger.info("Exported SOAP notes to %s", file_path)
