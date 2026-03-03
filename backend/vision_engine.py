"""
Medical image analysis engine for STT Transcriber.

Sends medical images to a local LLM server (LM Studio or Ollama) that
has a vision-capable model loaded (e.g. MedGemma).  The image is
base64-encoded and sent via the server's chat/generate API.

**Not for clinical diagnosis** — research/development tool only.
"""

import base64
import json
import logging
import mimetypes
from pathlib import Path

import requests

from backend.errors import VisionEngineError

logger = logging.getLogger(__name__)

VISION_PRESETS: dict[str, str] = {
    "General": (
        "Describe this medical image. Identify any abnormalities, "
        "notable findings, or areas of concern."
    ),
    "Chest X-Ray": (
        "Analyze this chest X-ray. Evaluate heart size, lung fields, "
        "mediastinum, costophrenic angles, and note any effusions, "
        "opacities, or abnormalities."
    ),
    "CT Scan": (
        "Analyze this CT scan. Evaluate anatomical structures, identify "
        "any lesions, masses, or abnormal contrast enhancement patterns."
    ),
    "MRI": (
        "Analyze this MRI. Evaluate signal characteristics, structural "
        "changes, edema, and any abnormal findings."
    ),
    "Dermatology": (
        "Analyze this dermatological image. Describe the lesion morphology, "
        "borders, color distribution, and provide differential diagnoses."
    ),
    "Fundoscopy": (
        "Analyze this fundoscopic image. Evaluate the optic disc, macula, "
        "retinal vessels, and note any hemorrhages or abnormalities."
    ),
    "Histopathology": (
        "Analyze this histopathology slide. Evaluate tissue architecture, "
        "cell morphology, staining patterns, and any pathological findings."
    ),
}

# Image analysis may take longer than plain text generation
_REQUEST_TIMEOUT = 120


def _image_to_base64(image_path: str) -> tuple[str, str]:
    """Read an image file and return ``(base64_data, mime_type)``.

    Raises:
        VisionEngineError: If the file cannot be read.
    """
    path = Path(image_path)
    if not path.is_file():
        raise VisionEngineError(f"Image file not found: {image_path}")

    mime, _ = mimetypes.guess_type(str(path))
    if mime is None:
        mime = "image/png"  # safe fallback

    try:
        raw = path.read_bytes()
        return base64.b64encode(raw).decode("ascii"), mime
    except OSError as exc:
        raise VisionEngineError(
            f"Failed to read image file: {exc}"
        ) from exc


def analyze_image(
    image_path: str,
    query: str,
    endpoint: str,
    model: str,
    provider: str,
) -> str:
    """Send a medical image to a local LLM server for analysis.

    Args:
        image_path: Path to the image file on disk.
        query: Natural-language question about the image.
        endpoint: LLM server URL (e.g.
            ``http://localhost:1234/v1/chat/completions``).
        model: Model name (e.g. ``"medgemma-1.5-4b-it"``).
        provider: ``"lm_studio"`` or ``"ollama"``.

    Returns:
        The model's analysis text.

    Raises:
        VisionEngineError: If the image cannot be read or the server
            returns an error.
    """
    b64_data, mime = _image_to_base64(image_path)

    if provider == "lm_studio":
        return _call_lm_studio(endpoint, model, query, [(b64_data, mime)])
    return _call_ollama(endpoint, model, query, [(b64_data, mime)])


def synthesize_series(
    per_image_results: list[tuple[str, str]],
    query: str,
    endpoint: str,
    model: str,
    provider: str,
) -> str:
    """Synthesize per-image analyses into a combined comparative report.

    This is a text-only call (no images) that asks the model to compare
    and correlate findings across all slices in a series.

    Args:
        per_image_results: List of ``(filename, analysis_text)`` pairs.
        query: The original user query for context.
        endpoint: LLM server URL.
        model: Model name.
        provider: ``"lm_studio"`` or ``"ollama"``.

    Returns:
        Combined synthesis text.

    Raises:
        VisionEngineError: On server/network errors.
    """
    sections = "\n\n".join(
        f"[{fname}]\n{text}" for fname, text in per_image_results
    )
    prompt = (
        f"You previously analyzed {len(per_image_results)} images from the "
        f"same imaging series individually. The original query was:\n"
        f"\"{query}\"\n\n"
        f"Here are the per-image findings:\n\n{sections}\n\n"
        f"Now provide a COMBINED ANALYSIS that:\n"
        f"1. Compares findings across all slices\n"
        f"2. Notes any changes or progression between slices\n"
        f"3. Identifies findings that are consistent across images\n"
        f"4. Provides an overall impression of the series as a whole\n"
        f"5. Notes any areas that warrant further investigation"
    )

    logger.info("Synthesizing %d per-image analyses", len(per_image_results))

    if provider == "lm_studio":
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
        }
        try:
            resp = requests.post(endpoint, json=payload, timeout=_REQUEST_TIMEOUT)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except (requests.RequestException, json.JSONDecodeError, KeyError,
                IndexError) as exc:
            raise VisionEngineError(
                f"Series synthesis failed: {exc}"
            ) from exc
    else:
        # Ollama
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0},
        }
        try:
            resp = requests.post(endpoint, json=payload, timeout=_REQUEST_TIMEOUT)
            resp.raise_for_status()
            data = resp.json()
            return data.get("response", "")
        except (requests.RequestException, json.JSONDecodeError, KeyError) as exc:
            raise VisionEngineError(
                f"Series synthesis failed: {exc}"
            ) from exc


def _call_lm_studio(
    endpoint: str,
    model: str,
    query: str,
    images: list[tuple[str, str]],
) -> str:
    """Call LM Studio's OpenAI-compatible vision API."""
    content: list[dict] = []
    for b64_data, mime in images:
        data_url = f"data:{mime};base64,{b64_data}"
        content.append({
            "type": "image_url",
            "image_url": {"url": data_url},
        })
    content.append({"type": "text", "text": query})

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "temperature": 0,
    }
    n_images = len(images)
    logger.info("Sending %d image(s) to LM Studio (%s)", n_images, model)
    try:
        resp = requests.post(endpoint, json=payload, timeout=_REQUEST_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]
    except requests.ConnectionError as exc:
        raise VisionEngineError(
            f"Cannot connect to LM Studio at {endpoint}. "
            "Is the LM Studio server running?"
        ) from exc
    except requests.Timeout as exc:
        raise VisionEngineError(
            f"LM Studio request timed out after {_REQUEST_TIMEOUT}s. "
            "Image analysis can be slow — try a smaller image or simpler query."
        ) from exc
    except requests.HTTPError as exc:
        raise VisionEngineError(
            f"LM Studio returned HTTP error: {exc.response.status_code} — "
            f"{exc.response.text[:200]}"
        ) from exc
    except (json.JSONDecodeError, KeyError, IndexError) as exc:
        raise VisionEngineError(
            f"Unexpected response from LM Studio: {exc}"
        ) from exc


def _call_ollama(
    endpoint: str,
    model: str,
    query: str,
    images: list[tuple[str, str]],
) -> str:
    """Call Ollama's generate API with one or more images."""
    payload = {
        "model": model,
        "prompt": query,
        "images": [b64_data for b64_data, _mime in images],
        "stream": False,
        "options": {"temperature": 0},
    }
    try:
        resp = requests.post(endpoint, json=payload, timeout=_REQUEST_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        return data.get("response", "")
    except requests.ConnectionError as exc:
        raise VisionEngineError(
            f"Cannot connect to Ollama at {endpoint}. "
            "Is the Ollama server running?"
        ) from exc
    except requests.Timeout as exc:
        raise VisionEngineError(
            f"Ollama request timed out after {_REQUEST_TIMEOUT}s. "
            "Image analysis can be slow — try a smaller image or simpler query."
        ) from exc
    except requests.HTTPError as exc:
        raise VisionEngineError(
            f"Ollama returned HTTP error: {exc.response.status_code} — "
            f"{exc.response.text[:200]}"
        ) from exc
    except (json.JSONDecodeError, KeyError) as exc:
        raise VisionEngineError(
            f"Unexpected response from Ollama: {exc}"
        ) from exc
