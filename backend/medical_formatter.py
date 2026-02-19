"""
Medical SOAP formatting via local LLM server.

Sends a medical transcript to an Ollama or LM Studio server running
on localhost and parses the response into SOAP note sections.
"""

import json
import logging
import re

import requests

from backend.errors import LLMConnectionError

logger = logging.getLogger(__name__)

SOAP_PROMPT_TEMPLATE = """\
You are a medical documentation assistant. Given the following medical \
transcription, format it into a SOAP note with four clearly labeled sections.

Use EXACTLY these section headers on their own lines:
Subjective:
Objective:
Assessment:
Plan:

Place the relevant content under each header. If a section has no relevant \
content, write "N/A" under that header.

--- Transcript ---
{transcript}
--- End Transcript ---

Format the transcript above as a SOAP note now."""

# Timeout for LLM inference (seconds)
_REQUEST_TIMEOUT = 60


def format_soap(
    transcript: str,
    endpoint: str,
    model: str,
    provider: str,
) -> dict:
    """Send transcript to a local LLM and parse the SOAP response.

    Args:
        transcript: The medical transcript text.
        endpoint: LLM server URL (e.g. ``http://localhost:11434/api/generate``).
        model: Model name to use (e.g. ``"medllama2"``).
        provider: ``"ollama"`` or ``"lm_studio"``.

    Returns:
        Dict with keys ``"subjective"``, ``"objective"``,
        ``"assessment"``, ``"plan"``.

    Raises:
        LLMConnectionError: If the server is unreachable or returns an error.
    """
    prompt = SOAP_PROMPT_TEMPLATE.format(transcript=transcript)

    if provider == "lm_studio":
        response_text = _call_lm_studio(endpoint, model, prompt)
    else:
        response_text = _call_ollama(endpoint, model, prompt)

    return _parse_soap(response_text)


def _call_ollama(endpoint: str, model: str, prompt: str) -> str:
    """Call the Ollama API and return the response text."""
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
    }
    try:
        resp = requests.post(
            endpoint,
            json=payload,
            timeout=_REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("response", "")
    except requests.ConnectionError as exc:
        raise LLMConnectionError(
            f"Cannot connect to Ollama at {endpoint}. "
            "Is the Ollama server running?"
        ) from exc
    except requests.Timeout as exc:
        raise LLMConnectionError(
            f"Ollama request timed out after {_REQUEST_TIMEOUT}s. "
            "The model may be too large or the server is overloaded."
        ) from exc
    except requests.HTTPError as exc:
        raise LLMConnectionError(
            f"Ollama returned HTTP error: {exc.response.status_code} — "
            f"{exc.response.text[:200]}"
        ) from exc
    except (json.JSONDecodeError, KeyError) as exc:
        raise LLMConnectionError(
            f"Unexpected response from Ollama: {exc}"
        ) from exc


def _call_lm_studio(endpoint: str, model: str, prompt: str) -> str:
    """Call the LM Studio (OpenAI-compatible) API and return the response text."""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
    }
    try:
        resp = requests.post(
            endpoint,
            json=payload,
            timeout=_REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]
    except requests.ConnectionError as exc:
        raise LLMConnectionError(
            f"Cannot connect to LM Studio at {endpoint}. "
            "Is the LM Studio server running?"
        ) from exc
    except requests.Timeout as exc:
        raise LLMConnectionError(
            f"LM Studio request timed out after {_REQUEST_TIMEOUT}s. "
            "The model may be too large or the server is overloaded."
        ) from exc
    except requests.HTTPError as exc:
        raise LLMConnectionError(
            f"LM Studio returned HTTP error: {exc.response.status_code} — "
            f"{exc.response.text[:200]}"
        ) from exc
    except (json.JSONDecodeError, KeyError, IndexError) as exc:
        raise LLMConnectionError(
            f"Unexpected response from LM Studio: {exc}"
        ) from exc


def _parse_soap(text: str) -> dict:
    """Parse LLM output into SOAP sections.

    Looks for section headers like "Subjective:", "Objective:", etc.
    If headers are not found, the entire text is placed under "assessment"
    as a fallback.

    Returns:
        Dict with keys: subjective, objective, assessment, plan.
    """
    pattern = re.compile(
        r"(?:^|\n)\s*(?:#+\s*)?"  # optional markdown heading
        r"(subjective|objective|assessment|plan)"
        r"\s*:?\s*\n",
        re.IGNORECASE,
    )

    splits = pattern.split(text)

    # Build a mapping from section name → content
    sections: dict[str, str] = {}
    i = 1  # skip the text before the first header
    while i < len(splits) - 1:
        key = splits[i].lower().strip()
        content = splits[i + 1].strip()
        sections[key] = content
        i += 2

    # If we found at least one section, fill missing ones with ""
    if sections:
        return {
            "subjective": sections.get("subjective", ""),
            "objective": sections.get("objective", ""),
            "assessment": sections.get("assessment", ""),
            "plan": sections.get("plan", ""),
        }

    # Fallback: headers not found — put everything in assessment
    logger.warning("SOAP section headers not found in LLM response, using fallback")
    return {
        "subjective": "",
        "objective": "",
        "assessment": text.strip(),
        "plan": "",
    }
