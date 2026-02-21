"""
Configuration management for STT Transcriber.

Wraps a JSON config file with typed getters/setters and
defensive merging with defaults on load.
"""

import json
import logging
import os
from typing import Any, Optional

from backend.paths import CONFIG_DIR

logger = logging.getLogger(__name__)

CONFIG_FILE = os.path.join(CONFIG_DIR, "config.json")


def _get_defaults() -> dict[str, Any]:
    """Return the full default configuration."""
    return {
        "mode": "general",
        "audio_source": "microphone",
        "audio_device_index": None,
        "whisper_model_size": "base",
        "medasr_device": "auto",
        "llm_endpoint": "http://localhost:1234/v1/chat/completions",
        "llm_model": "medgemma-1.5-4b-it",
        "llm_provider": "lm_studio",
        "export_directory": "",
        "font_size": 12,
        "diarization_enabled": False,
        "last_import_dir": "",
        "hf_token": "",
        "soap_layout": "grid",
    }


class ConfigManager:
    """Manages application configuration backed by a JSON file."""

    def __init__(self, config_path: str = CONFIG_FILE) -> None:
        self._path = config_path
        self._config: dict[str, Any] = {}
        self.load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load config from disk, merging with defaults for any missing keys."""
        defaults = _get_defaults()
        if os.path.exists(self._path):
            try:
                with open(self._path, "r", encoding="utf-8") as f:
                    stored = json.load(f)
                # Merge: defaults first, then overwrite with stored values
                defaults.update(stored)
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("Failed to load config, using defaults: %s", e)
        self._config = defaults
        # Persist immediately so that first-run creates the file
        self.save()

    def save(self) -> None:
        """Write current config to disk."""
        try:
            os.makedirs(os.path.dirname(self._path), exist_ok=True)
            with open(self._path, "w", encoding="utf-8") as f:
                json.dump(self._config, f, indent=2)
        except OSError as e:
            logger.error("Failed to save config: %s", e)

    # ------------------------------------------------------------------
    # Generic access
    # ------------------------------------------------------------------

    def get(self, key: str, default: Any = None) -> Any:
        return self._config.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self._config[key] = value
        self.save()

    # ------------------------------------------------------------------
    # Typed properties
    # ------------------------------------------------------------------

    @property
    def mode(self) -> str:
        return self._config.get("mode", "general")

    @mode.setter
    def mode(self, value: str) -> None:
        self.set("mode", value)

    @property
    def audio_source(self) -> str:
        return self._config.get("audio_source", "microphone")

    @audio_source.setter
    def audio_source(self, value: str) -> None:
        self.set("audio_source", value)

    @property
    def audio_device_index(self) -> Optional[int]:
        return self._config.get("audio_device_index")

    @audio_device_index.setter
    def audio_device_index(self, value: Optional[int]) -> None:
        self.set("audio_device_index", value)

    @property
    def whisper_model_size(self) -> str:
        return self._config.get("whisper_model_size", "base")

    @whisper_model_size.setter
    def whisper_model_size(self, value: str) -> None:
        self.set("whisper_model_size", value)

    @property
    def medasr_device(self) -> str:
        return self._config.get("medasr_device", "auto")

    @medasr_device.setter
    def medasr_device(self, value: str) -> None:
        self.set("medasr_device", value)

    @property
    def llm_endpoint(self) -> str:
        return self._config.get("llm_endpoint", "http://localhost:11434/api/generate")

    @llm_endpoint.setter
    def llm_endpoint(self, value: str) -> None:
        self.set("llm_endpoint", value)

    @property
    def llm_model(self) -> str:
        return self._config.get("llm_model", "medllama2")

    @llm_model.setter
    def llm_model(self, value: str) -> None:
        self.set("llm_model", value)

    @property
    def llm_provider(self) -> str:
        return self._config.get("llm_provider", "ollama")

    @llm_provider.setter
    def llm_provider(self, value: str) -> None:
        self.set("llm_provider", value)

    @property
    def export_directory(self) -> str:
        return self._config.get("export_directory", "")

    @export_directory.setter
    def export_directory(self, value: str) -> None:
        self.set("export_directory", value)

    @property
    def font_size(self) -> int:
        return self._config.get("font_size", 12)

    @font_size.setter
    def font_size(self, value: int) -> None:
        self.set("font_size", value)

    @property
    def diarization_enabled(self) -> bool:
        return self._config.get("diarization_enabled", False)

    @diarization_enabled.setter
    def diarization_enabled(self, value: bool) -> None:
        self.set("diarization_enabled", value)

    @property
    def last_import_dir(self) -> str:
        return self._config.get("last_import_dir", "")

    @last_import_dir.setter
    def last_import_dir(self, value: str) -> None:
        self.set("last_import_dir", value)

    @property
    def hf_token(self) -> str:
        return self._config.get("hf_token", "")

    @hf_token.setter
    def hf_token(self, value: str) -> None:
        self.set("hf_token", value)

    @property
    def soap_layout(self) -> str:
        return self._config.get("soap_layout", "grid")

    @soap_layout.setter
    def soap_layout(self, value: str) -> None:
        self.set("soap_layout", value)
