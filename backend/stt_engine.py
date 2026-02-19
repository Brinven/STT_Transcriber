"""
Speech-to-text engine for STT Transcriber.

Wraps faster-whisper for general-mode transcription.
"""

import logging
from typing import Optional

import numpy as np

from backend.errors import STTEngineError

logger = logging.getLogger(__name__)

VALID_MODEL_SIZES = ("tiny", "base", "small", "medium", "large-v2", "large-v3")


class STTEngine:
    """Offline speech-to-text using faster-whisper (CTranslate2).

    Args:
        model_size: Whisper model size. One of: tiny, base, small,
            medium, large-v2, large-v3.
    """

    def __init__(self, model_size: str = "base") -> None:
        if model_size not in VALID_MODEL_SIZES:
            raise ValueError(
                f"Invalid model size '{model_size}'. "
                f"Must be one of: {', '.join(VALID_MODEL_SIZES)}"
            )
        self._model_size = model_size
        self._model = None  # type: ignore[assignment]

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load_model(self) -> None:
        """Download (if needed) and load the Whisper model.

        This is a blocking call — run it on a worker thread.
        First-time use will download the model from HuggingFace.

        Raises:
            STTEngineError: If model loading fails.
        """
        if self._model is not None:
            logger.info("Model already loaded, skipping")
            return

        logger.info("Loading faster-whisper model: %s", self._model_size)
        try:
            from faster_whisper import WhisperModel

            self._model = WhisperModel(
                self._model_size,
                device="cpu",
                compute_type="int8",
            )
            logger.info("Model loaded successfully")
        except Exception as exc:
            raise STTEngineError(
                f"Failed to load Whisper model '{self._model_size}': {exc}"
            ) from exc

    def transcribe(self, audio: np.ndarray) -> str:
        """Transcribe a chunk of 16 kHz mono float32 audio.

        Args:
            audio: 1-D float32 numpy array at 16 kHz.

        Returns:
            Transcribed text, or empty string for silence/noise.

        Raises:
            STTEngineError: If the model is not loaded or transcription fails.
        """
        if self._model is None:
            raise STTEngineError("Model not loaded — call load_model() first")

        if len(audio) == 0:
            return ""

        try:
            segments, _info = self._model.transcribe(
                audio,
                vad_filter=True,
                language="en",
            )
            text = " ".join(seg.text.strip() for seg in segments)
            return text.strip()
        except Exception as exc:
            raise STTEngineError(f"Transcription failed: {exc}") from exc

    def unload_model(self) -> None:
        """Release the model from memory."""
        if self._model is not None:
            self._model = None
            logger.info("STT model unloaded")


class MedASREngine:
    """Medical speech-to-text using Google MedASR (Conformer CTC).

    Uses the ``google/medasr`` model from HuggingFace via the
    ``transformers`` automatic-speech-recognition pipeline.

    Args:
        device: ``"auto"`` (CUDA if available, else CPU), ``"cpu"``,
            or ``"cuda"``.
    """

    def __init__(self, device: str = "auto") -> None:
        self._device_str = device
        self._pipeline = None  # type: ignore[assignment]

    @property
    def is_loaded(self) -> bool:
        return self._pipeline is not None

    def load_model(self) -> None:
        """Download (if needed) and load the MedASR model.

        This is a blocking call — run it on a worker thread.
        First-time use will download from HuggingFace (requires
        accepting the Health AI Developer Foundations terms).

        Raises:
            STTEngineError: If model loading fails.
        """
        if self._pipeline is not None:
            logger.info("MedASR model already loaded, skipping")
            return

        device = self._resolve_device()
        logger.info("Loading MedASR model on device: %s", device)

        try:
            import torch  # noqa: F401
            from transformers import pipeline

            self._pipeline = pipeline(
                "automatic-speech-recognition",
                model="google/medasr",
                device=device,
            )
            logger.info("MedASR model loaded successfully")
        except Exception as exc:
            raise STTEngineError(
                f"Failed to load MedASR model: {exc}"
            ) from exc

    def transcribe(self, audio: np.ndarray) -> str:
        """Transcribe a chunk of 16 kHz mono float32 audio.

        Args:
            audio: 1-D float32 numpy array at 16 kHz.

        Returns:
            Transcribed text, or empty string for silence/noise.

        Raises:
            STTEngineError: If the model is not loaded or transcription fails.
        """
        if self._pipeline is None:
            raise STTEngineError("MedASR model not loaded — call load_model() first")

        if len(audio) == 0:
            return ""

        try:
            result = self._pipeline(
                {"raw": audio, "sampling_rate": 16000},
                chunk_length_s=20,
                stride_length_s=2,
            )
            text = result.get("text", "").strip()
            return text
        except Exception as exc:
            raise STTEngineError(f"MedASR transcription failed: {exc}") from exc

    def unload_model(self) -> None:
        """Release the model from memory."""
        if self._pipeline is not None:
            self._pipeline = None
            logger.info("MedASR model unloaded")

    def _resolve_device(self) -> str:
        """Resolve the device string to use for inference."""
        if self._device_str == "auto":
            try:
                import torch
                return "cuda:0" if torch.cuda.is_available() else "cpu"
            except ImportError:
                return "cpu"
        return self._device_str
