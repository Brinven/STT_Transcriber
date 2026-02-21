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

    Uses ``google/medasr`` (AutoModelForCTC + AutoProcessor) with manual
    CTC greedy decoding.  The transformers ASR pipeline has a broken CTC
    decoder for this model that produces stuttered/duplicated output, so
    we drive inference directly.

    Args:
        device: ``"auto"`` (CUDA if available, else CPU), ``"cpu"``,
            or ``"cuda"``.
        hf_token: HuggingFace token for gated model access.
    """

    _MODEL_NAME = "google/medasr"

    def __init__(self, device: str = "auto", hf_token: str = "") -> None:
        self._device_str = device
        self._hf_token = hf_token
        self._model = None
        self._processor = None
        self._device = None  # resolved torch.device

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load_model(self) -> None:
        """Download (if needed) and load the MedASR model.

        This is a blocking call — run it on a worker thread.
        First-time use will download from HuggingFace (requires
        accepting the Health AI Developer Foundations terms).

        Raises:
            STTEngineError: If model loading fails.
        """
        if self._model is not None:
            logger.info("MedASR model already loaded, skipping")
            return

        self._patch_lasr_feature_extractor()

        device_str = self._resolve_device()
        logger.info("Loading MedASR model on device: %s", device_str)

        try:
            import torch
            from transformers import AutoModelForCTC, AutoProcessor

            kwargs: dict = {}
            if self._hf_token:
                kwargs["token"] = self._hf_token

            self._processor = AutoProcessor.from_pretrained(
                self._MODEL_NAME, **kwargs,
            )
            self._model = AutoModelForCTC.from_pretrained(
                self._MODEL_NAME, **kwargs,
            )
            self._device = torch.device(device_str)
            self._model.to(self._device)
            self._model.eval()
            logger.info("MedASR model loaded successfully")
        except Exception as exc:
            self._model = None
            self._processor = None
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
        if self._model is None or self._processor is None:
            raise STTEngineError("MedASR model not loaded — call load_model() first")

        if len(audio) == 0:
            return ""

        try:
            import torch

            inputs = self._processor.feature_extractor(
                audio,
                sampling_rate=16000,
                return_tensors="pt",
                return_attention_mask=True,
            )
            inputs = {k: v.to(self._device) for k, v in inputs.items()}

            with torch.no_grad():
                logits = self._model(**inputs).logits

            # CTC greedy decode: argmax → collapse consecutive duplicates → remove blanks
            pred_ids = torch.argmax(logits, dim=-1)[0].cpu().tolist()
            text = self._ctc_decode(pred_ids)
            return text
        except Exception as exc:
            raise STTEngineError(f"MedASR transcription failed: {exc}") from exc

    def _ctc_decode(self, token_ids: list[int]) -> str:
        """Greedy CTC decoding: collapse consecutive duplicates, remove blanks."""
        # Step 1: collapse consecutive identical tokens
        collapsed: list[int] = []
        prev = None
        for idx in token_ids:
            if idx != prev:
                collapsed.append(idx)
            prev = idx

        # Step 2: remove blank token (epsilon = 0)
        collapsed = [idx for idx in collapsed if idx != 0]

        if not collapsed:
            return ""

        # Step 3: decode token IDs to text
        text = self._processor.tokenizer.decode(  # type: ignore[union-attr]
            collapsed, skip_special_tokens=True,
        )
        # Strip leading stray punctuation/brackets from first-frame artifacts
        return text.strip().lstrip("]}).,:;!? ")

    def unload_model(self) -> None:
        """Release the model from memory."""
        if self._model is not None:
            self._model = None
            self._processor = None
            self._device = None
            logger.info("MedASR model unloaded")

    @staticmethod
    def _patch_lasr_feature_extractor() -> None:
        """Patch a bug in transformers' LasrFeatureExtractor.

        The ``__call__`` method passes ``(waveform, device, center)`` to
        ``_torch_extract_fbank_features`` but the method signature only
        accepts ``(self, waveform, device)``.  We patch it to accept and
        ignore the extra ``center`` kwarg so transcription works.
        """
        try:
            from transformers.models.lasr.feature_extraction_lasr import (
                LasrFeatureExtractor,
            )
        except ImportError:
            return  # lasr not available in this transformers version

        import inspect

        sig = inspect.signature(LasrFeatureExtractor._torch_extract_fbank_features)
        if "center" in sig.parameters:
            return  # already patched or fixed upstream

        original = LasrFeatureExtractor._torch_extract_fbank_features

        def _patched(self, waveform, device="cpu", center=True):  # type: ignore[override]
            return original(self, waveform, device)

        LasrFeatureExtractor._torch_extract_fbank_features = _patched
        logger.info("Patched LasrFeatureExtractor._torch_extract_fbank_features")

    def _resolve_device(self) -> str:
        """Resolve the device string to use for inference."""
        if self._device_str == "auto":
            try:
                import torch
                return "cuda:0" if torch.cuda.is_available() else "cpu"
            except ImportError:
                return "cpu"
        return self._device_str
