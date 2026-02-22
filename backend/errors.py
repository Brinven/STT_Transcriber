"""
Centralized error handling for STT Transcriber.

Provides custom exception classes for the major subsystems.
"""


class STTEngineError(RuntimeError):
    """Raised when speech-to-text engine fails during loading or inference."""


class AudioCaptureError(RuntimeError):
    """Raised when audio capture fails (device not found, permission denied, etc.)."""


class LLMConnectionError(RuntimeError):
    """Raised when the local LLM server (Ollama/LM Studio) is unreachable or returns an error."""


class DiarizationError(RuntimeError):
    """Raised when speaker diarization fails during loading or inference."""


class VisionEngineError(RuntimeError):
    """Raised when medical image analysis fails during loading or inference."""
