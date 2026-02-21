"""
QThread workers for STT Transcriber.

All long-running operations (model loading, audio capture, transcription,
file import) run on worker threads to keep the UI responsive.
"""

import logging
import queue
from typing import Optional

import numpy as np
import soundfile as sf
from PySide6.QtCore import QThread, Signal

from backend.audio_capture import AudioCapture, resample, TARGET_SR
from backend.errors import AudioCaptureError, DiarizationError, LLMConnectionError, STTEngineError
from backend.medical_formatter import format_soap
from backend.stt_engine import MedASREngine, STTEngine

logger = logging.getLogger(__name__)


class ModelLoadWorker(QThread):
    """Loads the STT model on a background thread.

    Signals:
        finished: Emitted with the loaded STTEngine on success.
        error: Emitted with an error message string on failure.
        progress: Emitted with status text during loading.
    """

    finished = Signal(object)
    error = Signal(str)
    progress = Signal(str)

    def __init__(self, model_size: str = "base", parent: Optional[object] = None) -> None:
        super().__init__(parent)
        self._model_size = model_size

    def run(self) -> None:
        try:
            self.progress.emit("Loading STT model (first run may download)...")
            engine = STTEngine(self._model_size)
            engine.load_model()
            self.finished.emit(engine)
        except (STTEngineError, ValueError) as exc:
            logger.exception("Model loading failed")
            self.error.emit(str(exc))
        except Exception as exc:
            logger.exception("Unexpected error loading model")
            self.error.emit(f"Unexpected error: {exc}")


class AudioWorker(QThread):
    """Captures audio and pushes 5-second chunks onto a shared queue.

    When stopped (via ``requestInterruption``), flushes the remaining
    buffer and puts a ``None`` sentinel on the queue to signal the
    TranscribeWorker to stop.

    Signals:
        started_recording: Emitted once the audio stream is open.
        error: Emitted with an error message string on failure.
    """

    started_recording = Signal()
    error = Signal(str)

    def __init__(
        self,
        chunk_queue: queue.Queue,
        source: str = "microphone",
        device_index: Optional[int] = None,
        parent: Optional[object] = None,
    ) -> None:
        super().__init__(parent)
        self._chunk_queue = chunk_queue
        self._source = source
        self._device_index = device_index
        self._capture: Optional[AudioCapture] = None

    def run(self) -> None:
        try:
            self._capture = AudioCapture(
                on_chunk=self._on_chunk,
                source=self._source,
                device_index=self._device_index,
            )
            self._capture.start()
            self.started_recording.emit()
        except AudioCaptureError as exc:
            logger.exception("Audio capture failed to start")
            self.error.emit(str(exc))
            return
        except Exception as exc:
            logger.exception("Unexpected error starting audio capture")
            self.error.emit(f"Audio capture error: {exc}")
            return

        # Stay alive until interruption is requested
        while not self.isInterruptionRequested():
            self.msleep(100)

        # Shut down capture and signal TranscribeWorker
        try:
            self._capture.stop()
        except Exception:
            logger.exception("Error stopping audio capture")
        self._chunk_queue.put(None)  # sentinel
        logger.info("AudioWorker finished")

    def _on_chunk(self, chunk: np.ndarray) -> None:
        """Called by AudioCapture when a 5-second chunk is ready."""
        self._chunk_queue.put(chunk)

    def pause(self) -> None:
        if self._capture is not None:
            self._capture.pause()

    def resume(self) -> None:
        if self._capture is not None:
            self._capture.resume()

    @property
    def is_paused(self) -> bool:
        if self._capture is not None:
            return self._capture.is_paused
        return False


class TranscribeWorker(QThread):
    """Pulls audio chunks from a queue and transcribes them.

    Runs until it receives a ``None`` sentinel on the queue.

    Signals:
        text_ready: Emitted with transcribed text for each chunk.
        error: Emitted with an error message for non-fatal transcription errors.
    """

    text_ready = Signal(str)
    error = Signal(str)

    def __init__(
        self,
        engine: STTEngine,
        chunk_queue: queue.Queue,
        parent: Optional[object] = None,
    ) -> None:
        super().__init__(parent)
        self._engine = engine
        self._chunk_queue = chunk_queue

    def run(self) -> None:
        logger.info("TranscribeWorker started")
        while True:
            try:
                chunk = self._chunk_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if chunk is None:
                logger.info("TranscribeWorker received sentinel, stopping")
                break

            try:
                text = self._engine.transcribe(chunk)
                if text:
                    self.text_ready.emit(text)
            except STTEngineError as exc:
                logger.warning("Transcription error (non-fatal): %s", exc)
                self.error.emit(str(exc))
            except Exception as exc:
                logger.warning("Unexpected transcription error: %s", exc)
                self.error.emit(f"Transcription error: {exc}")

        logger.info("TranscribeWorker finished")


class FileTranscribeWorker(QThread):
    """Reads an audio file (WAV, MP3, FLAC, OGG) and transcribes it in segments.

    Signals:
        text_ready: Emitted with transcribed text for each segment.
        progress: Emitted with progress percentage (0-100).
        error: Emitted with an error message on failure.
    """

    text_ready = Signal(str)
    progress = Signal(int)
    error = Signal(str)

    SEGMENT_SECONDS = 30

    def __init__(
        self,
        engine: STTEngine,
        file_path: str,
        parent: Optional[object] = None,
    ) -> None:
        super().__init__(parent)
        self._engine = engine
        self._file_path = file_path

    def run(self) -> None:
        logger.info("FileTranscribeWorker started: %s", self._file_path)
        try:
            audio, sr = sf.read(self._file_path, dtype="float32")
        except Exception as exc:
            logger.exception("Failed to read audio file")
            self.error.emit(
                f"Cannot read file: {exc}\n"
                "Supported formats: WAV, MP3, FLAC, OGG"
            )
            return

        # Convert to mono
        if audio.ndim > 1:
            audio = audio.mean(axis=1).astype(np.float32)

        # Resample to 16 kHz
        if sr != TARGET_SR:
            audio = resample(audio, sr, TARGET_SR)

        total_samples = len(audio)
        if total_samples == 0:
            self.error.emit("Audio file is empty")
            return

        segment_samples = self.SEGMENT_SECONDS * TARGET_SR
        processed = 0

        while processed < total_samples:
            if self.isInterruptionRequested():
                logger.info("FileTranscribeWorker interrupted")
                return

            end = min(processed + segment_samples, total_samples)
            segment = audio[processed:end]

            try:
                text = self._engine.transcribe(segment)
                if text:
                    self.text_ready.emit(text)
            except STTEngineError as exc:
                logger.warning("Segment transcription error: %s", exc)
                self.error.emit(str(exc))

            processed = end
            pct = int((processed / total_samples) * 100)
            self.progress.emit(pct)

        self.progress.emit(100)
        logger.info("FileTranscribeWorker finished")


class MedASRModelLoadWorker(QThread):
    """Loads the MedASR model on a background thread.

    Signals:
        finished: Emitted with the loaded MedASREngine on success.
        error: Emitted with an error message string on failure.
        progress: Emitted with status text during loading.
    """

    finished = Signal(object)
    error = Signal(str)
    progress = Signal(str)

    def __init__(self, device: str = "auto", hf_token: str = "", parent: Optional[object] = None) -> None:
        super().__init__(parent)
        self._device = device
        self._hf_token = hf_token

    def run(self) -> None:
        try:
            self.progress.emit(
                "Loading MedASR model (first run may download from HuggingFace)..."
            )
            engine = MedASREngine(self._device, hf_token=self._hf_token)
            engine.load_model()
            self.finished.emit(engine)
        except STTEngineError as exc:
            logger.exception("MedASR model loading failed")
            self.error.emit(str(exc))
        except Exception as exc:
            logger.exception("Unexpected error loading MedASR model")
            self.error.emit(f"Unexpected error: {exc}")


class DiarizeModelLoadWorker(QThread):
    """Loads pyannote diarization pipeline on a background thread.

    Signals:
        finished: Emitted with the loaded DiarizationEngine on success.
        error: Emitted with an error message string on failure.
        progress: Emitted with status text during loading.
    """

    finished = Signal(object)
    error = Signal(str)
    progress = Signal(str)

    def __init__(self, hf_token: str = "", parent: Optional[object] = None) -> None:
        super().__init__(parent)
        self._hf_token = hf_token

    def run(self) -> None:
        try:
            self.progress.emit(
                "Loading speaker diarization models (first run may download)..."
            )
            from backend.diarization import DiarizationEngine

            engine = DiarizationEngine(hf_token=self._hf_token)
            engine.load_models()
            self.finished.emit(engine)
        except DiarizationError as exc:
            logger.exception("Diarization model loading failed")
            self.error.emit(str(exc))
        except Exception as exc:
            logger.exception("Unexpected error loading diarization models")
            self.error.emit(f"Unexpected error: {exc}")


class DiarizeWorker(QThread):
    """Runs speaker diarization on an audio file.

    Signals:
        diarization_ready: Emitted with the list of SpeakerSegment on success.
        progress: Emitted with status text during processing.
        error: Emitted with an error message string on failure.
    """

    diarization_ready = Signal(list)
    progress = Signal(str)
    error = Signal(str)

    def __init__(
        self,
        engine: object,
        file_path: str,
        num_speakers: Optional[int] = None,
        parent: Optional[object] = None,
    ) -> None:
        super().__init__(parent)
        self._engine = engine
        self._file_path = file_path
        self._num_speakers = num_speakers

    def run(self) -> None:
        logger.info("DiarizeWorker started: %s", self._file_path)
        self.progress.emit("Identifying speakers...")
        try:
            audio, sr = sf.read(self._file_path, dtype="float32")
        except Exception as exc:
            logger.exception("Failed to read audio file for diarization")
            self.error.emit(f"Cannot read file for diarization: {exc}")
            return

        # Convert to mono
        if audio.ndim > 1:
            audio = audio.mean(axis=1).astype(np.float32)

        # Resample to 16 kHz
        if sr != TARGET_SR:
            audio = resample(audio, sr, TARGET_SR)

        try:
            timeline = self._engine.diarize(  # type: ignore[union-attr]
                audio,
                num_speakers=self._num_speakers,
            )
            self.diarization_ready.emit(timeline)
        except DiarizationError as exc:
            logger.exception("Diarization failed")
            self.error.emit(str(exc))
        except Exception as exc:
            logger.exception("Unexpected diarization error")
            self.error.emit(f"Diarization error: {exc}")


class SoapFormatWorker(QThread):
    """Sends transcript to a local LLM for SOAP formatting.

    Signals:
        soap_ready: Emitted with the parsed SOAP dict on success.
        error: Emitted with an error message string on failure.
    """

    soap_ready = Signal(dict)
    error = Signal(str)

    def __init__(
        self,
        transcript: str,
        endpoint: str,
        model: str,
        provider: str,
        parent: Optional[object] = None,
    ) -> None:
        super().__init__(parent)
        self._transcript = transcript
        self._endpoint = endpoint
        self._model = model
        self._provider = provider

    def run(self) -> None:
        try:
            logger.info("SoapFormatWorker: calling LLM (%s)", self._provider)
            result = format_soap(
                self._transcript,
                self._endpoint,
                self._model,
                self._provider,
            )
            self.soap_ready.emit(result)
        except LLMConnectionError as exc:
            logger.exception("SOAP formatting failed (LLM connection)")
            self.error.emit(str(exc))
        except Exception as exc:
            logger.exception("Unexpected error during SOAP formatting")
            self.error.emit(f"SOAP formatting error: {exc}")
