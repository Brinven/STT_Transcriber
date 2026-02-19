"""
Audio capture module for STT Transcriber.

Provides mic input and WASAPI loopback recording via sounddevice,
with automatic resampling to 16 kHz mono for STT engines.
"""

from __future__ import annotations

import logging
import threading
from typing import Callable, Optional

import numpy as np
import sounddevice as sd

from backend.errors import AudioCaptureError

logger = logging.getLogger(__name__)

TARGET_SR = 16000
CHUNK_SECONDS = 5
CHUNK_SAMPLES = TARGET_SR * CHUNK_SECONDS  # 80,000 samples at 16 kHz


def resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample audio using linear interpolation.

    Args:
        audio: 1-D float32 array of audio samples.
        orig_sr: Original sample rate.
        target_sr: Target sample rate.

    Returns:
        Resampled 1-D float32 array.
    """
    if orig_sr == target_sr:
        return audio
    duration = len(audio) / orig_sr
    target_len = int(duration * target_sr)
    if target_len == 0:
        return np.array([], dtype=np.float32)
    orig_indices = np.linspace(0, len(audio) - 1, num=target_len)
    src_indices = np.arange(len(audio))
    return np.interp(orig_indices, src_indices, audio).astype(np.float32)


def list_input_devices() -> list[dict]:
    """Enumerate available microphone (input) devices.

    Returns:
        List of dicts with keys: index, name, channels, default_samplerate.
    """
    devices = sd.query_devices()
    results = []
    for i, dev in enumerate(devices):
        if dev["max_input_channels"] > 0 and dev.get("hostapi") is not None:
            results.append({
                "index": i,
                "name": dev["name"],
                "channels": dev["max_input_channels"],
                "default_samplerate": dev["default_samplerate"],
            })
    return results


def list_loopback_devices() -> list[dict]:
    """Enumerate output devices usable as WASAPI loopback sources.

    Returns:
        List of dicts with keys: index, name, channels, default_samplerate.
    """
    devices = sd.query_devices()
    hostapis = sd.query_hostapis()

    # Find the WASAPI host API index
    wasapi_index: Optional[int] = None
    for idx, api in enumerate(hostapis):
        if "wasapi" in api["name"].lower():
            wasapi_index = idx
            break

    if wasapi_index is None:
        logger.warning("WASAPI host API not found — loopback unavailable")
        return []

    results = []
    for i, dev in enumerate(devices):
        if dev["hostapi"] == wasapi_index and dev["max_output_channels"] > 0:
            results.append({
                "index": i,
                "name": dev["name"],
                "channels": dev["max_output_channels"],
                "default_samplerate": dev["default_samplerate"],
            })
    return results


class AudioCapture:
    """Captures audio from a mic or WASAPI loopback device.

    Audio is accumulated in an internal buffer. When 5 seconds of 16 kHz
    mono audio has been collected, the ``on_chunk`` callback is invoked
    with the chunk as a float32 numpy array.

    Args:
        on_chunk: Called with each 5-second chunk (float32, 16 kHz, mono).
        source: ``"microphone"`` or ``"loopback"``.
        device_index: Specific device index, or ``None`` for default.
    """

    def __init__(
        self,
        on_chunk: Callable[[np.ndarray], None],
        source: str = "microphone",
        device_index: Optional[int] = None,
    ) -> None:
        self._on_chunk = on_chunk
        self._source = source
        self._device_index = device_index

        self._stream: Optional[sd.InputStream] = None
        self._buffer: list[np.ndarray] = []
        self._buffer_samples = 0
        self._lock = threading.Lock()
        self._paused = threading.Event()  # clear = paused, set = running
        self._paused.set()  # start in "running" state
        self._running = False
        self._native_sr: int = TARGET_SR

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def is_paused(self) -> bool:
        return not self._paused.is_set()

    # ------------------------------------------------------------------
    # Control
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Open the audio stream and begin capturing."""
        if self._running:
            return

        try:
            if self._source == "loopback":
                self._start_loopback()
            else:
                self._start_microphone()
            self._running = True
            logger.info(
                "Audio capture started: source=%s, device=%s, native_sr=%d",
                self._source, self._device_index, self._native_sr,
            )
        except Exception as exc:
            raise AudioCaptureError(str(exc)) from exc

    def stop(self) -> None:
        """Stop the audio stream and flush the remaining buffer."""
        if not self._running:
            return
        self._running = False
        try:
            if self._stream is not None:
                self._stream.stop()
                self._stream.close()
                self._stream = None
        except Exception:
            logger.exception("Error closing audio stream")

        # Flush any remaining buffered audio as a final chunk
        self._flush_buffer()
        logger.info("Audio capture stopped")

    def pause(self) -> None:
        """Pause audio capture (stream stays open, data is discarded)."""
        self._paused.clear()
        logger.info("Audio capture paused")

    def resume(self) -> None:
        """Resume audio capture after pause."""
        self._paused.set()
        logger.info("Audio capture resumed")

    # ------------------------------------------------------------------
    # Stream setup helpers
    # ------------------------------------------------------------------

    def _start_microphone(self) -> None:
        """Open a mic input stream, attempting 16 kHz first."""
        device = self._device_index
        try:
            self._native_sr = TARGET_SR
            self._stream = sd.InputStream(
                samplerate=TARGET_SR,
                channels=1,
                dtype="float32",
                device=device,
                callback=self._audio_callback,
            )
            self._stream.start()
        except sd.PortAudioError:
            # Device may not support 16 kHz — fall back to native rate
            logger.info("Device rejected 16 kHz, falling back to native rate")
            if device is not None:
                info = sd.query_devices(device)
            else:
                info = sd.query_devices(kind="input")
            self._native_sr = int(info["default_samplerate"])
            self._stream = sd.InputStream(
                samplerate=self._native_sr,
                channels=1,
                dtype="float32",
                device=device,
                callback=self._audio_callback,
            )
            self._stream.start()

    def _start_loopback(self) -> None:
        """Open a WASAPI loopback stream on an output device."""
        device = self._device_index
        if device is None:
            loopback_devs = list_loopback_devices()
            if not loopback_devs:
                raise AudioCaptureError(
                    "No WASAPI loopback devices found. "
                    "System audio loopback requires Windows with WASAPI support."
                )
            device = loopback_devs[0]["index"]

        info = sd.query_devices(device)
        self._native_sr = int(info["default_samplerate"])
        channels = info["max_output_channels"]

        wasapi_settings = sd.WasapiSettings(exclusive=False)
        self._stream = sd.InputStream(
            samplerate=self._native_sr,
            channels=channels,
            dtype="float32",
            device=device,
            callback=self._audio_callback,
            extra_settings=wasapi_settings,
        )
        self._stream.start()

    # ------------------------------------------------------------------
    # Callback
    # ------------------------------------------------------------------

    def _audio_callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info: object,
        status: sd.CallbackFlags,
    ) -> None:
        """sounddevice callback — accumulates audio and fires on_chunk."""
        if status:
            logger.warning("Audio callback status: %s", status)

        if not self._paused.is_set():
            return  # Paused — discard audio

        # Convert to mono if multi-channel
        if indata.ndim > 1 and indata.shape[1] > 1:
            mono = indata.mean(axis=1).astype(np.float32)
        else:
            mono = indata[:, 0].copy() if indata.ndim > 1 else indata.copy().flatten()

        # Resample to 16 kHz if needed
        if self._native_sr != TARGET_SR:
            mono = resample(mono, self._native_sr, TARGET_SR)

        with self._lock:
            self._buffer.append(mono)
            self._buffer_samples += len(mono)

            while self._buffer_samples >= CHUNK_SAMPLES:
                self._emit_chunk()

    def _emit_chunk(self) -> None:
        """Extract one CHUNK_SAMPLES-sized chunk from the buffer and fire callback.

        Must be called while holding ``self._lock``.
        """
        combined = np.concatenate(self._buffer)
        chunk = combined[:CHUNK_SAMPLES]
        remainder = combined[CHUNK_SAMPLES:]

        self._buffer = [remainder] if len(remainder) > 0 else []
        self._buffer_samples = len(remainder)

        try:
            self._on_chunk(chunk)
        except Exception:
            logger.exception("Error in on_chunk callback")

    def _flush_buffer(self) -> None:
        """Send any remaining buffered audio as a final chunk."""
        with self._lock:
            if self._buffer_samples > 0:
                chunk = np.concatenate(self._buffer)
                self._buffer = []
                self._buffer_samples = 0
                try:
                    self._on_chunk(chunk)
                except Exception:
                    logger.exception("Error in on_chunk callback (flush)")
