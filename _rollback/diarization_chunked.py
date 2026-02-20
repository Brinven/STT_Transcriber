"""
Speaker diarization for STT Transcriber.

Uses Silero VAD for voice activity detection, WavLM (via HuggingFace
transformers) for speaker embeddings, and agglomerative clustering to
assign speaker labels to audio segments.  All processing is local and
offline (models are downloaded once on first use and cached).
"""

import logging
from dataclasses import dataclass

import numpy as np
import torch
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist

from backend.errors import DiarizationError

logger = logging.getLogger(__name__)

# Hex colours for up to 8 speakers in the transcript view.
SPEAKER_COLORS: list[str] = [
    "#2196F3",  # blue
    "#E91E63",  # pink
    "#4CAF50",  # green
    "#FF9800",  # orange
    "#9C27B0",  # purple
    "#00BCD4",  # cyan
    "#F44336",  # red
    "#8BC34A",  # light green
]

# Minimum speech segment duration (seconds) kept after VAD.
_MIN_SEGMENT_SEC: float = 0.5

# Maximum window size (seconds) for embedding extraction.  Long VAD
# segments are sliced into windows of this length so that each window
# is likely to contain a single speaker.
_MAX_WINDOW_SEC: float = 5.0

# Sample rate expected by Silero VAD and WavLM.
_MODEL_SR: int = 16_000

# HuggingFace model ID for speaker-verification embeddings.
_WAVLM_MODEL: str = "microsoft/wavlm-base-plus-sv"


@dataclass
class SpeakerSegment:
    """A contiguous time span attributed to a single speaker."""

    start_sec: float
    end_sec: float
    speaker_id: int


class DiarizationEngine:
    """Offline speaker-diarization pipeline.

    1. Silero VAD  → speech timestamps
    2. WavLM      → 512-dim speaker embeddings per segment
    3. Agglomerative hierarchical clustering → speaker IDs
    """

    def __init__(self) -> None:
        self._vad_model: object | None = None
        self._vad_utils: tuple | None = None
        self._feature_extractor: object | None = None
        self._encoder: object | None = None
        self._loaded = False

    # ------------------------------------------------------------------
    # Model lifecycle
    # ------------------------------------------------------------------

    def load_models(self) -> None:
        """Download (first run) and load VAD + speaker-embedding models."""
        try:
            self._load_vad()
            self._load_wavlm()
            self._loaded = True
            logger.info("Diarization models loaded")
        except DiarizationError:
            raise
        except Exception as exc:
            raise DiarizationError(f"Failed to load diarization models: {exc}") from exc

    def unload_models(self) -> None:
        """Release models from memory."""
        self._vad_model = None
        self._vad_utils = None
        self._feature_extractor = None
        self._encoder = None
        self._loaded = False
        logger.info("Diarization models unloaded")

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    # ------------------------------------------------------------------
    # Internal model loaders
    # ------------------------------------------------------------------

    def _load_vad(self) -> None:
        try:
            model, utils = torch.hub.load(
                "snakers4/silero-vad",
                "silero_vad",
                trust_repo=True,
            )
            self._vad_model = model
            self._vad_utils = utils
        except Exception as exc:
            raise DiarizationError(f"Silero VAD load failed: {exc}") from exc

    def _load_wavlm(self) -> None:
        """Load WavLM speaker-verification model from HuggingFace."""
        try:
            from transformers import AutoFeatureExtractor, WavLMForXVector

            self._feature_extractor = AutoFeatureExtractor.from_pretrained(
                _WAVLM_MODEL
            )
            model = WavLMForXVector.from_pretrained(_WAVLM_MODEL)
            model.eval()
            self._encoder = model
            logger.info("WavLM speaker-embedding model loaded from %s", _WAVLM_MODEL)
        except Exception as exc:
            raise DiarizationError(
                f"WavLM speaker-embedding model load failed: {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def diarize(
        self,
        audio: np.ndarray,
        num_speakers: int | None = None,
        threshold: float = 0.4,
    ) -> list[SpeakerSegment]:
        """Run the full diarization pipeline on 16 kHz mono float32 audio.

        Args:
            audio: 1-D float32 array at 16 kHz.
            num_speakers: If known, fix the cluster count.  Otherwise the
                threshold is used to cut the dendrogram.
            threshold: Cosine-distance threshold for ``fcluster`` when
                ``num_speakers`` is *None*.

        Returns:
            Sorted list of ``SpeakerSegment`` instances.

        Raises:
            DiarizationError: On any pipeline failure.
        """
        if not self._loaded:
            raise DiarizationError("Models not loaded — call load_models() first")

        try:
            speech_segments = self._run_vad(audio)
        except Exception as exc:
            raise DiarizationError(f"VAD failed: {exc}") from exc

        if not speech_segments:
            logger.info("No speech segments detected by VAD")
            return []

        # Split long VAD segments into ≤ _MAX_WINDOW_SEC windows so each
        # window is dominated by a single speaker.
        windows = self._split_into_windows(speech_segments)
        logger.info("Split %d VAD segments into %d windows",
                     len(speech_segments), len(windows))

        try:
            embeddings = self._extract_embeddings(audio, windows)
        except Exception as exc:
            raise DiarizationError(f"Embedding extraction failed: {exc}") from exc

        if len(embeddings) <= 1:
            # Only one window — trivially one speaker.
            return [
                SpeakerSegment(
                    start_sec=s / _MODEL_SR,
                    end_sec=e / _MODEL_SR,
                    speaker_id=0,
                )
                for s, e in windows
            ]

        try:
            labels = self._cluster(embeddings, num_speakers, threshold)
        except Exception as exc:
            raise DiarizationError(f"Clustering failed: {exc}") from exc

        # Build segments with speaker labels.
        raw: list[SpeakerSegment] = []
        for (start_sample, end_sample), label in zip(windows, labels):
            raw.append(
                SpeakerSegment(
                    start_sec=start_sample / _MODEL_SR,
                    end_sec=end_sample / _MODEL_SR,
                    speaker_id=int(label),
                )
            )

        smoothed = self._smooth_labels(raw)
        merged = self._merge_consecutive(smoothed)
        return merged

    # ------------------------------------------------------------------
    # Pipeline stages
    # ------------------------------------------------------------------

    @staticmethod
    def _split_into_windows(
        segments: list[tuple[int, int]],
    ) -> list[tuple[int, int]]:
        """Split VAD segments longer than *_MAX_WINDOW_SEC* into shorter windows."""
        max_samples = int(_MAX_WINDOW_SEC * _MODEL_SR)
        windows: list[tuple[int, int]] = []
        for start, end in segments:
            length = end - start
            if length <= max_samples:
                windows.append((start, end))
            else:
                pos = start
                while pos < end:
                    win_end = min(pos + max_samples, end)
                    # Only keep if long enough
                    if (win_end - pos) / _MODEL_SR >= _MIN_SEGMENT_SEC:
                        windows.append((pos, win_end))
                    pos = win_end
        return windows

    def _run_vad(self, audio: np.ndarray) -> list[tuple[int, int]]:
        """Return list of (start_sample, end_sample) for speech regions."""
        get_speech_timestamps = self._vad_utils[0]  # type: ignore[index]

        wav = torch.from_numpy(audio).float()
        timestamps = get_speech_timestamps(
            wav,
            self._vad_model,
            sampling_rate=_MODEL_SR,
            min_speech_duration_ms=int(_MIN_SEGMENT_SEC * 1000),
        )

        segments: list[tuple[int, int]] = []
        for ts in timestamps:
            start = ts["start"]
            end = ts["end"]
            if (end - start) / _MODEL_SR >= _MIN_SEGMENT_SEC:
                segments.append((start, end))

        logger.info("VAD found %d speech segments", len(segments))
        return segments

    def _extract_embeddings(
        self,
        audio: np.ndarray,
        segments: list[tuple[int, int]],
    ) -> np.ndarray:
        """Extract L2-normalised 512-dim WavLM speaker embeddings."""
        embs: list[np.ndarray] = []
        for start, end in segments:
            chunk = audio[start:end]
            inputs = self._feature_extractor(  # type: ignore[misc]
                chunk,
                sampling_rate=_MODEL_SR,
                return_tensors="pt",
                padding=True,
            )
            with torch.no_grad():
                output = self._encoder(**inputs)  # type: ignore[misc]
                emb = output.embeddings  # shape: [1, 512]
            embs.append(emb.squeeze(0).cpu().numpy())

        mat = np.vstack(embs)
        # L2-normalise so cosine distance works correctly.
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        mat = mat / norms
        return mat

    def _cluster(
        self,
        embeddings: np.ndarray,
        num_speakers: int | None,
        threshold: float,
    ) -> np.ndarray:
        """Agglomerative clustering on cosine distances."""
        dists = pdist(embeddings, metric="cosine")
        Z = linkage(dists, method="average")

        if num_speakers is not None:
            labels = fcluster(Z, t=num_speakers, criterion="maxclust")
        else:
            labels = fcluster(Z, t=threshold, criterion="distance")

        # Re-map to contiguous 0-indexed IDs (fcluster can skip numbers).
        unique_sorted = sorted(set(labels))
        remap = {old: new for new, old in enumerate(unique_sorted)}
        labels = np.array([remap[l] for l in labels])
        logger.info("Clustering produced %d speakers", len(unique_sorted))
        return labels

    @staticmethod
    def _smooth_labels(
        segments: list[SpeakerSegment],
        min_duration: float = 2.0,
    ) -> list[SpeakerSegment]:
        """Correct isolated speaker-label flips caused by embedding noise.

        If a segment is shorter than *min_duration* and both its neighbours
        belong to the same (different) speaker, reassign it to match.
        """
        if len(segments) < 3:
            return segments

        smoothed = list(segments)
        for i in range(1, len(smoothed) - 1):
            seg = smoothed[i]
            prev = smoothed[i - 1]
            nxt = smoothed[i + 1]
            duration = seg.end_sec - seg.start_sec

            if (
                duration < min_duration
                and prev.speaker_id == nxt.speaker_id
                and seg.speaker_id != prev.speaker_id
            ):
                smoothed[i] = SpeakerSegment(
                    start_sec=seg.start_sec,
                    end_sec=seg.end_sec,
                    speaker_id=prev.speaker_id,
                )

        return smoothed

    @staticmethod
    def _merge_consecutive(
        segments: list[SpeakerSegment],
        max_gap: float = 1.5,
    ) -> list[SpeakerSegment]:
        """Merge adjacent same-speaker segments only when the silence gap
        between them is ≤ *max_gap* seconds.  Gaps larger than that are
        treated as dead air and force a new line even for the same speaker.
        """
        if not segments:
            return []
        merged = [segments[0]]
        for seg in segments[1:]:
            prev = merged[-1]
            gap = seg.start_sec - prev.end_sec
            if seg.speaker_id == prev.speaker_id and gap <= max_gap:
                merged[-1] = SpeakerSegment(
                    start_sec=prev.start_sec,
                    end_sec=seg.end_sec,
                    speaker_id=prev.speaker_id,
                )
            else:
                merged.append(seg)
        return merged
