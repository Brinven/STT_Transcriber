"""
Speaker diarization for STT Transcriber.

Uses pyannote.audio's end-to-end speaker diarization pipeline for accurate
speaker segmentation and labeling.  All processing is local and offline
(models are downloaded once on first use and cached).

The pyannote pipeline handles VAD, speaker segmentation, embedding
extraction, and clustering internally — no manual feature engineering
required.

Requires a HuggingFace access token with accepted terms for:
- https://huggingface.co/pyannote/speaker-diarization-3.1
- https://huggingface.co/pyannote/segmentation-3.0
"""

import logging
from dataclasses import dataclass

import numpy as np
import torch

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

# Sample rate expected by pyannote pipeline.
_MODEL_SR: int = 16_000

# Pyannote model identifier (gated — requires HuggingFace token).
_PIPELINE_MODEL: str = "pyannote/speaker-diarization-3.1"


@dataclass
class SpeakerSegment:
    """A contiguous time span attributed to a single speaker."""

    start_sec: float
    end_sec: float
    speaker_id: int


class DiarizationEngine:
    """Offline speaker-diarization using pyannote.audio.

    Wraps pyannote's pretrained speaker-diarization pipeline which
    handles VAD, segmentation, embedding, and clustering end-to-end.

    Requires a HuggingFace access token with accepted terms for the
    gated pyannote models.
    """

    def __init__(self, hf_token: str = "") -> None:
        self._hf_token = hf_token
        self._pipeline: object | None = None
        self._loaded = False

    # ------------------------------------------------------------------
    # Model lifecycle
    # ------------------------------------------------------------------

    def load_models(self) -> None:
        """Download (first run) and load the pyannote diarization pipeline."""
        if not self._hf_token:
            raise DiarizationError(
                "HuggingFace token required for pyannote speaker diarization.\n"
                "1. Create a token at https://huggingface.co/settings/tokens\n"
                "2. Accept terms at https://huggingface.co/pyannote/speaker-diarization-3.1\n"
                "3. Accept terms at https://huggingface.co/pyannote/segmentation-3.0\n"
                "4. Enter your token in Settings > HuggingFace Token"
            )
        try:
            from pyannote.audio import Pipeline

            logger.info("Loading pyannote pipeline: %s", _PIPELINE_MODEL)
            pipeline = Pipeline.from_pretrained(
                _PIPELINE_MODEL,
                token=self._hf_token,
            )

            # Use GPU if available.
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            pipeline.to(device)
            logger.info("Pyannote pipeline loaded on %s", device)

            self._pipeline = pipeline
            self._loaded = True
        except DiarizationError:
            raise
        except Exception as exc:
            raise DiarizationError(
                f"Failed to load diarization pipeline: {exc}"
            ) from exc

    def unload_models(self) -> None:
        """Release the pipeline from memory."""
        self._pipeline = None
        self._loaded = False
        logger.info("Diarization pipeline unloaded")

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def diarize(
        self,
        audio: np.ndarray,
        num_speakers: int | None = None,
        threshold: float = 0.4,
    ) -> list[SpeakerSegment]:
        """Run speaker diarization on 16 kHz mono float32 audio.

        Args:
            audio: 1-D float32 array at 16 kHz.
            num_speakers: If known, fix the speaker count.
                Otherwise pyannote estimates it automatically.
            threshold: Unused (kept for API compat).

        Returns:
            Sorted list of :class:`SpeakerSegment` instances.
        """
        if not self._loaded or self._pipeline is None:
            raise DiarizationError(
                "Pipeline not loaded — call load_models() first"
            )

        # Build in-memory waveform dict to bypass torchcodec file I/O.
        waveform = torch.from_numpy(audio).unsqueeze(0).float()  # (1, T)
        audio_input = {"waveform": waveform, "sample_rate": _MODEL_SR}

        # Build kwargs for the pipeline.
        kwargs: dict = {}
        if num_speakers is not None:
            kwargs["num_speakers"] = num_speakers

        logger.info(
            "Running pyannote diarization (samples=%d, num_speakers=%s)",
            len(audio),
            num_speakers,
        )

        try:
            result = self._pipeline(audio_input, **kwargs)
        except Exception as exc:
            raise DiarizationError(
                f"Diarization pipeline failed: {exc}"
            ) from exc

        # pyannote 4.x returns DiarizeOutput; the Annotation is on
        # .speaker_diarization (or .exclusive_speaker_diarization for
        # non-overlapping turns).
        annotation = getattr(result, "speaker_diarization", result)

        # Convert pyannote Annotation to SpeakerSegment list.
        # Speaker labels are like "SPEAKER_00", "SPEAKER_01", etc.
        speaker_label_to_id: dict[str, int] = {}
        segments: list[SpeakerSegment] = []

        for turn, _, speaker_label in annotation.itertracks(yield_label=True):
            if speaker_label not in speaker_label_to_id:
                speaker_label_to_id[speaker_label] = len(speaker_label_to_id)

            segments.append(
                SpeakerSegment(
                    start_sec=turn.start,
                    end_sec=turn.end,
                    speaker_id=speaker_label_to_id[speaker_label],
                )
            )

        # Merge consecutive same-speaker segments with small gaps.
        merged = self._merge_consecutive(segments)

        # Log summary.
        unique_speakers = {s.speaker_id for s in merged}
        logger.info(
            "Diarization complete: %d speakers, %d segments",
            len(unique_speakers),
            len(merged),
        )
        for sid in sorted(unique_speakers):
            count = sum(1 for s in merged if s.speaker_id == sid)
            total_dur = sum(
                s.end_sec - s.start_sec for s in merged if s.speaker_id == sid
            )
            msg = (
                f"[DIARIZE] Speaker {sid}: {count} segments, "
                f"{total_dur:.1f}s total"
            )
            print(msg)
            logger.info(msg)

        return merged

    # ------------------------------------------------------------------
    # Post-processing
    # ------------------------------------------------------------------

    @staticmethod
    def _merge_consecutive(
        segments: list[SpeakerSegment],
        max_gap: float = 0.3,
    ) -> list[SpeakerSegment]:
        """Merge adjacent same-speaker segments when the gap is <= max_gap."""
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
