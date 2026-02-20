"""
Embedded audio player widget for STT Transcriber.

Provides playback controls (play/pause, stop, seek, volume) for imported
audio files. Uses QMediaPlayer + QAudioOutput — Qt handles playback
internally, so no worker thread is needed.
"""

import logging
from pathlib import Path

from PySide6.QtCore import QUrl, Qt
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QWidget,
)

logger = logging.getLogger(__name__)


class AudioPlayerWidget(QWidget):
    """Horizontal audio playback bar for imported audio files.

    Hidden by default. Call :meth:`load_file` after a file import to
    make the widget visible and enable controls.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setVisible(False)

        self._seeking = False  # anti-jitter flag for seek slider

        # -- Qt multimedia objects --
        self._player = QMediaPlayer(self)
        self._audio_output = QAudioOutput(self)
        self._player.setAudioOutput(self._audio_output)
        self._audio_output.setVolume(0.5)

        # -- Controls --
        self.btn_play_pause = QPushButton("Play")
        self.btn_play_pause.setFixedWidth(60)
        self.btn_play_pause.setEnabled(False)

        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setFixedWidth(50)
        self.btn_stop.setEnabled(False)

        self.lbl_time = QLabel("00:00 / 00:00")
        self.lbl_time.setFixedWidth(110)

        self.seek_slider = QSlider(Qt.Orientation.Horizontal)
        self.seek_slider.setRange(0, 0)
        self.seek_slider.setEnabled(False)

        self.volume_slider = QSlider(Qt.Orientation.Horizontal)
        self.volume_slider.setRange(0, 100)
        self.volume_slider.setValue(50)
        self.volume_slider.setFixedWidth(80)

        self.speed_combo = QComboBox()
        self.speed_combo.addItems(["1x", "1.5x", "2x", "3x", "4x"])
        self.speed_combo.setFixedWidth(55)
        self.speed_combo.setToolTip("Playback speed")

        self.lbl_filename = QLabel("")
        self.lbl_filename.setStyleSheet("color: gray; font-style: italic;")

        # -- Layout --
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 2, 0, 2)
        layout.addWidget(self.btn_play_pause)
        layout.addWidget(self.btn_stop)
        layout.addWidget(self.lbl_time)
        layout.addWidget(self.seek_slider, stretch=1)
        layout.addWidget(self.speed_combo)
        layout.addWidget(QLabel("Vol:"))
        layout.addWidget(self.volume_slider)
        layout.addWidget(self.lbl_filename)

        # -- Signal wiring (buttons) --
        self.btn_play_pause.clicked.connect(self._on_play_pause)
        self.btn_stop.clicked.connect(self._on_stop)
        self.volume_slider.valueChanged.connect(self._on_volume_changed)
        self.speed_combo.currentTextChanged.connect(self._on_speed_changed)

        # -- Signal wiring (seek slider anti-jitter) --
        self.seek_slider.sliderPressed.connect(self._on_slider_pressed)
        self.seek_slider.sliderReleased.connect(self._on_slider_released)

        # -- Signal wiring (player) --
        self._player.positionChanged.connect(self._on_position_changed)
        self._player.durationChanged.connect(self._on_duration_changed)
        self._player.playbackStateChanged.connect(self._on_state_changed)
        self._player.errorOccurred.connect(self._on_error)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_file(self, file_path: str) -> None:
        """Load an audio file for playback, enable controls, and show widget."""
        self._player.stop()
        self._player.setSource(QUrl.fromLocalFile(file_path))
        self.lbl_filename.setText(Path(file_path).name)
        self.btn_play_pause.setEnabled(True)
        self.btn_stop.setEnabled(True)
        self.seek_slider.setEnabled(True)
        self.setVisible(True)
        logger.info("Audio player loaded: %s", file_path)

    def stop_playback(self) -> None:
        """Stop playback without hiding the widget."""
        self._player.stop()

    def clear(self) -> None:
        """Stop playback, reset state, and hide the widget."""
        self._player.stop()
        self._player.setSource(QUrl())
        self.seek_slider.setValue(0)
        self.seek_slider.setRange(0, 0)
        self.seek_slider.setEnabled(False)
        self.btn_play_pause.setText("Play")
        self.btn_play_pause.setEnabled(False)
        self.btn_stop.setEnabled(False)
        self.lbl_time.setText("00:00 / 00:00")
        self.speed_combo.setCurrentIndex(0)  # reset to 1x
        self.lbl_filename.setText("")
        self.setVisible(False)

    # ------------------------------------------------------------------
    # Button handlers
    # ------------------------------------------------------------------

    def _on_play_pause(self) -> None:
        state = self._player.playbackState()
        if state == QMediaPlayer.PlaybackState.PlayingState:
            self._player.pause()
        else:
            self._player.play()

    def _on_stop(self) -> None:
        self._player.stop()

    def _on_volume_changed(self, value: int) -> None:
        self._audio_output.setVolume(value / 100.0)

    def _on_speed_changed(self, text: str) -> None:
        rate = float(text.rstrip("x"))
        self._player.setPlaybackRate(rate)

    # ------------------------------------------------------------------
    # Seek slider anti-jitter
    # ------------------------------------------------------------------

    def _on_slider_pressed(self) -> None:
        self._seeking = True

    def _on_slider_released(self) -> None:
        self._seeking = False
        self._player.setPosition(self.seek_slider.value())

    # ------------------------------------------------------------------
    # Player signal handlers
    # ------------------------------------------------------------------

    def _on_position_changed(self, position_ms: int) -> None:
        if not self._seeking:
            self.seek_slider.setValue(position_ms)
        duration_ms = self._player.duration()
        self.lbl_time.setText(
            f"{self._format_time(position_ms)} / {self._format_time(duration_ms)}"
        )

    def _on_duration_changed(self, duration_ms: int) -> None:
        self.seek_slider.setRange(0, duration_ms)

    def _on_state_changed(self, state: QMediaPlayer.PlaybackState) -> None:
        if state == QMediaPlayer.PlaybackState.PlayingState:
            self.btn_play_pause.setText("Pause")
        else:
            self.btn_play_pause.setText("Play")

    def _on_error(self, error: QMediaPlayer.Error, message: str) -> None:
        logger.error("Audio player error (%s): %s", error, message)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _format_time(ms: int) -> str:
        """Format milliseconds as ``MM:SS`` or ``H:MM:SS``."""
        total_seconds = max(0, ms) // 1000
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        if hours:
            return f"{hours}:{minutes:02d}:{seconds:02d}"
        return f"{minutes:02d}:{seconds:02d}"
