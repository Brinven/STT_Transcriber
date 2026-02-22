"""
Main application window for STT Transcriber.
"""

import html
import logging
import queue
import time
from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QShortcut, QKeySequence
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMenu,
    QMenuBar,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
    QWidgetAction,
)

from backend.config_manager import ConfigManager
from backend.file_manager import (
    default_filename,
    export_csv,
    export_md,
    export_soap_txt,
    export_txt,
)
from backend.diarization import SPEAKER_COLORS, SpeakerSegment
from backend.paths import ensure_data_dirs
from backend.stt_engine import MedASREngine, STTEngine
from backend.vision_engine import VISION_PRESETS
from gui.widgets.audio_player import AudioPlayerWidget
from gui.widgets.image_analysis_view import ImageAnalysisView
from gui.widgets.soap_view import SoapView
from gui.workers import (
    AudioWorker,
    DiarizeModelLoadWorker,
    DiarizeWorker,
    FileTranscribeWorker,
    MedASRModelLoadWorker,
    ModelLoadWorker,
    SoapFormatWorker,
    TranscribeWorker,
    VisionAnalysisWorker,
    VisionModelLoadWorker,
)

logger = logging.getLogger(__name__)


class MainWindow(QMainWindow):
    """Primary application window."""

    def __init__(self) -> None:
        super().__init__()

        ensure_data_dirs()
        self.config = ConfigManager()

        self.setWindowTitle("STT Transcriber")
        self.resize(900, 650)

        # -- M2 state --
        self._stt_engine: Optional[STTEngine] = None
        self._chunk_queue: Optional[queue.Queue] = None

        self._model_load_worker: Optional[ModelLoadWorker] = None
        self._audio_worker: Optional[AudioWorker] = None
        self._transcribe_worker: Optional[TranscribeWorker] = None
        self._file_transcribe_worker: Optional[FileTranscribeWorker] = None

        self._is_recording = False
        self._pending_action: Optional[str] = None  # "record" or "import"
        self._pending_file_path: Optional[str] = None

        # -- M3 state --
        self._segments: list[dict] = []
        self._recording_start_time: Optional[float] = None
        self._medasr_engine: Optional[MedASREngine] = None
        self._medasr_load_worker: Optional[MedASRModelLoadWorker] = None
        self._soap_worker: Optional[SoapFormatWorker] = None

        # -- Vision state --
        self._vision_engine: object | None = None
        self._vision_load_worker: Optional[VisionModelLoadWorker] = None
        self._vision_analysis_worker: Optional[VisionAnalysisWorker] = None
        self._pending_vision_analysis: tuple[str, str] | None = None

        # -- Diarization state --
        self._diarize_engine: object | None = None
        self._diarize_model_load_worker: Optional[DiarizeModelLoadWorker] = None
        self._diarize_worker: Optional[DiarizeWorker] = None
        self._speaker_timeline: list[SpeakerSegment] = []
        self._speaker_names: dict[int, str] = {}
        self._current_file_path: Optional[str] = None

        self._build_menu_bar()
        self._build_central_widget()
        self._build_status_bar()

        # Connect button signals
        self.btn_record.clicked.connect(self._on_record_clicked)
        self.btn_pause.clicked.connect(self._on_pause_clicked)
        self.btn_stop.clicked.connect(self._on_stop_clicked)
        self.btn_import_wav.clicked.connect(self._on_import_wav_clicked)
        self.btn_export.clicked.connect(self._on_export_clicked)
        self.export_action.triggered.connect(self._on_export_clicked)
        self.btn_soapify.clicked.connect(self._on_soapify_clicked)
        self.btn_upload_image.clicked.connect(self._on_upload_image_clicked)
        self.image_analysis_view.analyze_requested.connect(
            self._on_analyze_image
        )
        self.btn_clear.clicked.connect(self._on_clear_all_clicked)

        # Enable Record and Import WAV buttons
        self.btn_record.setEnabled(True)
        self.btn_import_wav.setEnabled(True)

        # Keyboard shortcuts
        self._space_shortcut = QShortcut(QKeySequence(Qt.Key.Key_Space), self)
        self._space_shortcut.activated.connect(self._on_space_pressed)

        self._mode_shortcut = QShortcut(QKeySequence("Ctrl+M"), self)
        self._mode_shortcut.activated.connect(self._on_toggle_mode)

        # Apply saved mode on startup
        self.apply_mode(self.config.mode)

    # ------------------------------------------------------------------
    # Menu bar
    # ------------------------------------------------------------------

    def _build_menu_bar(self) -> None:
        menu_bar: QMenuBar = self.menuBar()

        # File menu
        file_menu = menu_bar.addMenu("&File")

        self.export_action = file_menu.addAction("&Export")
        self.export_action.setShortcut("Ctrl+E")
        self.export_action.setEnabled(False)

        file_menu.addSeparator()

        exit_action = file_menu.addAction("E&xit")
        exit_action.triggered.connect(self.close)

        # Settings menu
        settings_menu = menu_bar.addMenu("&Settings")

        font_size_action = QWidgetAction(self)
        font_widget = QWidget()
        font_layout = QHBoxLayout(font_widget)
        font_layout.setContentsMargins(8, 4, 8, 4)
        font_layout.addWidget(QLabel("Font Size:"))
        self.font_spin = QSpinBox()
        self.font_spin.setRange(8, 24)
        self.font_spin.setValue(self.config.font_size)
        self.font_spin.valueChanged.connect(self._on_font_size_changed)
        font_layout.addWidget(self.font_spin)
        font_size_action.setDefaultWidget(font_widget)
        settings_menu.addAction(font_size_action)

        hf_token_action = settings_menu.addAction("HuggingFace &Token...")
        hf_token_action.triggered.connect(self._on_hf_token_settings)

        llm_settings_action = settings_menu.addAction("&LLM Server...")
        llm_settings_action.triggered.connect(self._on_llm_settings)

        soap_layout_action = settings_menu.addAction("SOAP &Layout...")
        soap_layout_action.triggered.connect(self._on_soap_layout_settings)

        vision_device_action = settings_menu.addAction("&Vision Device...")
        vision_device_action.triggered.connect(self._on_vision_device_settings)

    # ------------------------------------------------------------------
    # Central widget
    # ------------------------------------------------------------------

    def _build_central_widget(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # -- Top bar: mode toggle + status label --
        top_bar = QHBoxLayout()

        self.btn_general = QPushButton("General")
        self.btn_general.setCheckable(True)
        self.btn_medical = QPushButton("Medical")
        self.btn_medical.setCheckable(True)

        self.mode_group = QButtonGroup(self)
        self.mode_group.setExclusive(True)
        self.mode_group.addButton(self.btn_general, 0)
        self.mode_group.addButton(self.btn_medical, 1)
        self.mode_group.idClicked.connect(self._on_mode_toggled)

        top_bar.addWidget(self.btn_general)
        top_bar.addWidget(self.btn_medical)
        top_bar.addStretch()

        self.status_label = QLabel("Ready")
        top_bar.addWidget(self.status_label)

        layout.addLayout(top_bar)

        # -- Audio controls bar --
        audio_bar = QHBoxLayout()

        self.source_combo = QComboBox()
        self.source_combo.addItems(["Microphone", "System Loopback"])
        if self.config.audio_source == "loopback":
            self.source_combo.setCurrentIndex(1)
        self.source_combo.currentIndexChanged.connect(self._on_source_changed)

        self.btn_record = QPushButton("Record")
        self.btn_record.setEnabled(False)

        self.btn_pause = QPushButton("Pause")
        self.btn_pause.setEnabled(False)

        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setEnabled(False)

        self.btn_import_wav = QPushButton("Import Audio")
        self.btn_import_wav.setEnabled(False)

        lbl_num_speakers = QLabel("Speakers:")
        self.spn_num_speakers = QSpinBox()
        self.spn_num_speakers.setRange(0, 8)
        self.spn_num_speakers.setValue(0)
        self.spn_num_speakers.setSpecialValueText("Auto")
        self.spn_num_speakers.setToolTip(
            "Set the number of speakers (0 = auto-detect).\n"
            "If you know how many people are talking, setting\n"
            "this improves speaker identification accuracy."
        )
        self.spn_num_speakers.setMinimumWidth(70)

        self.btn_diarize = QPushButton("Identify Speakers")
        self.btn_diarize.setEnabled(False)
        self.btn_diarize.setToolTip(
            "Run speaker identification on the imported audio.\n"
            "Set the speaker count first for best results."
        )
        self.btn_diarize.clicked.connect(self._on_diarize_clicked)

        audio_bar.addWidget(self.source_combo)
        audio_bar.addWidget(self.btn_record)
        audio_bar.addWidget(self.btn_pause)
        audio_bar.addWidget(self.btn_stop)
        audio_bar.addWidget(self.btn_import_wav)
        audio_bar.addStretch()  # push speaker controls to the right
        audio_bar.addWidget(lbl_num_speakers)
        audio_bar.addWidget(self.spn_num_speakers)
        audio_bar.addWidget(self.btn_diarize)

        layout.addLayout(audio_bar)

        # -- Transcript view --
        self.transcript_view = QTextEdit()
        self.transcript_view.setReadOnly(True)
        self.transcript_view.setPlaceholderText(
            "Transcription will appear here..."
        )
        self.transcript_view.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self.transcript_view.setContextMenuPolicy(
            Qt.ContextMenuPolicy.CustomContextMenu
        )
        self.transcript_view.customContextMenuRequested.connect(
            self._on_transcript_context_menu
        )
        layout.addWidget(self.transcript_view)

        # -- Audio player (hidden by default, shown after file import) --
        self.audio_player = AudioPlayerWidget()
        layout.addWidget(self.audio_player)

        # -- SOAP view (hidden by default, shown in medical mode) --
        self.soap_view = SoapView(layout_mode=self.config.soap_layout)
        self.soap_view.setVisible(False)
        layout.addWidget(self.soap_view)

        # -- Image analysis view (hidden by default, shown on Upload Image) --
        self.image_analysis_view = ImageAnalysisView()
        self.image_analysis_view.populate_presets(VISION_PRESETS)
        self.image_analysis_view.setVisible(False)
        layout.addWidget(self.image_analysis_view)

        # -- Bottom bar --
        bottom_bar = QHBoxLayout()

        self.btn_soapify = QPushButton("SOAPify")
        self.btn_soapify.setEnabled(False)
        self.btn_soapify.setVisible(False)

        self.btn_upload_image = QPushButton("Upload Image")
        self.btn_upload_image.setEnabled(True)
        self.btn_upload_image.setVisible(False)

        self.btn_clear = QPushButton("Clear All")
        self.btn_clear.setEnabled(False)

        self.btn_export = QPushButton("Export")
        self.btn_export.setEnabled(False)

        bottom_bar.addWidget(self.btn_soapify)
        bottom_bar.addWidget(self.btn_upload_image)
        bottom_bar.addStretch()
        bottom_bar.addWidget(self.btn_clear)
        bottom_bar.addWidget(self.btn_export)

        layout.addLayout(bottom_bar)

    # ------------------------------------------------------------------
    # Status bar
    # ------------------------------------------------------------------

    def _build_status_bar(self) -> None:
        self.mode_status = QLabel()
        self.statusBar().addPermanentWidget(self.mode_status)

    # ------------------------------------------------------------------
    # Mode handling
    # ------------------------------------------------------------------

    def apply_mode(self, mode: str) -> None:
        """Toggle UI state between 'general' and 'medical' mode."""
        is_medical = mode == "medical"

        self.btn_general.setChecked(not is_medical)
        self.btn_medical.setChecked(is_medical)
        self.btn_soapify.setVisible(is_medical)
        self.btn_upload_image.setVisible(is_medical)

        # Enable SOAPify only if medical mode and we have segments
        if is_medical:
            self.btn_soapify.setEnabled(bool(self._segments))
        else:
            # Switching away from medical — hide views, unload models
            self.soap_view.setVisible(False)
            self.image_analysis_view.setVisible(False)
            if self._medasr_engine is not None:
                self._medasr_engine.unload_model()
                self._medasr_engine = None
            if self._vision_engine is not None:
                self._vision_engine.unload_model()  # type: ignore[union-attr]
                self._vision_engine = None

        mode_text = "Medical" if is_medical else "General"
        self.mode_status.setText(f"Mode: {mode_text}")
        self.statusBar().showMessage(f"Switched to {mode_text} mode", 3000)

        self.config.mode = mode
        logger.info("Mode set to %s", mode)

    def _on_mode_toggled(self, button_id: int) -> None:
        self.apply_mode("medical" if button_id == 1 else "general")

    def _on_toggle_mode(self) -> None:
        """Ctrl+M: toggle between general and medical mode."""
        if self._is_recording:
            return  # Don't switch while recording
        new_mode = "general" if self.config.mode == "medical" else "medical"
        self.apply_mode(new_mode)

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _ensure_model_loaded(self, then_action: str, **kwargs: object) -> bool:
        """Start model loading if needed. Returns True if already loaded.

        Routes to MedASR or Whisper depending on the current mode.

        Args:
            then_action: ``"record"`` or ``"import"`` — action to take
                after the model finishes loading.
            **kwargs: Extra state to save (e.g. file_path for import).
        """
        if self.config.mode == "medical":
            return self._ensure_medasr_loaded(then_action, **kwargs)

        if self._stt_engine is not None and self._stt_engine.is_loaded:
            return True

        if self._model_load_worker is not None:
            return False  # Already loading

        self._pending_action = then_action
        self._pending_file_path = kwargs.get("file_path")  # type: ignore[assignment]
        self._set_controls_busy(True)

        self._model_load_worker = ModelLoadWorker(
            model_size=self.config.whisper_model_size,
            parent=self,
        )
        self._model_load_worker.progress.connect(self._on_model_progress)
        self._model_load_worker.finished.connect(self._on_model_loaded)
        self._model_load_worker.error.connect(self._on_model_error)
        self._model_load_worker.finished.connect(self._model_load_worker.deleteLater)
        self._model_load_worker.error.connect(self._model_load_worker.deleteLater)
        self._model_load_worker.start()
        return False

    def _ensure_medasr_loaded(self, then_action: str, **kwargs: object) -> bool:
        """Start MedASR model loading if needed. Returns True if already loaded."""
        if self._medasr_engine is not None and self._medasr_engine.is_loaded:
            return True

        if self._medasr_load_worker is not None:
            return False  # Already loading

        self._pending_action = then_action
        self._pending_file_path = kwargs.get("file_path")  # type: ignore[assignment]
        self._set_controls_busy(True)

        self._medasr_load_worker = MedASRModelLoadWorker(
            device=self.config.medasr_device,
            hf_token=self.config.hf_token,
            parent=self,
        )
        self._medasr_load_worker.progress.connect(self._on_model_progress)
        self._medasr_load_worker.finished.connect(self._on_medasr_loaded)
        self._medasr_load_worker.error.connect(self._on_medasr_error)
        self._medasr_load_worker.finished.connect(
            self._medasr_load_worker.deleteLater
        )
        self._medasr_load_worker.error.connect(
            self._medasr_load_worker.deleteLater
        )
        self._medasr_load_worker.start()
        return False

    def _on_model_progress(self, text: str) -> None:
        self.status_label.setText(text)

    def _on_model_loaded(self, engine: object) -> None:
        self._stt_engine = engine  # type: ignore[assignment]
        self._model_load_worker = None
        self._set_controls_busy(False)
        self.status_label.setText("Ready")
        logger.info("STT engine loaded and ready")

        self._execute_pending_action()

    def _on_model_error(self, message: str) -> None:
        self._model_load_worker = None
        self._pending_action = None
        self._pending_file_path = None
        self._set_controls_busy(False)
        self.status_label.setText("Ready")
        QMessageBox.critical(
            self,
            "Model Load Error",
            f"Failed to load STT model:\n\n{message}",
        )

    def _on_medasr_loaded(self, engine: object) -> None:
        self._medasr_engine = engine  # type: ignore[assignment]
        self._medasr_load_worker = None
        self._set_controls_busy(False)
        self.status_label.setText("Ready")
        logger.info("MedASR engine loaded and ready")

        self._execute_pending_action()

    def _on_medasr_error(self, message: str) -> None:
        self._medasr_load_worker = None
        self._pending_action = None
        self._pending_file_path = None
        self._set_controls_busy(False)
        self.status_label.setText("Ready")

        hint = ""
        if "401" in message or "access" in message.lower():
            hint = (
                "\n\nYou may need to accept the model terms at:\n"
                "https://huggingface.co/google/medasr"
            )
        QMessageBox.critical(
            self,
            "MedASR Load Error",
            f"Failed to load MedASR model:\n\n{message}{hint}",
        )

    def _execute_pending_action(self) -> None:
        """Execute the deferred action after a model finishes loading."""
        action = self._pending_action
        self._pending_action = None
        if action == "record":
            self._start_recording()
        elif action == "import":
            file_path = self._pending_file_path
            self._pending_file_path = None
            if file_path:
                self._start_file_transcription(file_path)

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def _on_record_clicked(self) -> None:
        if self._is_recording:
            return
        if not self._ensure_model_loaded("record"):
            return  # Will continue in _on_model_loaded
        self._start_recording()

    def _start_recording(self) -> None:
        self.audio_player.stop_playback()
        source = self.config.audio_source
        device_index = self.config.audio_device_index

        # Pick the correct engine based on mode
        if self.config.mode == "medical":
            engine = self._medasr_engine
        else:
            engine = self._stt_engine

        if engine is None or not engine.is_loaded:
            logger.error("Cannot start recording — engine not loaded")
            return

        self._recording_start_time = time.monotonic()
        self._chunk_queue = queue.Queue()

        self._audio_worker = AudioWorker(
            chunk_queue=self._chunk_queue,
            source=source,
            device_index=device_index,
            parent=self,
        )
        self._audio_worker.started_recording.connect(self._on_recording_started)
        self._audio_worker.error.connect(self._on_audio_error)
        self._audio_worker.finished.connect(self._on_audio_worker_finished)

        self._transcribe_worker = TranscribeWorker(
            engine=engine,
            chunk_queue=self._chunk_queue,
            parent=self,
        )
        self._transcribe_worker.text_ready.connect(self._on_text_ready)
        self._transcribe_worker.error.connect(self._on_transcribe_error)
        self._transcribe_worker.finished.connect(self._on_transcribe_worker_finished)

        self._is_recording = True
        self._set_controls_recording(True)
        self.status_label.setText("Starting recording...")

        self._audio_worker.start()
        self._transcribe_worker.start()

    def _on_recording_started(self) -> None:
        self.status_label.setText("Recording...")
        self.statusBar().showMessage("Recording started", 2000)

    def _on_audio_error(self, message: str) -> None:
        self._is_recording = False
        self._set_controls_recording(False)
        self.status_label.setText("Ready")
        QMessageBox.critical(
            self,
            "Audio Capture Error",
            f"Failed to start audio capture:\n\n{message}",
        )
        self._cleanup_recording_workers()

    def _on_text_ready(self, text: str) -> None:
        """Append transcribed text to the transcript view and segment list."""
        # Compute timestamp
        if self._recording_start_time is not None:
            elapsed = time.monotonic() - self._recording_start_time
        else:
            elapsed = 0.0
        hours, remainder = divmod(int(elapsed), 3600)
        minutes, seconds = divmod(remainder, 60)
        timestamp = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

        # Store structured segment
        self._segments.append({"timestamp": timestamp, "text": text, "speaker": None})

        # Display in transcript view (quote=False: avoid &#x27; Qt rendering bug)
        safe_text = html.escape(text, quote=False)
        safe_ts = html.escape(timestamp, quote=False)
        self.transcript_view.append(f"[{safe_ts}] {safe_text}")

        # Auto-scroll to bottom
        scrollbar = self.transcript_view.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

        # Enable export/clear if we have content
        has_text = bool(self._segments)
        self.btn_export.setEnabled(has_text)
        self.btn_clear.setEnabled(has_text)
        self.export_action.setEnabled(has_text)

        # Enable SOAPify in medical mode if we have segments
        if self.config.mode == "medical":
            self.btn_soapify.setEnabled(has_text)

    def _on_transcribe_error(self, message: str) -> None:
        self.statusBar().showMessage(f"Transcription warning: {message}", 5000)

    # ------------------------------------------------------------------
    # Pause / Stop
    # ------------------------------------------------------------------

    def _on_pause_clicked(self) -> None:
        if not self._is_recording or self._audio_worker is None:
            return

        if self._audio_worker.is_paused:
            self._audio_worker.resume()
            self.btn_pause.setText("Pause")
            self.status_label.setText("Recording...")
        else:
            self._audio_worker.pause()
            self.btn_pause.setText("Resume")
            self.status_label.setText("Paused")

    def _on_stop_clicked(self) -> None:
        if not self._is_recording:
            return
        self.status_label.setText("Stopping...")
        if self._audio_worker is not None:
            self._audio_worker.requestInterruption()

    def _on_audio_worker_finished(self) -> None:
        if self._audio_worker is not None:
            self._audio_worker.deleteLater()
            self._audio_worker = None

    def _on_transcribe_worker_finished(self) -> None:
        if self._transcribe_worker is not None:
            self._transcribe_worker.deleteLater()
            self._transcribe_worker = None

        self._is_recording = False
        self._set_controls_recording(False)
        self.btn_pause.setText("Pause")
        self.status_label.setText("Ready")
        self.statusBar().showMessage("Recording stopped", 2000)
        logger.info("Recording session ended")

    def _cleanup_recording_workers(self) -> None:
        """Force cleanup of recording workers (e.g. after audio error)."""
        if self._audio_worker is not None:
            self._audio_worker.requestInterruption()
            self._audio_worker.wait(2000)
            self._audio_worker.deleteLater()
            self._audio_worker = None
        if self._transcribe_worker is not None:
            # Put sentinel so transcribe worker exits
            if self._chunk_queue is not None:
                self._chunk_queue.put(None)
            self._transcribe_worker.wait(2000)
            self._transcribe_worker.deleteLater()
            self._transcribe_worker = None

    # ------------------------------------------------------------------
    # WAV file import
    # ------------------------------------------------------------------

    def _on_import_wav_clicked(self) -> None:
        if self._is_recording or self._file_transcribe_worker is not None:
            return

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Import Audio File",
            self.config.last_import_dir,
            "Audio Files (*.wav *.mp3);;WAV Files (*.wav);;MP3 Files (*.mp3);;All Files (*)",
        )
        if not file_path:
            return

        self.config.last_import_dir = str(Path(file_path).parent)

        if not self._ensure_model_loaded("import", file_path=file_path):
            return  # Will continue in _on_model_loaded
        self._start_file_transcription(file_path)

    def _start_file_transcription(self, file_path: str) -> None:
        self._current_file_path = file_path
        self.audio_player.load_file(file_path)
        self._set_controls_busy(True)
        self.status_label.setText("Transcribing file...")
        self._recording_start_time = time.monotonic()

        # Pick the correct engine based on mode
        engine = (
            self._medasr_engine
            if self.config.mode == "medical"
            else self._stt_engine
        )

        self._file_transcribe_worker = FileTranscribeWorker(
            engine=engine,
            file_path=file_path,
            parent=self,
        )
        self._file_transcribe_worker.text_ready.connect(self._on_text_ready)
        self._file_transcribe_worker.progress.connect(self._on_file_progress)
        self._file_transcribe_worker.error.connect(self._on_file_error)
        self._file_transcribe_worker.finished.connect(
            self._on_file_transcribe_finished
        )
        self._file_transcribe_worker.start()

    def _on_file_progress(self, pct: int) -> None:
        self.status_label.setText(f"Transcribing file... {pct}%")

    def _on_file_error(self, message: str) -> None:
        QMessageBox.critical(
            self,
            "File Transcription Error",
            f"Error transcribing file:\n\n{message}",
        )

    def _on_file_transcribe_finished(self) -> None:
        if self._file_transcribe_worker is not None:
            self._file_transcribe_worker.deleteLater()
            self._file_transcribe_worker = None

        self._set_controls_busy(False)
        self.status_label.setText("Ready")
        self.statusBar().showMessage("File transcription complete", 3000)

        # Enable diarize button if we have a transcribed file.
        if self._current_file_path and self._segments:
            self.btn_diarize.setEnabled(True)

    # ------------------------------------------------------------------
    # UI state helpers
    # ------------------------------------------------------------------

    def _set_controls_recording(self, recording: bool) -> None:
        """Enable/disable controls for recording state."""
        self.btn_record.setEnabled(not recording)
        self.btn_pause.setEnabled(recording)
        self.btn_stop.setEnabled(recording)
        self.btn_import_wav.setEnabled(not recording)
        self.source_combo.setEnabled(not recording)
        self.btn_general.setEnabled(not recording)
        self.btn_medical.setEnabled(not recording)

    def _set_controls_busy(self, busy: bool) -> None:
        """Enable/disable controls during model loading or file transcription."""
        self.btn_record.setEnabled(not busy)
        self.btn_import_wav.setEnabled(not busy)
        self.source_combo.setEnabled(not busy)
        self.btn_general.setEnabled(not busy)
        self.btn_medical.setEnabled(not busy)

    # ------------------------------------------------------------------
    # Keyboard shortcut
    # ------------------------------------------------------------------

    def _on_space_pressed(self) -> None:
        """Space bar toggles Record/Stop unless transcript_view has focus."""
        if self.transcript_view.hasFocus():
            return
        if self._is_recording:
            self._on_stop_clicked()
        else:
            self._on_record_clicked()

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def _on_export_clicked(self) -> None:
        """Export transcript or SOAP notes to a file."""
        # Build filter string — include SOAP option if soap_view has content
        filters = "Text Files (*.txt);;CSV Files (*.csv);;Markdown Files (*.md)"
        if self.soap_view.isVisible():
            filters = "SOAP Text (*.txt);;" + filters

        initial_dir = self.config.export_directory or ""
        suggested = default_filename("Transcript", "txt")

        file_path, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Export Transcript",
            f"{initial_dir}/{suggested}" if initial_dir else suggested,
            filters,
        )
        if not file_path:
            return

        try:
            spk_names = self._speaker_names or None
            if selected_filter == "SOAP Text (*.txt)":
                soap_dict = self.soap_view.get_soap_dict()
                export_soap_txt(soap_dict, file_path)
            elif file_path.endswith(".csv"):
                export_csv(self._segments, file_path, speaker_names=spk_names)
            elif file_path.endswith(".md"):
                export_md(self._segments, file_path, speaker_names=spk_names)
            else:
                export_txt(self._segments, file_path, speaker_names=spk_names)

            self.statusBar().showMessage(f"Exported to {file_path}", 5000)
            logger.info("Exported to %s", file_path)
        except OSError as exc:
            QMessageBox.critical(
                self,
                "Export Error",
                f"Failed to export file:\n\n{exc}",
            )

    # ------------------------------------------------------------------
    # Clear all
    # ------------------------------------------------------------------

    def _on_clear_all_clicked(self) -> None:
        """Clear transcript, segments, SOAP view, and diarization state."""
        self._segments.clear()
        self._recording_start_time = None
        self._speaker_timeline.clear()
        self._speaker_names.clear()
        self._current_file_path = None
        self.transcript_view.clear()
        self.audio_player.clear()
        self.soap_view.clear()
        self.soap_view.setVisible(False)
        self.image_analysis_view.clear()
        self.image_analysis_view.setVisible(False)
        self.btn_export.setEnabled(False)
        self.btn_clear.setEnabled(False)
        self.btn_soapify.setEnabled(False)
        self.export_action.setEnabled(False)
        self.statusBar().showMessage("Cleared", 2000)

    # ------------------------------------------------------------------
    # SOAP formatting
    # ------------------------------------------------------------------

    def _on_soapify_clicked(self) -> None:
        """Send the current transcript to the LLM for SOAP formatting."""
        if self._soap_worker is not None:
            return  # Already running

        if not self._segments:
            return

        transcript = "\n".join(seg["text"] for seg in self._segments)
        self.status_label.setText("Formatting SOAP notes...")
        self.btn_soapify.setEnabled(False)

        self._soap_worker = SoapFormatWorker(
            transcript=transcript,
            endpoint=self.config.llm_endpoint,
            model=self.config.llm_model,
            provider=self.config.llm_provider,
            parent=self,
        )
        self._soap_worker.soap_ready.connect(self._on_soap_ready)
        self._soap_worker.error.connect(self._on_soap_error)
        self._soap_worker.finished.connect(self._on_soap_worker_finished)
        self._soap_worker.start()

    def _on_soap_ready(self, soap: dict) -> None:
        """Handle successful SOAP formatting."""
        self.soap_view.set_soap(soap)
        self.soap_view.setVisible(True)
        self.status_label.setText("SOAP notes ready")
        self.statusBar().showMessage("SOAP formatting complete", 3000)
        logger.info("SOAP notes generated successfully")

    def _on_soap_error(self, message: str) -> None:
        """Handle SOAP formatting failure."""
        self.status_label.setText("Ready")
        QMessageBox.critical(
            self,
            "SOAP Formatting Error",
            f"Failed to format SOAP notes:\n\n{message}\n\n"
            "Make sure your LLM server (Ollama or LM Studio) is running "
            "and the endpoint is configured correctly in Settings.",
        )

    def _on_soap_worker_finished(self) -> None:
        if self._soap_worker is not None:
            self._soap_worker.deleteLater()
            self._soap_worker = None
        self.btn_soapify.setEnabled(bool(self._segments))

    # ------------------------------------------------------------------
    # Vision: image upload + analysis
    # ------------------------------------------------------------------

    def _on_upload_image_clicked(self) -> None:
        """Open file dialog for medical image upload."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Upload Medical Image",
            self.config.last_import_dir,
            "Image Files (*.png *.jpg *.jpeg *.bmp *.tiff *.tif *.webp);;All Files (*)",
        )
        if not file_path:
            return

        self.config.last_import_dir = str(Path(file_path).parent)

        if self.image_analysis_view.load_image(file_path):
            self.image_analysis_view.setVisible(True)

    def _on_analyze_image(self, image_path: str, query: str) -> None:
        """Handle analyze_requested signal from ImageAnalysisView."""
        if self._vision_analysis_worker is not None:
            return  # Already running

        if not self._ensure_vision_loaded(image_path, query):
            return  # Will continue after model loads

        self._run_vision_analysis(image_path, query)

    def _ensure_vision_loaded(self, image_path: str, query: str) -> bool:
        """Start vision model loading if needed. Returns True if already loaded."""
        if self._vision_engine is not None and self._vision_engine.is_loaded:  # type: ignore[union-attr]
            return True

        if self._vision_load_worker is not None:
            return False  # Already loading

        # Save pending analysis to run after model loads
        self._pending_vision_analysis = (image_path, query)
        self.image_analysis_view.set_busy(True)

        self._vision_load_worker = VisionModelLoadWorker(
            device=self.config.vision_device,
            hf_token=self.config.hf_token,
            parent=self,
        )
        self._vision_load_worker.progress.connect(self._on_model_progress)
        self._vision_load_worker.finished.connect(self._on_vision_loaded)
        self._vision_load_worker.error.connect(self._on_vision_load_error)
        self._vision_load_worker.finished.connect(
            self._vision_load_worker.deleteLater
        )
        self._vision_load_worker.error.connect(
            self._vision_load_worker.deleteLater
        )
        self._vision_load_worker.start()
        return False

    def _on_vision_loaded(self, engine: object) -> None:
        """Vision model loaded — save engine and run pending analysis."""
        self._vision_engine = engine
        self._vision_load_worker = None
        self.status_label.setText("Ready")
        logger.info("MedGemma vision engine loaded and ready")

        # Run the pending analysis if any
        pending = self._pending_vision_analysis
        self._pending_vision_analysis = None
        if pending:
            self._run_vision_analysis(*pending)
        else:
            self.image_analysis_view.set_busy(False)

    def _on_vision_load_error(self, message: str) -> None:
        """Vision model load failed."""
        self._vision_load_worker = None
        self._pending_vision_analysis = None
        self.image_analysis_view.set_busy(False)
        self.status_label.setText("Ready")

        hint = ""
        if "401" in message or "access" in message.lower():
            hint = (
                "\n\nYou may need to accept the model terms at:\n"
                "https://huggingface.co/google/medgemma-1.5-4b-it\n\n"
                "Set your HuggingFace token in Settings > HuggingFace Token."
            )
        QMessageBox.critical(
            self,
            "Vision Model Load Error",
            f"Failed to load MedGemma vision model:\n\n{message}{hint}",
        )

    def _run_vision_analysis(self, image_path: str, query: str) -> None:
        """Create and start the VisionAnalysisWorker."""
        self.image_analysis_view.set_busy(True)
        self.status_label.setText("Analyzing image...")

        self._vision_analysis_worker = VisionAnalysisWorker(
            engine=self._vision_engine,
            image_path=image_path,
            query=query,
            parent=self,
        )
        self._vision_analysis_worker.analysis_ready.connect(
            self._on_analysis_ready
        )
        self._vision_analysis_worker.error.connect(self._on_analysis_error)
        self._vision_analysis_worker.progress.connect(self._on_model_progress)
        self._vision_analysis_worker.finished.connect(
            self._on_vision_analysis_finished
        )
        self._vision_analysis_worker.start()

    def _on_analysis_ready(self, text: str) -> None:
        """Handle successful image analysis."""
        self.image_analysis_view.set_results(text)
        self.status_label.setText("Analysis complete")
        self.statusBar().showMessage("Image analysis complete", 3000)
        logger.info("Image analysis completed successfully")

    def _on_analysis_error(self, message: str) -> None:
        """Handle image analysis failure."""
        self.status_label.setText("Ready")
        QMessageBox.critical(
            self,
            "Image Analysis Error",
            f"Failed to analyze image:\n\n{message}",
        )

    def _on_vision_analysis_finished(self) -> None:
        """Clean up vision analysis worker."""
        if self._vision_analysis_worker is not None:
            self._vision_analysis_worker.deleteLater()
            self._vision_analysis_worker = None
        self.image_analysis_view.set_busy(False)
        self.status_label.setText("Ready")

    # ------------------------------------------------------------------
    # Vision device settings
    # ------------------------------------------------------------------

    def _on_vision_device_settings(self) -> None:
        """Let user pick the device for MedGemma vision model."""
        options = ["auto", "cuda", "cpu"]
        current = self.config.vision_device
        current_idx = options.index(current) if current in options else 0

        choice, ok = QInputDialog.getItem(
            self,
            "Vision Device",
            "Select device for MedGemma vision model:",
            options,
            current_idx,
            False,
        )
        if ok and choice != current:
            self.config.vision_device = choice
            # Unload vision engine so it reloads on new device
            if self._vision_engine is not None:
                self._vision_engine.unload_model()  # type: ignore[union-attr]
                self._vision_engine = None
            self.statusBar().showMessage(f"Vision device set to: {choice}", 3000)

    # ------------------------------------------------------------------
    # Diarization
    # ------------------------------------------------------------------

    def _on_diarize_clicked(self) -> None:
        """User clicked 'Identify Speakers' — start diarization."""
        if not self._current_file_path or not self._segments:
            return
        self.btn_diarize.setEnabled(False)
        self._start_diarization(self._current_file_path)

    def _start_diarization(self, file_path: str) -> None:
        """Kick off speaker diarization: load models if needed, then run."""
        # Ensure HF token is available (required for pyannote gated models).
        if not self.config.hf_token:
            token = self._prompt_hf_token()
            if not token:
                self.btn_diarize.setEnabled(True)
                self.status_label.setText("Ready")
                return
            self.config.hf_token = token

        self.status_label.setText("Preparing speaker identification...")

        if self._diarize_engine is not None:
            # Models already loaded — go straight to worker.
            self._launch_diarize_worker(file_path)
            return

        if self._diarize_model_load_worker is not None:
            return  # Already loading

        self._diarize_model_load_worker = DiarizeModelLoadWorker(
            hf_token=self.config.hf_token, parent=self,
        )
        self._diarize_model_load_worker.progress.connect(self._on_model_progress)
        self._diarize_model_load_worker.finished.connect(
            lambda engine: self._on_diarize_models_loaded(engine, file_path)
        )
        self._diarize_model_load_worker.error.connect(self._on_diarize_model_error)
        self._diarize_model_load_worker.finished.connect(
            self._diarize_model_load_worker.deleteLater
        )
        self._diarize_model_load_worker.error.connect(
            self._diarize_model_load_worker.deleteLater
        )
        self._diarize_model_load_worker.start()

    def _on_diarize_models_loaded(self, engine: object, file_path: str) -> None:
        """Diarization models ready — save engine and launch worker."""
        self._diarize_engine = engine
        self._diarize_model_load_worker = None
        logger.info("Diarization models loaded")
        self._launch_diarize_worker(file_path)

    def _on_diarize_model_error(self, message: str) -> None:
        """Diarization model load failed — non-fatal, show warning."""
        self._diarize_model_load_worker = None
        self._set_controls_busy(False)
        self.status_label.setText("Ready")
        self.statusBar().showMessage("File transcription complete", 3000)

        # Give specific advice for common errors.
        hint = ""
        msg_lower = message.lower()
        if "403" in message or "forbidden" in msg_lower or "gated" in msg_lower:
            hint = (
                "\n\nThis usually means:\n"
                "- You haven't accepted the model terms yet (visit the\n"
                "  links in Settings > HuggingFace Token), OR\n"
                "- Your token is a fine-grained token without the\n"
                '  "Access to public gated repos" permission.\n\n'
                "Try using a classic Read token instead."
            )
            # Clear the bad token so the user gets re-prompted.
            self.config.hf_token = ""
            self._diarize_engine = None
        elif "401" in message or "unauthorized" in msg_lower:
            hint = "\n\nYour token appears to be invalid. Please check it."
            self.config.hf_token = ""
            self._diarize_engine = None

        QMessageBox.warning(
            self,
            "Speaker Identification Unavailable",
            f"Could not load speaker diarization models:\n\n{message}{hint}\n\n"
            "The transcript is still available without speaker labels.",
        )
        logger.warning("Diarization model load failed: %s", message)

    def _launch_diarize_worker(self, file_path: str) -> None:
        """Create and start the DiarizeWorker."""
        if self._diarize_worker is not None:
            return  # Already running

        num_spk = self.spn_num_speakers.value()
        self._diarize_worker = DiarizeWorker(
            engine=self._diarize_engine,
            file_path=file_path,
            num_speakers=num_spk if num_spk > 0 else None,
            parent=self,
        )
        self._diarize_worker.diarization_ready.connect(self._on_diarization_ready)
        self._diarize_worker.progress.connect(self._on_model_progress)
        self._diarize_worker.error.connect(self._on_diarize_worker_error)
        self._diarize_worker.finished.connect(self._on_diarize_worker_finished)
        self._diarize_worker.start()

    def _on_diarization_ready(self, timeline: list) -> None:
        """Diarization complete — align speakers to segments and re-render."""
        self._speaker_timeline = timeline
        if not timeline:
            logger.info("No speakers detected in diarization")
            return

        # Build default speaker names
        speaker_ids = sorted({seg.speaker_id for seg in timeline})
        self._speaker_names = {
            sid: f"Speaker {sid + 1}" for sid in speaker_ids
        }

        self._align_speakers_to_segments()
        self._render_transcript()
        logger.info(
            "Diarization complete: %d speakers identified", len(speaker_ids)
        )

    def _on_diarize_worker_error(self, message: str) -> None:
        """Diarization failed — non-fatal."""
        self.statusBar().showMessage(
            "Speaker identification failed — transcript available without labels",
            5000,
        )
        logger.warning("Diarization failed: %s", message)
        # Re-enable the button so the user can retry.
        if self._current_file_path and self._segments:
            self.btn_diarize.setEnabled(True)

    def _on_diarize_worker_finished(self) -> None:
        """Clean up diarize worker and restore UI."""
        if self._diarize_worker is not None:
            self._diarize_worker.deleteLater()
            self._diarize_worker = None
        self._set_controls_busy(False)
        self.status_label.setText("Ready")
        self.statusBar().showMessage("Speaker identification complete", 3000)
        # Re-enable so user can re-run with different speaker count.
        if self._current_file_path and self._segments:
            self.btn_diarize.setEnabled(True)

    # ------------------------------------------------------------------
    # Speaker alignment & rendering
    # ------------------------------------------------------------------

    def _align_speakers_to_segments(self) -> None:
        """Assign each transcript block to its majority speaker.

        For every 30-second STT block, find the diarization speaker with
        the most overlap time and tag the whole block with that speaker.
        This avoids splitting sentences across speakers — each block
        stays intact with a single speaker label.
        """
        from gui.workers import FileTranscribeWorker

        seg_dur = FileTranscribeWorker.SEGMENT_SECONDS

        for i, seg in enumerate(self._segments):
            seg_start = i * seg_dur
            seg_end = (i + 1) * seg_dur

            # Accumulate overlap time per speaker.
            overlap_by_speaker: dict[int, float] = {}
            for ds in self._speaker_timeline:
                ov_start = max(seg_start, ds.start_sec)
                ov_end = min(seg_end, ds.end_sec)
                if ov_end > ov_start:
                    overlap = ov_end - ov_start
                    overlap_by_speaker[ds.speaker_id] = (
                        overlap_by_speaker.get(ds.speaker_id, 0.0) + overlap
                    )

            if overlap_by_speaker:
                # Assign the speaker with the most overlap time.
                best_speaker = max(
                    overlap_by_speaker, key=overlap_by_speaker.get  # type: ignore[arg-type]
                )
                seg["speaker"] = best_speaker

    def _render_transcript(self) -> None:
        """Re-render the transcript view with optional speaker colour labels."""
        self.transcript_view.clear()
        for seg in self._segments:
            safe_text = html.escape(seg["text"], quote=False)
            safe_ts = html.escape(seg["timestamp"], quote=False)

            speaker_id = seg.get("speaker")
            if speaker_id is not None and self._speaker_names:
                name = html.escape(
                    self._speaker_names.get(speaker_id, f"Speaker {speaker_id + 1}"),
                    quote=False,
                )
                color = SPEAKER_COLORS[speaker_id % len(SPEAKER_COLORS)]
                line = (
                    f'<span style="color: {color}; font-weight: bold;">'
                    f"[{name}]</span> [{safe_ts}] {safe_text}"
                )
            else:
                line = f"[{safe_ts}] {safe_text}"

            self.transcript_view.append(line)

        # Scroll to bottom
        scrollbar = self.transcript_view.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    # ------------------------------------------------------------------
    # Speaker rename context menu
    # ------------------------------------------------------------------

    def _on_transcript_context_menu(self, position: object) -> None:
        """Right-click on transcript — offer rename if a speaker label was clicked."""
        if not self._speaker_names:
            return

        cursor = self.transcript_view.cursorForPosition(position)  # type: ignore[arg-type]
        block_text = cursor.block().text()

        # Detect "[Speaker N]" pattern at the start of the line
        if not block_text.startswith("["):
            return
        close_bracket = block_text.find("]")
        if close_bracket == -1:
            return
        label_text = block_text[1:close_bracket]

        # Find which speaker_id this label belongs to
        target_id: int | None = None
        for sid, name in self._speaker_names.items():
            if name == label_text:
                target_id = sid
                break
        if target_id is None:
            return

        menu = QMenu(self)
        rename_action = menu.addAction(f'Rename "{label_text}"')
        chosen = menu.exec(self.transcript_view.mapToGlobal(position))  # type: ignore[arg-type]
        if chosen == rename_action:
            new_name, ok = QInputDialog.getText(
                self,
                "Rename Speaker",
                f'New name for "{label_text}":',
                text=label_text,
            )
            if ok and new_name.strip():
                self._speaker_names[target_id] = new_name.strip()
                self._render_transcript()

    # ------------------------------------------------------------------
    # Existing callbacks
    # ------------------------------------------------------------------

    def _on_source_changed(self, index: int) -> None:
        source = "loopback" if index == 1 else "microphone"
        self.config.audio_source = source
        logger.info("Audio source set to %s", source)

    def _on_font_size_changed(self, size: int) -> None:
        self.config.font_size = size
        self.transcript_view.setStyleSheet(f"font-size: {size}pt;")
        logger.info("Font size set to %d", size)

    def _on_llm_settings(self) -> None:
        """Open a dialog to configure the LLM server for SOAP formatting."""
        dlg = QDialog(self)
        dlg.setWindowTitle("LLM Server Settings")
        dlg.setMinimumWidth(450)
        layout = QVBoxLayout(dlg)

        layout.addWidget(QLabel(
            "<b>LLM server for SOAP note formatting</b><br>"
            "<i>Used when you click SOAPify in Medical mode.</i>"
        ))

        # Provider
        prov_layout = QHBoxLayout()
        prov_layout.addWidget(QLabel("Provider:"))
        prov_combo = QComboBox()
        prov_combo.addItems(["LM Studio", "Ollama"])
        if self.config.llm_provider == "ollama":
            prov_combo.setCurrentIndex(1)
        prov_layout.addWidget(prov_combo)
        layout.addLayout(prov_layout)

        # Endpoint
        ep_layout = QHBoxLayout()
        ep_layout.addWidget(QLabel("Endpoint:"))
        ep_input = QLineEdit()
        ep_input.setText(self.config.llm_endpoint)
        ep_layout.addWidget(ep_input)
        layout.addLayout(ep_layout)

        # Auto-fill endpoint when provider changes
        def _on_provider_changed(idx: int) -> None:
            current = ep_input.text()
            if idx == 0 and "11434" in current:
                ep_input.setText("http://localhost:1234/v1/chat/completions")
            elif idx == 1 and "1234" in current:
                ep_input.setText("http://localhost:11434/api/generate")
        prov_combo.currentIndexChanged.connect(_on_provider_changed)

        # Model
        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel("Model:"))
        model_input = QLineEdit()
        model_input.setText(self.config.llm_model)
        model_input.setPlaceholderText("e.g. medgemma-1.5-4b-it")
        model_layout.addWidget(model_input)
        layout.addLayout(model_layout)

        # Buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        layout.addWidget(buttons)

        if dlg.exec() == QDialog.DialogCode.Accepted:
            self.config.llm_provider = "ollama" if prov_combo.currentIndex() == 1 else "lm_studio"
            self.config.llm_endpoint = ep_input.text().strip()
            self.config.llm_model = model_input.text().strip()
            self.statusBar().showMessage("LLM settings saved", 3000)
            logger.info(
                "LLM settings: provider=%s, endpoint=%s, model=%s",
                self.config.llm_provider,
                self.config.llm_endpoint,
                self.config.llm_model,
            )

    def _on_soap_layout_settings(self) -> None:
        """Toggle SOAP layout between grid (2x2) and vertical."""
        current = self.config.soap_layout
        new_layout = "vertical" if current == "grid" else "grid"
        self.config.soap_layout = new_layout
        self.soap_view.set_layout_mode(new_layout)
        label = "Grid (2\u00d72)" if new_layout == "grid" else "Vertical"
        self.statusBar().showMessage(f"SOAP layout: {label}", 3000)

    def _on_hf_token_settings(self) -> None:
        """Open a dialog to set the HuggingFace access token."""
        token = self._prompt_hf_token(self.config.hf_token)
        if token is not None:
            self.config.hf_token = token
            if token:
                self.statusBar().showMessage("HuggingFace token saved", 3000)
            else:
                self.statusBar().showMessage("HuggingFace token cleared", 3000)

    def _prompt_hf_token(self, current: str = "") -> str | None:
        """Show a dialog with clickable links to get/set the HF token.

        Returns the token string if accepted, or None if cancelled.
        """
        dlg = QDialog(self)
        dlg.setWindowTitle("HuggingFace Token")
        dlg.setMinimumWidth(480)
        layout = QVBoxLayout(dlg)

        info = QLabel(
            "<p>Speaker identification uses "
            "<b>pyannote.audio</b>, which requires a free "
            "HuggingFace account and access token.</p>"
            "<p><b>Step 1:</b> Create or log in to your account and "
            'generate a <b>token</b> at:<br>'
            '<a href="https://huggingface.co/settings/tokens">'
            "https://huggingface.co/settings/tokens</a><br>"
            '<i>Recommended:</i> Use a <b>"Read"</b> token (classic). '
            "If using a fine-grained token, enable "
            '<b>"Access to public gated repos"</b>.</p>'
            "<p><b>Step 2:</b> Accept the model license terms "
            "(click each link, then click <b>Agree</b>):<br>"
            '&bull; <a href="https://huggingface.co/pyannote/speaker-diarization-3.1">'
            "pyannote/speaker-diarization-3.1</a><br>"
            '&bull; <a href="https://huggingface.co/pyannote/segmentation-3.0">'
            "pyannote/segmentation-3.0</a></p>"
            "<p><b>Step 3:</b> Paste your token below:</p>"
        )
        info.setWordWrap(True)
        info.setOpenExternalLinks(True)
        info.setTextFormat(Qt.TextFormat.RichText)
        layout.addWidget(info)

        token_input = QLineEdit()
        token_input.setPlaceholderText("hf_xxxxxxxxxxxxxxxxxxxx")
        token_input.setText(current)
        layout.addWidget(token_input)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        layout.addWidget(buttons)

        if dlg.exec() == QDialog.DialogCode.Accepted:
            return token_input.text().strip()
        return None
