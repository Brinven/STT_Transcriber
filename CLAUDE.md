# STT Transcriber — CLAUDE.md

## Project Identity
- **App name:** STT Transcriber
- **Purpose:** Offline, privacy-first desktop app for speech-to-text transcription of live meetings (Zoom/Teams) and WAV files, with an optional Medical Mode that formats transcripts into SOAP notes via a local LLM server.
- **Target platform:** Windows (primary), cross-platform possible later
- **PRD:** See `Stt_Transcriber20.2_plan.md` for full requirements

## Tech Stack (Resolved Decisions)

| Component | Choice | Notes |
|-----------|--------|-------|
| Language | Python 3.11+ | |
| UI Framework | **PySide6** | NOT Tkinter or PyQt5 (PRD was inconsistent) |
| General STT | **faster-whisper** | CTranslate2-based, 4x faster than openai-whisper on CPU |
| Medical STT | **MedASR** (`google/medasr`) | 105M-param Conformer (CTC), trained on medical dictation. 4.6-6.9% WER on medical audio. |
| Medical LLM | **Ollama / LM Studio API** | External local LLM server for SOAP formatting of MedASR transcript |
| Audio Capture | **sounddevice** | Supports both mic input and WASAPI loopback on Windows |
| Audio Processing | **numpy**, **soundfile** | Audio buffer handling and WAV I/O |
| Packaging | PyInstaller | Single executable distribution |
| Config Format | JSON | Stored in user's app data directory |
| Logging | Python `logging` module | |

### Key Libraries (requirements.txt)
- `PySide6` — Qt6 GUI framework
- `faster-whisper` — General-mode offline STT via CTranslate2
- `transformers` — HuggingFace transformers (required for MedASR, **>= 5.0.0** — MedASR `lasr_ctc` model type requires 5.x; we monkey-patch a bug in `LasrFeatureExtractor`)
- `torch` — PyTorch backend for MedASR inference
- `sounddevice` — Audio capture (mic + WASAPI loopback)
- `soundfile` — WAV file reading/writing
- `numpy` — Audio buffer handling
- `requests` — HTTP client for Ollama/LM Studio API
- `PyInstaller` — Build/packaging (dev dependency)
- `pytest`, `pytest-cov` — Testing (dev dependency)

### MedASR Model Details
- **Model:** `google/medasr` on HuggingFace (requires accepting Health AI Developer Foundations terms)
- **Architecture:** Conformer (CTC-based), 105M parameters
- **Input:** 16 kHz mono int16 audio
- **Inference:** `AutoModelForCTC` + `AutoProcessor` from transformers, or `pipeline("automatic-speech-recognition")`
- **Chunking:** Use `chunk_length_s=20, stride_length_s=2` for long audio
- **Performance:** 4.6% WER on radiology dictation, 5.8% on family medicine (with 6-gram LM)
- **Limitation:** English only, best on high-quality mic audio, may lack very recent medical terms
- **GPU:** Supports CUDA if available, falls back to CPU

## Architecture

### Directory Structure
```
STT_Transcriber/
├── CLAUDE.md
├── Stt_Transcriber20.2_plan.md   # PRD (reference only)
├── requirements.txt
├── main.py                        # Entry point
├── backend/
│   ├── __init__.py
│   ├── audio_capture.py           # Mic and WASAPI loopback recording
│   ├── stt_engine.py              # STT engine abstraction (faster-whisper + MedASR)
│   ├── medical_formatter.py       # Ollama/LM Studio SOAP formatting
│   ├── file_manager.py            # Export to TXT, CSV, MD
│   ├── config_manager.py          # JSON config read/write
│   └── paths.py                   # Centralized path resolution
├── gui/
│   ├── __init__.py
│   ├── app.py                     # MainWindow (QMainWindow)
│   ├── widgets/
│   │   ├── __init__.py
│   │   ├── audio_controls.py      # Record/Stop/Pause buttons, source selector
│   │   ├── transcript_view.py     # Scrollable transcript display
│   │   └── soap_view.py           # SOAP notes display (medical mode)
│   └── workers.py                 # QThread workers for audio, STT, LLM
├── resources/
│   └── icons/                     # UI icons (if any)
├── tests/
│   ├── __init__.py
│   ├── test_audio_capture.py
│   ├── test_stt_engine.py
│   ├── test_medical_formatter.py
│   ├── test_file_manager.py
│   └── data/                      # Test WAV files
└── stt_transcriber.spec           # PyInstaller spec
```

### Key Design Patterns
- **Backend/GUI separation** — Backend modules have zero PySide6 imports. GUI calls backend through workers.
- **QThread workers** — All long-running ops (recording, STT inference, LLM calls) run on worker threads, never on the main/UI thread.
- **Signals/slots** — Workers emit Qt signals for progress, results, and errors. GUI connects to these.
- **Centralized paths** — `backend/paths.py` resolves all file paths relative to a known app root. No scattered relative paths.

### Audio Pipeline — Dual STT Engine
```
                                         ┌─ General Mode ─→ faster-whisper ──→ transcript text
Mic/Loopback → sounddevice → 16kHz mono ─┤
                                         └─ Medical Mode ─→ MedASR ──→ medical transcript ─→ Ollama/LM Studio → SOAP notes
```

- Audio is captured at the device's native sample rate and resampled to **16 kHz mono** (both engines expect this).
- **General Mode:** Audio chunks (~5-10s) are fed to faster-whisper incrementally for live transcription.
- **Medical Mode:** Audio chunks are fed to MedASR (`google/medasr`) which is optimized for medical terminology. The resulting transcript is then optionally sent to Ollama/LM Studio for SOAP formatting.
- For WAV file ingestion, the file is read, converted to 16 kHz mono if needed, and processed in segments.
- `stt_engine.py` provides a common interface (`transcribe(audio) -> str`) with two backend implementations, selected by the current mode.

### Medical Mode
Medical mode has two stages:

**Stage 1 — Medical STT (MedASR)**
- Uses `google/medasr` (Conformer CTC, 105M params) instead of faster-whisper.
- Model is loaded on first use (lazy loading) — it stays in memory while medical mode is active.
- When switching back to general mode, the MedASR model can be unloaded to free memory.
- MedASR produces a more accurate transcript for medical terminology than general-purpose Whisper.

**Stage 2 — SOAP Formatting (Ollama/LM Studio)**
- Requires a running **Ollama** or **LM Studio** server on localhost.
- User configures the API endpoint URL and model name in settings.
- The MedASR transcript is sent to the LLM with a SOAP formatting prompt.
- Response is parsed and displayed in structured SOAP sections (Subjective, Objective, Assessment, Plan).
- SOAP formatting is triggered explicitly by the user ("SOAPify" button), not automatically.
- If the LLM server is unreachable, show a clear error — do NOT fall back silently.
- The raw medical transcript is always available even without SOAP formatting.

### Config Structure (`config.json`)
```json
{
  "mode": "general",
  "audio_source": "microphone",
  "audio_device_index": null,
  "whisper_model_size": "base",
  "medasr_device": "auto",
  "llm_endpoint": "http://localhost:11434/api/generate",
  "llm_model": "medllama2",
  "llm_provider": "ollama",
  "export_directory": "",
  "font_size": 12
}
```

- `medasr_device`: `"auto"` (use CUDA if available, else CPU), `"cpu"`, or `"cuda"`

## Coding Conventions

### General
- Follow the same patterns as the ThothAI project (sibling in this monorepo)
- Use type hints on all function signatures
- Use `logging` module — no `print()` for diagnostics
- Use `pathlib.Path` for all file path operations
- Use `html.escape()` when inserting any dynamic text into Qt HTML views
- Anchor all paths via `backend/paths.py` — no relative paths scattered in code

### Threading
- Never access GUI widgets from worker threads — use signals only
- Workers must be cancellable (check a flag or use `QThread.isInterruptionRequested()`)
- Protect shared state with `QMutex` or `threading.Lock` if needed
- Don't overwrite a running worker — check if one is active before starting another

### Error Handling
- Catch exceptions in workers and emit error signals (don't let them crash silently)
- Show user-facing errors in a `QMessageBox` with actionable text
- Log full tracebacks to the log file
- App should remain functional after recoverable errors (no restart required)

### Security & Privacy
- All STT processing is local and offline (both faster-whisper and MedASR run on-device)
- LLM calls go to localhost only (Ollama/LM Studio)
- No external network requests during normal operation
- **One-time exception:** MedASR and faster-whisper models are downloaded from HuggingFace on first use, then cached locally. User should be informed of this download.
- Temp files are cleaned up after processing
- Export files default to the user's chosen directory

## Build & Run

### Development Setup
```bash
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
python main.py
```

### Running Tests
```bash
pytest tests/ --cov=backend --cov=gui -v
```

### Building Executable
```bash
pyinstaller stt_transcriber.spec
```

## Keyboard Shortcuts
- **Space** — Start/Stop recording (when transcript view is not focused)
- **Ctrl+E** — Export transcript
- **Ctrl+M** — Toggle Medical Mode

## Known PRD Inconsistencies (Resolved)
1. **UI Framework**: PRD says Tkinter in architecture, PyQt5 in milestones → **Resolved: PySide6**
2. **STT Engine**: PRD mentions both Vosk and Whisper → **Resolved: faster-whisper**
3. **Medical LLM**: PRD conflicts between llama-cpp-python, regex, Ollama/LM Studio → **Resolved: Ollama/LM Studio API**
4. **Audio sample rate**: PRD says 16 kHz in ACs, 48 kHz in release checklist → **Resolved: capture at native rate, resample to 16 kHz for STT**
5. **Python version**: PRD says 3.12 in one place, 3.10+ in another → **Resolved: 3.11+**
6. **WER metric**: PRD says "≥ 90% WER" which would be terrible (WER = error rate, lower is better) → they meant ≥ 90% accuracy (≤ 10% WER)
7. **"Synchronous function calls"** in architecture → **Resolved: async via QThread workers** (required for responsive UI)
8. **Milestone/backlog numbering mismatch** → Ignore backlog milestone numbers; follow the 5-milestone plan in order
